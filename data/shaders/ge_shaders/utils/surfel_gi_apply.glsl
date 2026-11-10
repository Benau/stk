// The irradiance of the pixels from the surfels (GEVulkanSurfelGI, see
// utils/surfel_gi.glsl), shared by surfel_gi_apply.comp, surfel_gi_gather.comp
// and surfel_gi_spawn.comp, which all run on the pixels of the viewport of a
// camera (pc.m_rect).
//
// The irradiance of the surface of a pixel is the weighted mean of the surfels
// which touch it, in the 2x2x2 cells nearest to it (by the distance to their
// center and the normal), its alpha is how much they cover it (the lighting
// pass mixes it with the irradiance of the environment by it). A surfel which
// is used is kept alive.
//
// A pixel which they don't cover enough (a kart, or a hole between them) gets
// the irradiance of the surfels within a cell around it too, the ones which it
// is in front of, like an irradiance probe made of them (a kart in a room is
// lit by the surfels of its floor and its walls, not by the sky). They are in
// the 3x3x3 cells around the cell of the pixel: the 2x2x2 nearest cells only
// have all the surfels within half a cell, a farther one would be in them on
// one side of the border of a cell and not on the other, which is a seam on
// the kart, and half a cell is too short (the ground under a kart is hidden,
// it has no surfels). The ones which face the same way weigh the most, but all
// of them weigh some, so a side which no surfel faces still gets the color of
// the room. It doesn't keep them alive or stop a spawn.
//
// Every 8x8 tile of the viewport spawns a surfel at its least covered pixel if
// the surfels don't cover it enough and its cell is not full (with the
// surfels spawned in this frame), only on the static scene: the ray from the
// camera to the pixel must hit the static BLAS there (a kart or anything
// moving is not static, and the surfels never move).
//
// With the frame cache (SURFEL_CACHE_FRAME), the irradiance of a pixel whose
// surfels converged and cover it doesn't change anymore: it's in the history
// (u_history_out), and the next frame reuses it if its surface is at the same
// pixel of the history (the position seen by the camera of the last frame, at
// the same distance from it and with the same normal), so it doesn't look for
// its surfels. A tile looks for them in one of SURFEL_HISTORY_PERIOD frames
// anyway (they take turns), which keeps them alive and follows their radius.
// A pixel which reuses its irradiance doesn't spawn, it's covered.
//
// Set 0 is the one of GEVulkanRTScene, set 1 the camera and the global light
// of the viewport, set 2 the g-buffer, the output, the surfels and the
// history, and set 3 the mesh textures (of the alpha tested geometries). The
// shader which includes it enables GL_EXT_ray_query

#define SURFEL_SET 2
#define MESH_TEXTURES_SET 3

layout(set = 2, binding = 0) uniform sampler2D u_depth;
layout(set = 2, binding = 1) uniform sampler2D u_normal;
layout(set = 2, binding = 2, rgba16f) uniform writeonly image2D u_out;
// The history of the last frame and the one of this frame (the next frame
// reads it), the irradiance (packSurfelIrradiance) and the surface
// (packSurfelHistorySurface) of a pixel, 0 if it can't be reused
layout(set = 2, binding = 12, rg32ui) uniform readonly uimage2D u_history_in;
layout(set = 2, binding = 13, rg32ui) uniform writeonly uimage2D u_history_out;

// The pixels which surfel_gi_apply.comp couldn't reuse (packSurfelPixel), for
// surfel_gi_gather.comp, which is dispatched with m_dispatch (it's the number
// of groups of 64 of them)
layout(std430, set = 2, binding = 9) buffer SurfelMissList
{
    uint m_count;
    // The indirect dispatch (x, y, z groups)
    uint m_dispatch[3];
    uint m_pixels[];
} u_misses;

// The spawn of every 8x8 tile of the viewport (getSurfelSpawnKey), the least
// covered pixel of the tile which needs a surfel, 0xFFFFFFFF for none, for
// surfel_gi_spawn.comp
layout(std430, set = 2, binding = 10) buffer SurfelTileSpawns
{
    uint u_tile_spawns[];
};

#include "camera.glsl"
#include "constants_utils.glsl"
#include "unproject_position.glsl"
#include "../../utils/decodeNormal.frag"
#include "get_vertex_color.glsl"
#include "sample_mesh_texture.glsl"
#include "rt_utils.glsl"
#include "rt_scene.glsl"
#include "surfel_gi.glsl"

// A pixel less covered than it gets a new surfel
const float SURFEL_SPAWN_COVERAGE = 0.5;
// Chance of a tile to spawn in a step (1/60 s, the chance of a frame is the
// one of its steps), so the tiles around a hole don't all spawn in the same
// frame (the new surfels are in the grid in the next one)
const float SURFEL_SPAWN_CHANCE = 0.25;
// With the frame cache, the steps of samples which all the surfels of a pixel
// need before its irradiance is in the history: it changes little after them
// (a pixel which reuses it looks for its surfels again in
// SURFEL_HISTORY_PERIOD frames)
const float SURFEL_CACHE_MIN_SAMPLES = 64.0;
// With the frame cache, the coverage of the surfels with enough samples around
// a new surfel which it starts with their samples, see spawnSurfel()
const float SURFEL_SEED_COVERAGE = 0.25;

// The surface of a pixel in the g-buffer
struct SurfelPixel
{
    vec3 m_position;
    vec3 m_normal;
    // As it's stored in the g-buffer (for the history)
    vec2 m_encoded_normal;
    // To the closest camera (the level of the grid)
    float m_camera_distance;
};

// The surfels which touch a pixel
struct SurfelGather
{
    // The irradiance of the surfels weighted by their weight and confidence,
    // and the sum of the weights
    vec3 m_irradiance;
    float m_weight_sum;
    // The sum of their weights without the confidence
    float m_coverage;
    // The fewest steps of samples of them
    float m_min_samples;
    // The cell of the pixel is full
    bool m_cell_full;
};

// ----------------------------------------------------------------------------
// False for the sky (reverse z, depth 0)
bool loadSurfelPixel(ivec2 pixel, out SurfelPixel p)
{
    p.m_position = vec3(0.0);
    p.m_normal = vec3(0.0, 1.0, 0.0);
    p.m_encoded_normal = vec2(0.0);
    p.m_camera_distance = 0.0;
    float depth = texelFetch(u_depth, pixel, 0).x;
    if (depth == 0.0)
        return false;
    vec3 xpos = getPosFromUVDepth(vec3(vec2(pixel) + 0.5, depth),
        u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
    p.m_position = (u_camera.m_inverse_view_matrix * vec4(xpos, 1.0)).xyz;
    p.m_encoded_normal = texelFetch(u_normal, pixel, 0).xy;
    p.m_normal = DecodeNormal(p.m_encoded_normal);
    p.m_camera_distance = getSurfelCameraDistance(p.m_position);
    return true;
}   // loadSurfelPixel

// ----------------------------------------------------------------------------
// The surfels of the 2x2x2 nearest cells which touch the pixel, they are kept
// alive
SurfelGather gatherSurfels(SurfelPixel p)
{
    SurfelGather g;
    g.m_irradiance = vec3(0.0);
    g.m_weight_sum = 0.0;
    g.m_coverage = 0.0;
    g.m_min_samples = SURFEL_CACHED_MAX_SAMPLES;
    uint level = getSurfelLevel(p.m_camera_distance);
    float cell_size = getSurfelCellSize(level);
    g.m_cell_full = u_cell_count[getSurfelCell(p.m_position, level)] >=
        SURFEL_CELL_CAPACITY;
    ivec3 first = getSurfelNearestCells(p.m_position, cell_size);
    for (int c = 0; c < 8; c++)
    {
        uint cell = getSurfelCell(first + ivec3(c & 1, (c >> 1) & 1, c >> 2),
            level);
        uint count = min(u_cell_count[cell], SURFEL_CELL_CAPACITY);
        for (uint i = 0u; i < count; i++)
        {
            uint id = u_cell_surfels[cell * SURFEL_CELL_CAPACITY + i];
            Surfel surfel = u_surfels[id];
            vec3 offset = p.m_position - surfel.m_position;
            float offset_distance = length(offset);
            float plane_distance = dot(offset, surfel.m_normal);
            if (offset_distance >= surfel.m_radius)
                continue;
            float facing = dot(p.m_normal, surfel.m_normal);
            // Not on the same side of a thin wall, or on another plane
            if (facing <= 0.0 || abs(plane_distance) > surfel.m_radius * 0.5)
                continue;
            float falloff = 1.0 - offset_distance / surfel.m_radius;
            float weight = falloff * falloff * facing;
            g.m_coverage += weight;
            g.m_min_samples = min(g.m_min_samples, surfel.m_samples);
            // A new surfel has a noisy mean
            float confidence =
                min(surfel.m_samples / SURFEL_CONFIDENT_SAMPLES, 1.0);
            g.m_irradiance += surfel.m_irradiance * (weight * confidence);
            g.m_weight_sum += weight * confidence;
            u_surfels[id].m_last_seen = pc.m_time;
        }
    }
    return g;
}   // gatherSurfels

// ----------------------------------------------------------------------------
// The irradiance of the surfels within a cell of a pixel which they don't
// cover (alpha is how much of it there is), see above. min_samples is lowered
// to the fewest samples of them
vec4 getNearbyIrradiance(SurfelPixel p, inout float min_samples)
{
    uint level = getSurfelLevel(p.m_camera_distance);
    float cell_size = getSurfelCellSize(level);
    ivec3 center = ivec3(floor(p.m_position / cell_size));
    vec3 irradiance = vec3(0.0);
    float weight_sum = 0.0;
    for (int c = 0; c < 27; c++)
    {
        uint cell = getSurfelCell(center + ivec3(c % 3, (c / 3) % 3, c / 9) -
            1, level);
        uint count = min(u_cell_count[cell], SURFEL_CELL_CAPACITY);
        for (uint i = 0u; i < count; i++)
        {
            Surfel surfel =
                u_surfels[u_cell_surfels[cell * SURFEL_CELL_CAPACITY + i]];
            vec3 offset = p.m_position - surfel.m_position;
            float offset_distance = length(offset);
            // Behind it is the other side of its wall
            if (offset_distance >= cell_size ||
                dot(offset, surfel.m_normal) < -0.05 * cell_size)
                continue;
            float facing = 0.5 + 0.5 * dot(p.m_normal, surfel.m_normal);
            float falloff = 1.0 - offset_distance / cell_size;
            // A new surfel has a noisy mean
            float weight = falloff * falloff * facing * facing *
                min(surfel.m_samples / SURFEL_CONFIDENT_SAMPLES, 1.0);
            min_samples = min(min_samples, surfel.m_samples);
            irradiance += surfel.m_irradiance * weight;
            weight_sum += weight;
        }
    }
    if (weight_sum == 0.0)
        return vec4(0.0);
    return vec4(irradiance / weight_sum, clamp(weight_sum, 0.0, 1.0));
}   // getNearbyIrradiance

// ----------------------------------------------------------------------------
// The output of a pixel from its surfels (and the ones around it if they don't
// cover it), cacheable is set if it can be in the history: the history has no
// alpha (a pixel which reuses it is covered), and only the irradiance of
// surfels with enough samples doesn't change much anymore
vec4 resolveSurfelPixel(SurfelPixel p, SurfelGather g, out bool cacheable)
{
    vec4 gi = vec4(0.0);
    if (g.m_weight_sum > 0.0)
    {
        gi = vec4(g.m_irradiance / g.m_weight_sum,
            clamp(g.m_weight_sum * 2.0, 0.0, 1.0));
    }
    float min_samples = g.m_min_samples;
    if (gi.a < 1.0)
    {
        vec4 nearby = getNearbyIrradiance(p, min_samples);
        // The surfels on the surface first
        if (nearby.a > 0.0)
            gi = vec4(mix(nearby.rgb, gi.rgb, gi.a), max(gi.a, nearby.a));
    }
    cacheable = gi.a >= 1.0 && min_samples >= SURFEL_CACHE_MIN_SAMPLES;
    return gi;
}   // resolveSurfelPixel

// ----------------------------------------------------------------------------
// 9 bits for every channel and a shared exponent of 5 bits (like RGB9E5), so
// the history is small
uint packSurfelIrradiance(vec3 rgb)
{
    // The largest value of the exponent (16) with 511 in the bits
    rgb = clamp(rgb, vec3(0.0), vec3(65408.0));
    float max_channel = max(max(rgb.r, rgb.g), rgb.b);
    // 2 ^ (exponent - 1) <= max_channel < 2 ^ exponent, in [-15, 16]
    int exponent = max(int(floor(log2(max(max_channel, 1.0e-10)))), -16) + 1;
    float scale = exp2(float(exponent - 9));
    // Rounded up to 512
    if (floor(max_channel / scale + 0.5) >= 512.0)
    {
        exponent++;
        scale *= 2.0;
    }
    uvec3 bits = uvec3(min(floor(rgb / scale + 0.5), vec3(511.0)));
    return bits.r | (bits.g << 9u) | (bits.b << 18u) |
        (uint(exponent + 15) << 27u);
}   // packSurfelIrradiance

// ----------------------------------------------------------------------------
vec3 unpackSurfelIrradiance(uint packed)
{
    return vec3(uvec3(packed, packed >> 9u, packed >> 18u) & 511u) *
        exp2(float(int(packed >> 27u) - 15 - 9));
}   // unpackSurfelIrradiance

// ----------------------------------------------------------------------------
// The distance to the camera (half float) and the normal of the g-buffer (8
// bits for both of its components), never 0 (the camera is not on the
// surface)
uint packSurfelHistorySurface(float camera_distance, vec2 encoded_normal)
{
    uvec2 normal = uvec2(round(clamp(encoded_normal, 0.0, 1.0) * 255.0));
    return (packHalf2x16(vec2(camera_distance, 0.0)) & 0xFFFFu) |
        (normal.x << 16u) | (normal.y << 24u);
}   // packSurfelHistorySurface

// ----------------------------------------------------------------------------
// The irradiance of a position in the history, if the camera of the last frame
// saw it at a pixel which can be reused: the surface there is the same, it has
// the same distance from that camera (it's not hidden by something closer, or
// the pixel is the one of another surface behind it) and the same normal (the
// pixel is not on the other side of a corner)
bool getSurfelHistory(SurfelPixel p, out vec3 irradiance)
{
    irradiance = vec3(0.0);
    SurfelPrevCamera camera = u_prev_cameras[pc.m_viewport_index];
    vec4 clip = camera.m_projection_view_matrix * vec4(p.m_position, 1.0);
    if (clip.w <= 0.0)
        return false;
    // The inverse of getPosFromUVDepth
    vec2 uv = camera.m_viewport.xy + (clip.xy / clip.w * 0.5 + 0.5) *
        camera.m_viewport.zw;
    ivec2 pixel = ivec2(floor(uv));
    if (any(lessThan(pixel, pc.m_rect.xy)) ||
        any(greaterThanEqual(pixel, pc.m_rect.zw)))
        return false;
    uvec2 history = imageLoad(u_history_in, pixel).xy;
    if (history.y == 0u)
        return false;
    float history_distance = unpackHalf2x16(history.y & 0xFFFFu).x;
    float camera_distance = distance(p.m_position, camera.m_position.xyz);
    if (abs(history_distance - camera_distance) >
        camera_distance * 0.01 + 0.01)
        return false;
    vec2 history_normal =
        vec2(uvec2(history.y >> 16u, history.y >> 24u) & 255u) / 255.0;
    if (dot(DecodeNormal(history_normal), p.m_normal) < 0.9)
        return false;
    irradiance = unpackSurfelIrradiance(history.x);
    return true;
}   // getSurfelHistory

// ----------------------------------------------------------------------------
// The tiles take turns to look for their surfels even if their pixels can be
// reused (see above)
bool isSurfelHistoryTurn(ivec2 pixel)
{
    uvec2 tile = uvec2(pixel - pc.m_rect.xy) / 8u;
    return (tile.x + tile.y * 3u + pc.m_frame) % SURFEL_HISTORY_PERIOD != 0u;
}   // isSurfelHistoryTurn

// ----------------------------------------------------------------------------
// The output and the history (with the frame cache) of a pixel
void storeSurfelPixel(ivec2 pixel, SurfelPixel p, vec4 gi, bool cacheable)
{
    imageStore(u_out, pixel, gi);
    if ((pc.m_cache & SURFEL_CACHE_FRAME) == 0u)
        return;
    uvec2 history = uvec2(0u);
    if (cacheable)
    {
        history = uvec2(packSurfelIrradiance(gi.rgb),
            packSurfelHistorySurface(distance(p.m_position,
            u_camera.m_inverse_view_matrix[3].xyz), p.m_encoded_normal));
    }
    imageStore(u_history_out, pixel, uvec4(history, 0u, 0u));
}   // storeSurfelPixel

// ----------------------------------------------------------------------------
// The order of the pixels of a tile which need a surfel, the least covered one
// first: the coverage (rounded, a float which is not negative has the same
// order as its bits) and the pixel in the tile. 0xFFFFFFFF if it doesn't need
// one
uint getSurfelSpawnKey(ivec2 pixel, SurfelPixel p, SurfelGather g)
{
    if (g.m_cell_full || g.m_coverage >= SURFEL_SPAWN_COVERAGE ||
        p.m_camera_distance >= SURFEL_MAX_DISTANCE)
        return 0xFFFFFFFFu;
    uvec2 in_tile = uvec2(pixel - pc.m_rect.xy) & 7u;
    return (floatBitsToUint(g.m_coverage) & ~63u) | (in_tile.y * 8u) |
        in_tile.x;
}   // getSurfelSpawnKey

// ----------------------------------------------------------------------------
// In u_misses
uint packSurfelPixel(ivec2 pixel)
{
    return uint(pixel.x) | (uint(pixel.y) << 16u);
}   // packSurfelPixel

// ----------------------------------------------------------------------------
ivec2 unpackSurfelPixel(uint packed)
{
    return ivec2(packed & 0xFFFFu, packed >> 16u);
}   // unpackSurfelPixel

// ----------------------------------------------------------------------------
// No surfel
SurfelGather getEmptySurfelGather()
{
    SurfelGather g;
    g.m_irradiance = vec3(0.0);
    g.m_weight_sum = 0.0;
    g.m_coverage = 0.0;
    g.m_min_samples = SURFEL_CACHED_MAX_SAMPLES;
    g.m_cell_full = false;
    return g;
}   // getEmptySurfelGather

// ----------------------------------------------------------------------------
// The tile of a pixel in u_tile_spawns
uint getSurfelTile(ivec2 pixel)
{
    uvec2 tile = uvec2(pixel - pc.m_rect.xy) / 8u;
    uint tiles_x = uint(pc.m_rect.z - pc.m_rect.x + 7) / 8u;
    return tile.y * tiles_x + tile.x;
}   // getSurfelTile

// ----------------------------------------------------------------------------
// Whether the surface at the end of the ray from the camera is in the static
// scene, and the camera is not inside of a mesh of it: then the g-buffer has
// what is behind the faces around it (their back is culled), and a surfel there
// would be inside of the mesh too, so all its rays see the back of a triangle
// and it's black. The ray from the camera must not cross the back of a triangle
// before the surface (or at it, the mesh can touch the surface)
bool isStaticSurface(vec3 camera, vec3 position)
{
    vec3 to_position = position - camera;
    float position_distance = length(to_position);
    vec3 direction = to_position / position_distance;
    float ray_max = position_distance * 1.02 + 0.1;
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(ray_query, u_tlas, gl_RayFlagsNoneEXT,
        RT_MASK_STATIC, camera, 0.0, direction, ray_max);
    RT_PROCEED(ray_query)
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT ||
        !rayQueryGetIntersectionFrontFaceEXT(ray_query, true))
        return false;
    float hit = rayQueryGetIntersectionTEXT(ray_query, true);
    if (abs(hit - position_distance) >= position_distance * 0.02 + 0.05)
        return false;

    rayQueryEXT back_query;
    rayQueryInitializeEXT(back_query, u_tlas,
        gl_RayFlagsTerminateOnFirstHitEXT |
        gl_RayFlagsCullFrontFacingTrianglesEXT, RT_MASK_STATIC, camera, 0.0,
        direction, ray_max);
    RT_PROCEED(back_query)
    return rayQueryGetIntersectionTypeEXT(back_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT;
}   // isStaticSurface

// ----------------------------------------------------------------------------
// The chance of the frame of a tile to spawn at the pixel of its key
bool isSurfelSpawnTurn(ivec2 pixel)
{
    uint seed = (uint(pixel.x) * 1973u) ^ (uint(pixel.y) * 9277u) ^
        (pc.m_frame * 26699u);
    return getSurfelRandom(seed) <
        1.0 - pow(1.0 - SURFEL_SPAWN_CHANCE, pc.m_steps);
}   // isSurfelSpawnTurn

// ----------------------------------------------------------------------------
// The spawn of a tile at the pixel of its key (getSurfelSpawnKey) if it's its
// turn (isSurfelSpawnTurn)
void spawnSurfel(SurfelPixel p, SurfelGather g)
{
    if (u_pool.m_free_count <= 0)
        return;
    if (!isStaticSurface(u_camera.m_inverse_view_matrix[3].xyz,
        p.m_position) || !trySpawnSurfelInCell(getSurfelCell(p.m_position,
        getSurfelLevel(p.m_camera_distance))))
        return;
    int free_count = atomicAdd(u_pool.m_free_count, -1);
    if (free_count <= 0)
    {
        atomicAdd(u_pool.m_free_count, 1);
        return;
    }
    uint id = u_pool.m_free[free_count - 1];
    Surfel surfel;
    surfel.m_position = p.m_position;
    surfel.m_radius = getSurfelRadius(p.m_camera_distance);
    surfel.m_normal = p.m_normal;
    surfel.m_last_seen = pc.m_time;
    // What is already there, so it doesn't start black
    surfel.m_irradiance = g.m_weight_sum > 0.0 ?
        g.m_irradiance / g.m_weight_sum : vec3(0.0);
    surfel.m_samples = 0.0;
    // With the frame cache, a surfel which spawns between surfels with enough
    // samples starts with their mean (its irradiance above) and as many
    // samples as them: the pixels around it would not be in the history until
    // it has enough of its own, and a camera which moves keeps spawning
    // smaller surfels between the ones of its surfaces (they shrink when it
    // comes closer). A spawn is less covered than SURFEL_SPAWN_COVERAGE, so
    // it only needs a few of them around it
    if ((pc.m_cache & SURFEL_CACHE_FRAME) != 0u &&
        g.m_coverage >= SURFEL_SEED_COVERAGE)
    {
        surfel.m_samples = min(g.m_min_samples, SURFEL_CACHE_MIN_SAMPLES);
    }
    u_surfels[id] = surfel;
}   // spawnSurfel
