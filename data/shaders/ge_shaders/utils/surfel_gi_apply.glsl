#ifdef SUBGROUP_BALLOT
#extension GL_KHR_shader_subgroup_ballot : require
#endif

#define SURFEL_SET 2
#define MESH_TEXTURES_SET 3

layout(set = 2, binding = 0) uniform sampler2D u_depth;
layout(set = 2, binding = 1) uniform sampler2D u_normal;
#include "camera.glsl"
#include "constants_utils.glsl"
#include "unproject_position.glsl"
#include "../../utils/decodeNormal.frag"
#include "get_vertex_color.glsl"
#include "sample_mesh_texture.glsl"
#include "rt_utils.glsl"
#include "rt_scene.glsl"
#include "surfel_gi.glsl"
#include "surfel_gi_volume.glsl"
#include "surfel_gi_cache.glsl"
// A pixel less covered than it gets a new surfel
const float SURFEL_SPAWN_COVERAGE = 0.75;
// Chance of a tile to spawn in a step (1/60 s, the chance of a frame is the
// one of its steps), so the tiles around a hole don't all spawn in the same
// frame (the new surfels are in the grid in the next one)
const float SURFEL_SPAWN_CHANCE = 0.25;
const float SURFEL_SEED_COVERAGE = 0.25;

// The surface of a pixel in the g-buffer
struct SurfelPixel
{
    vec3 m_position;
    vec3 m_normal;
    // To the closest camera (the level of the grid)
    float m_camera_distance;
    float m_error;
};

// The surfels which touch a pixel
struct SurfelGather
{
    // The irradiance of the surfels weighted by their weight and confidence,
    // and the sum of the weights
    vec3 m_irradiance;
    float m_weight_sum;
    float m_origin_distance_sum;
    // The sum of their weights without the confidence
    float m_coverage;
    // The fewest steps of samples of them
    float m_min_samples;
};

// ----------------------------------------------------------------------------
// False for the sky (reverse z, depth 0)
bool loadSurfelPixel(ivec2 pixel, out SurfelPixel p)
{
    p.m_position = vec3(0.0);
    p.m_normal = vec3(0.0, 1.0, 0.0);
    p.m_camera_distance = 0.0;
    p.m_error = 0.0;
    float depth = texelFetch(u_depth, pixel, 0).x;
    if (depth == 0.0)
        return false;
    vec3 xpos = getPosFromUVDepth(vec3(vec2(pixel) + 0.5, depth),
        u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
    p.m_position = (u_camera.m_inverse_view_matrix * vec4(xpos, 1.0)).xyz;
    p.m_normal = DecodeNormal(texelFetch(u_normal, pixel, 0).xy);
    p.m_camera_distance = getSurfelCameraDistance(p.m_position);
    vec3 camera = u_camera.m_inverse_view_matrix[3].xyz;
    float view_distance = length(p.m_position - camera);
    float quantum = max(pc.m_depth_step, uintBitsToFloat(floatBitsToUint(depth) + 1u) - depth);
    p.m_error = view_distance * quantum / max(depth, quantum) +
        (length(p.m_position) + length(camera) + view_distance) * 4.76837e-7;
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
    g.m_origin_distance_sum = 0.0;
    g.m_coverage = 0.0;
    g.m_min_samples = SURFEL_MAX_SAMPLES;
    uint level = getSurfelLevel(p.m_camera_distance);
    float cell_size = getSurfelCellSize(level);
    ivec3 first = getSurfelNearestCells(p.m_position, cell_size);
    // The loops are bound by the surfels which they go through (few of them
    // touch the pixel), so the cells which are too far to have one are
    // skipped: near the cameras the surfels are much smaller than half a cell
    float max_radius = getSurfelMaxRadius(p.m_camera_distance);
    for (int c = 0; c < 8; c++)
    {
        ivec3 cell_position = first + ivec3(c & 1, (c >> 1) & 1, c >> 2);
        if (getSurfelCellDistance2(p.m_position, cell_position, cell_size) >=
            max_radius * max_radius)
            continue;
        uint lane_cell = getSurfelCell(cell_position, level);
        do
        {
#ifdef SUBGROUP_BALLOT
            uint cell = subgroupBroadcastFirst(lane_cell);
            if (cell != lane_cell)
                continue;
#else
            uint cell = lane_cell;
#endif
            uint count = u_cell_count[cell];
            uint list = cell * (SURFEL_CELL_CAPACITY / 4u);
            uint link = ~0u;
            for (uint i = 0u; i < count; i += 4u)
            {
                // The ids 4 at a time, the loop is bound by the loads which wait
                // for the one before (the id, then the surfel of the id)
                uvec4 ids;
                if (i < SURFEL_CELL_CAPACITY)
                    ids = u_cell_surfels4[list + i / 4u];
                else
                    for (uint entry = 0u; entry < min(count - i, 4u); ++entry)
                        ids[entry] = nextCellSurfel(cell, i + entry, link);
                for (uint j = 0u; j < 4u && i + j < count; j++)
                {
                    uint id = ids[j];
                    // Most of them don't touch the pixel, the rest of a surfel
                    // is only read if it's close enough
                    vec4 sphere = u_surfels_vec4[id * 3u];
                    vec3 offset = p.m_position - sphere.xyz;
                    float radius = sphere.w;
                    float offset_distance = length(offset);
                    if (offset_distance >= radius)
                        continue;
                    // The rest together (the normal and the last time seen, the
                    // irradiance and the samples)
                    vec4 normal_seen = u_surfels_vec4[id * 3u + 1u];
                    vec4 irradiance_samples = u_surfels_vec4[id * 3u + 2u];
                    vec3 normal = normal_seen.xyz;
                    float plane_distance = dot(offset, normal);
                    float facing = dot(p.m_normal, normal);
                    // Not on the same side of a thin wall, or on another plane
                    if (facing <= 0.0 || abs(plane_distance) > radius * 0.5)
                        continue;
                    float samples = irradiance_samples.w;
                    float radial = offset_distance / radius;
                    float placement = 1.0 - radial;
                    g.m_coverage += placement * placement * facing;
                    float falloff = 1.0 - radial * radial;
                    float weight = falloff * falloff * facing;
                    g.m_min_samples = min(g.m_min_samples, lighting[id].anchor.w != 0u ? 0.0 : samples);
                    // A new surfel has a noisy mean
                    float confidence =
                        min(samples / SURFEL_CONFIDENT_SAMPLES, 1.0);
                    g.m_irradiance += irradiance_samples.xyz *
                        (weight * confidence);
                    g.m_weight_sum += weight * confidence;
                    g.m_origin_distance_sum += length(offset - normal * lighting[id].bias) *
                        (weight * confidence);
                    // Most surfels are used by many pixels, the ones of the
                    // groups after the first which used it in the frame don't
                    // write it again
                    if (floatBitsToUint(normal_seen.w) != pc.m_time)
                        u_surfels[id].m_last_seen = pc.m_time;
                }
            }
            break;
        } while (true);
    }
    return g;
}   // gatherSurfels

// ----------------------------------------------------------------------------
uint getSurfelSpawnKey(uint seed, SurfelGather g)
{
    if (g.m_coverage >= SURFEL_SPAWN_COVERAGE)
        return 0xFFFFFFFFu;
    float sample_value = max(getSurfelRandom(seed), 1.0 / 16777216.0);
    float priority = -log2(sample_value) / (SURFEL_SPAWN_COVERAGE - g.m_coverage);
    return (floatBitsToUint(priority) & ~63u) | gl_LocalInvocationIndex;
}   // getSurfelSpawnKey

// ----------------------------------------------------------------------------
bool findAttachment(SurfelPixel surface, out vec3 position, out vec3 normal, out uvec4 anchor)
{
    vec3 camera = u_camera.m_inverse_view_matrix[3].xyz;
    vec3 delta = surface.m_position - camera;
    float depth = length(delta);
    vec3 direction = delta / max(depth, 1e-8);
    float tolerance = max(0.002, 4.0 * surface.m_error);
    rayQueryEXT query;
    rayQueryInitializeEXT(query, u_tlas, gl_RayFlagsNoneEXT, RT_MASK_STATIC | RT_MASK_DYNAMIC,
        camera, 0.0, direction, depth + tolerance);
    RT_PROCEED(query)
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT ||
        !rayQueryGetIntersectionFrontFaceEXT(query, true))
        return false;
    float hit = rayQueryGetIntersectionTEXT(query, true);
    if (abs(hit - depth) > tolerance)
        return false;
    uint geometry = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true) + rayQueryGetIntersectionGeometryIndexEXT(query, true);
    RTGeometry info = u_rt_geometries[geometry];
    if ((info.m_flags & RT_GEOMETRY_STATIC) == 0u && (info.m_owner == 0u ||
        !canAttachDynamicSurfel(info.m_bounds_min.xyz, info.m_bounds_max.xyz)))
        return false;
    uint primitive = rayQueryGetIntersectionPrimitiveIndexEXT(query, true);
    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(query, true);
    normal = getRTHitNormal(geometry, primitive, bary, mat3(rayQueryGetIntersectionWorldToObjectEXT(query, true)));
    if (dot(surface.m_normal, normal) < 0.7)
        return false;
    position = camera + hit * direction;
    anchor = uvec4(geometry, primitive, packUnorm2x16(bary), info.m_owner);
    return true;
}

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
    vec3 position;
    vec3 normal;
    uvec4 anchor;
    if (!findAttachment(p, position, normal, anchor))
        return;
    float camera_distance = getSurfelCameraDistance(position);
    if (!trySpawnSurfelInCell(getSurfelCell(position,
        getSurfelLevel(camera_distance))))
        return;
    int free_count = atomicAdd(u_pool.m_free_count, -1);
    if (free_count <= 0)
    {
        atomicAdd(u_pool.m_free_count, 1);
        return;
    }
    uint id = u_pool.m_free[free_count - 1];
    Surfel surfel;
    surfel.m_position = position;
    surfel.m_radius = getSurfelRadius(camera_distance);
    surfel.m_normal = normal;
    surfel.m_last_seen = pc.m_time;
    // What is already there, so it doesn't start black
    surfel.m_irradiance = g.m_weight_sum > 0.0 ?
        g.m_irradiance / g.m_weight_sum : vec3(0.0);
    vec4 volume = lookupVolume(position, normal, true);
    if (volume.a > 0.0)
        surfel.m_irradiance = mix(surfel.m_irradiance, volume.rgb,
            volume.a * (1.0 - min(g.m_coverage, 1.0)));
    surfel.m_samples = 0.0;
    if ((pc.m_cache & SURFEL_CACHE_FRAME) != 0u &&
        g.m_coverage >= SURFEL_SEED_COVERAGE)
    {
        surfel.m_samples = min(g.m_min_samples, SURFEL_CONFIDENT_SAMPLES);
    }
    vec3 ray = normalize(position - u_camera.m_inverse_view_matrix[3].xyz);
    float bias = abs(dot(normal, ray)) * p.m_error + dot(abs(position), abs(normal)) * 2.38419e-7;
    lighting[id] = SurfaceLighting(vec4(0.0), 0u, 0.0, 0.0,
        max(0.0001, bias), 0u, 0u, 0u, anchor);
    u_surfels[id] = surfel;
    uint entry = atomicAdd(new_surfels.count, 1u);
    new_surfels.ids[entry] = id;
    atomicMax(new_surfels.dispatch[0], entry / 64u + 1u);
}   // spawnSurfel
