#include "surfel_gi_grid.glsl"

const ivec3 SURFEL_GI_CLIP_SIZE = ivec3(16, 8, 16);
const uint SURFEL_GI_LAYER = 2048u;
const uint SURFEL_GI_BATCH = 256u;
const uint SURFEL_GI_RAYS = 16u;
const uint SURFEL_GI_PRIMARY_BUDGET = 100000u;
const int SURFEL_GI_BANDS = 8;
const uint SURFEL_GI_LIGHT_INTERIOR = 4u;
const uint SURFEL_GI_DEPTH_INTERIOR = 12u;
const uint SURFEL_GI_LIGHT_WIDTH = SURFEL_GI_LIGHT_INTERIOR + 2u;
const uint SURFEL_GI_DEPTH_WIDTH = SURFEL_GI_DEPTH_INTERIOR + 2u;

struct IrradianceProbe
{
    ivec3 cell;
    uint frame;
    vec4 statistics;
    vec3 offset;
    float proximity;
    uint light[SURFEL_GI_LIGHT_WIDTH * SURFEL_GI_LIGHT_WIDTH];
    uint depth[SURFEL_GI_DEPTH_WIDTH * SURFEL_GI_DEPTH_WIDTH];
    vec4 sky_visibility;
};

#ifdef SURFEL_GI_FORWARD
layout(std430, set = 1, binding = 5) readonly buffer ProbeField
#else
layout(std430, set = SURFEL_SET, binding = 10) buffer ProbeField
#endif
{
    vec4 volume_cameras[4];
    uvec4 volume_info;
    IrradianceProbe probes[];
};

uint clipLevels(uint camera)
{
    return min(uint(volume_cameras[camera].w), volume_info.y);
}

float clipSpacing(uint camera, uint level)
{
    if (level + 1u == volume_info.y)
        level = max(level, uint(volume_cameras[camera].w) - 1u);
    return SURFEL_CELL_SIZE * exp2(2.0 * float(level));
}

ivec3 clipOrigin(uint camera, uint level)
{
    return surfelGICellAt(volume_cameras[camera].xyz, clipSpacing(camera, level)) - SURFEL_GI_CLIP_SIZE / 2;
}

float intersectingClipSpacing(vec3 box_min, vec3 box_max)
{
    float best = 0.0;
    for (uint camera = 0u; camera < volume_info.x; ++camera)
    {
        for (uint level = 0u; level < clipLevels(camera); ++level)
        {
            float spacing = clipSpacing(camera, level);
            if (best > 0.0 && spacing >= best)
                break;
            ivec3 origin = clipOrigin(camera, level);
            vec3 clip_min = vec3(origin) * spacing;
            vec3 clip_max = vec3(origin + SURFEL_GI_CLIP_SIZE) * spacing;
            if (any(greaterThan(box_min, clip_max)) || any(lessThan(box_max, clip_min)))
                continue;
            if (spacing == SURFEL_CELL_SIZE)
                return spacing;
            best = spacing;
            break;
        }
    }
    return best;
}

bool canAttachDynamicSurfel(vec3 box_min, vec3 box_max)
{
    vec3 extent = box_max - box_min;
    float longest = max(extent.x, max(extent.y, extent.z));
    if (longest <= 2.0 * SURFEL_CELL_SIZE)
        return false;
    float spacing = intersectingClipSpacing(box_min, box_max);
    return spacing > 0.0 && longest > 2.0 * spacing;
}

uint probeSlot(ivec3 cell, uint level, uint camera)
{
    ivec3 ring = cell & (SURFEL_GI_CLIP_SIZE - 1);
    return (level * volume_info.x + camera) * SURFEL_GI_LAYER +
        uint((ring.z * SURFEL_GI_CLIP_SIZE.y + ring.y) * SURFEL_GI_CLIP_SIZE.x + ring.x);
}

ivec3 slotCell(uint slot)
{
    uint layer = slot / SURFEL_GI_LAYER;
    ivec3 origin = clipOrigin(layer % volume_info.x, layer / volume_info.x);
    ivec3 ring = ivec3(slot, slot >> 4u, slot >> 7u);
    return origin + ((ring - origin) & (SURFEL_GI_CLIP_SIZE - 1));
}

vec2 octSign(vec2 value)
{
    return mix(vec2(-1.0), vec2(1.0), greaterThanEqual(value, vec2(0.0)));
}

vec2 octUV(vec3 direction)
{
    vec3 oct = direction / max(dot(abs(direction), vec3(1.0)), 1e-8);
    return 0.5 + 0.5 * (oct.z < 0.0 ? (1.0 - abs(oct.yx)) * octSign(oct.xy) : oct.xy);
}

vec3 octDirection(uint index, uint width)
{
    vec2 coord = (vec2(index % width, index / width) + 0.5) * (2.0 / float(width)) - 1.0;
    vec3 direction = vec3(coord, 1.0 - dot(abs(coord), vec2(1.0)));
    direction.xy -= octSign(coord) * max(-direction.z, 0.0);
    return normalize(direction);
}

uint octInterior(uint index, uint pitch)
{
    int width = int(pitch) - 2;
    ivec2 coord = ivec2(index % pitch, index / pitch) - 1;
    if (coord.x < 0 || coord.x >= width)
        coord = ivec2(clamp(coord.x, 0, width - 1), width - 1 - coord.y);
    if (coord.y < 0 || coord.y >= width)
        coord = ivec2(width - 1 - coord.x, clamp(coord.y, 0, width - 1));
    return uint(coord.y * width + coord.x);
}

#include "surfel_gi_rgb.glsl"

vec3 probeBilinear(uint slot, vec3 direction, bool moments)
{
    uint pitch = moments ? SURFEL_GI_DEPTH_WIDTH : SURFEL_GI_LIGHT_WIDTH;
    vec2 coord = octUV(direction) * float(pitch - 2u) + 0.5;
    vec2 fraction = fract(coord);
    vec3 sum = vec3(0.0);
    for (int tap = 0; tap < 4; ++tap)
    {
        ivec2 corner = ivec2(tap & 1, tap >> 1);
        vec2 weight = mix(1.0 - fraction, fraction, vec2(corner));
        uvec2 texel = uvec2(floor(coord)) + uvec2(corner);
        uint index = texel.y * pitch + texel.x;
        vec3 value = moments ? vec3(unpackHalf2x16(probes[slot].depth[index]), 0.0) :
            unpackSurfelGIRGB(probes[slot].light[index]);
        sum += weight.x * weight.y * value;
    }
    return sum;
}

uint closestGICamera(vec3 position)
{
    uint best = 0u;
    float best_distance = 1e30;
    for (uint camera = 0u; camera < volume_info.x; ++camera)
    {
        vec3 delta = volume_cameras[camera].xyz - position;
        float distance_squared = dot(delta, delta);
        if (distance_squared < best_distance)
        {
            best = camera;
            best_distance = distance_squared;
        }
    }
    return best;
}

vec4 lookupVolume(vec3 position, vec3 direction, bool surface, vec3 light_direction,
                  out float sky_visibility)
{
    sky_visibility = 1.0;
    if (volume_info.x == 0u || probes.length() == 0)
        return vec4(0.0);
    uint camera = closestGICamera(position);
    vec3 sum = vec3(0.0);
    float total = 0.0;
    uint levels = clipLevels(camera);
    for (uint level = 0u; level < levels; ++level)
    {
        float spacing = clipSpacing(camera, level);
        vec3 grid = surfelGICenterGrid(position, spacing) + (surface ? 0.06 : 0.0) * direction;
        ivec3 origin = clipOrigin(camera, level);
        vec3 local = grid - vec3(origin);
        vec3 border = min(local, vec3(SURFEL_GI_CLIP_SIZE - 1) - local);
        float edge = min(border.x, min(border.y, border.z));
        if (edge <= 0.0)
            continue;
        vec3 fraction = fract(grid);
        vec3 light = vec3(0.0);
        float weights = 0.0;
        float coverage = 0.0;
        float blocked = 0.0;
        for (int tap = 0; tap < 8; ++tap)
        {
            ivec3 corner = ivec3(tap & 1, (tap >> 1) & 1, tap >> 2);
            ivec3 cell = ivec3(floor(grid)) + corner;
            uint slot = probeSlot(cell, level, camera);
            if (probes[slot].frame == 0u || probes[slot].cell != cell || probes[slot].proximity < 0.0)
                continue;
            vec3 delta = grid - (vec3(cell) + probes[slot].offset);
            float distance_cells = length(delta);
            vec3 axis = distance_cells > 1e-5 ? delta / distance_cells : direction;
            vec2 depth = probeBilinear(slot, axis, true).xy;
            float variance = max(depth.y - depth.x * depth.x, 0.0004);
            float gap = max(distance_cells - depth.x, 0.0);
            float visibility = variance / (variance + gap * gap);
            vec3 tri = mix(1.0 - fraction, fraction, vec3(corner));
            float support = tri.x * tri.y * tri.z * visibility * visibility * visibility;
            float weight = support * (surface ? 0.08 + pow(max(0.5 - 0.5 * dot(axis, direction), 0.0), 2.0) : 1.0);
            light += weight * probeBilinear(slot, light_direction, false);
            float sky = dot(probes[slot].sky_visibility, vec4(light_direction, 1.0));
            blocked += weight * (1.0 - clamp(sky, 0.0, 1.0));
            weights += weight;
            coverage += support;
        }
        if (weights <= 1e-6)
            continue;
        float blend = (1.0 - total) * smoothstep(0.0, 0.35, coverage) *
            (level + 1u == levels ? 1.0 : smoothstep(0.0, 1.5, edge));
        sum += blend * light / weights;
        sky_visibility -= blend * blocked / weights;
        total += blend;
        if (total >= 0.999)
            break;
    }
    return vec4(sum / max(total, 1e-6), total);
}

vec4 lookupVolume(vec3 position, vec3 direction, bool surface)
{
    float sky_visibility;
    return lookupVolume(position, direction, surface, direction, sky_visibility);
}

float adaptiveBlend(float value, float base, inout vec4 history)
{
    if (history.w < 1.0)
    {
        history = vec4(value, value, 0.0, 1.0);
        return 1.0;
    }
    float delta = value - history.x;
    history.x += 0.22 * delta;
    history.z = mix(history.z, 0.78 * delta * delta, 0.22);
    float change = abs(history.x - history.y) / (0.02 + sqrt(max(history.z, 0.0)));
    history.w = min(history.w + 1.0, 128.0);
    float blend = max(max(base, 1.0 / history.w), 0.45 * smoothstep(0.3, 1.6, change));
    history.y = mix(history.y, value, blend);
    return blend;
}
