#define SURFEL_SET 2
#define MESH_TEXTURES_SET 3
layout(set = 4, binding = 2) uniform samplerCube u_skybox;
layout(set = 4, binding = 3) uniform sampler2DArrayShadow u_shadow_map;
#include "camera.glsl"
#include "constants_utils.glsl"
#include "pbr_utils.glsl"
#include "get_vertex_color.glsl"
#include "sample_mesh_texture.glsl"
#include "global_light_data.glsl"
#include "pbr_light.glsl"
#include "rt_utils.glsl"
#include "rt_scene.glsl"
#include "surfel_gi.glsl"
#include "surfel_gi_volume.glsl"
#include "surfel_gi_cache.glsl"
#include "surfel_gi_ray_work.glsl"

layout(set = RT_SCENE_SET, binding = 4, std430) readonly buffer SurfelGIMaterials
{
    uvec2 u_surfel_gi_materials[];
};

float surfelGISunMap(vec3 position, vec3 direction, out vec3 segment)
{
    segment = vec3(0.0);
    bool mapped = (u_shadow_type & GST_SUN) != 0 && u_shadow_size > 0;
    if (mapped && pc.m_has_skybox != 0u)
    {
        for (int cascade = 0; cascade < SUN_CASCADES; ++cascade)
        {
            vec4 clip = u_global_light.m_shadow_projection_view_matrix[cascade] * vec4(position, 1.0);
            vec3 ndc = clip.xyz / clip.w;
            if (clip.w > 0.0 && all(lessThan(abs(ndc.xy), vec2(1.0 - 4.0 / float(u_shadow_size)))) &&
                ndc.z > 0.0 && ndc.z < 1.0)
                return getShadowPCF(u_shadow_map, ndc.xy * 0.5 + 0.5, cascade, ndc.z);
        }
    }
    if (!mapped && (u_rt_shadow_type & GST_SUN) == 0)
        return 1.0;
    segment = direction * RT_SHADOW_MAX;
    return 0.0;
}

vec3 surfelGILocalLight(int index, vec3 position, vec3 normal)
{
    vec4 source = u_global_light.m_lights[index].m_position_radius;
    vec4 color = u_global_light.m_lights[index].m_color_inverse_square_range;
    vec3 delta = source.xyz - position;
    float squared = dot(delta, delta);
    float radius = max(source.w, 1e-4);
    if (squared * color.w >= 1.0 || squared >= radius * radius)
        return vec3(0.0);
    vec3 direction = delta * inversesqrt(max(squared, 1e-8));
    vec4 spot = u_global_light.m_lights[index].m_direction_scale_offset;
    float cone = 1.0;
    if (spot.z != 0.0)
    {
        vec3 axis = vec3(spot.xy, sign(spot.z) * sqrt(max(1.0 - dot(spot.xy, spot.xy), 0.0)));
        cone = clamp(spot.w - abs(spot.z) * dot(axis, direction), 0.0, 1.0);
    }
    return color.rgb * (20.0 * max(dot(normal, direction), 0.0) * cone * cone *
        (1.0 - sqrt(squared) / radius) / (1.0 + squared));
}

float surfelGILightMap(int index, vec3 position, vec3 origin, vec3 sun, out vec3 segment)
{
    if (index < 0)
        return surfelGISunMap(position, sun, segment);
    segment = vec3(0.0);
    vec3 source = u_global_light.m_lights[index].m_position_radius.xyz;
    if (pc.m_has_skybox != 0u && (u_shadow_type & GST_POINTLIGHT) != 0 &&
        u_shadow_size > 0 && index < int(u_point_shadow_limit))
        return sampleOmniShadow(index, source, position);
    vec3 delta = source - origin;
    if (dot(delta, delta) <= 1e-8)
        return 1.0;
    segment = delta;
    return 0.0;
}

float surfelGITraceShadow(vec3 origin, vec3 segment)
{
    float range = length(segment);
    rayQueryEXT shadow;
    rayQueryInitializeEXT(shadow, u_tlas, gl_RayFlagsTerminateOnFirstHitEXT,
        RT_MASK_STATIC | RT_MASK_DYNAMIC, origin, 0.0001, segment / range, range);
    RT_PROCEED(shadow)
    return rayQueryGetIntersectionTypeEXT(shadow, true) == gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

vec3 surfelGIDirect(vec3 position, vec3 normal, vec3 origin,
                    out vec3 segment, out vec3 shadow_light)
{
    segment = vec3(0.0);
    shadow_light = vec3(0.0);
    vec3 sun = u_global_light.m_sun_direction;
    sun *= inversesqrt(max(dot(sun, sun), 1e-12));
    vec3 direct = vec3(0.0);
    for (int index = -1; index < u_global_light.m_light_count; ++index)
    {
        vec3 light = index < 0 ? u_global_light.m_sun_color * max(dot(sun, normal), 0.0) :
            surfelGILocalLight(index, position, normal);
        if (!any(greaterThan(light, vec3(0.0))))
            continue;
        vec3 shadow;
        float visibility = surfelGILightMap(index, position, origin, sun, shadow);
        if (any(notEqual(shadow, vec3(0.0))))
        {
            if (all(equal(segment, vec3(0.0))))
            {
                segment = shadow;
                shadow_light = light * (1.0 / 3.14159265);
                continue;
            }
            visibility = surfelGITraceShadow(origin, shadow);
        }
        direct += visibility * light;
    }
    return direct * (1.0 / 3.14159265);
}

vec3 surfelGISky(vec3 direction)
{
    return getAmbientColor(u_global_light.m_ambient_color) + (pc.m_has_skybox != 0u ?
        textureLod(u_skybox, direction, 0.0).rgb : getAmbientIrradiance());
}

vec3 surfelGIHitBounce(vec3 position, vec3 normal, uint owner)
{
    uint cell = getSurfelCell(position, getSurfelLevel(getSurfelCameraDistance(position)));
    uint count = u_cell_count[cell];
    uint link = ~0u;
    vec4 sum = vec4(0.0);
    for (uint index = 0u; index < count; ++index)
    {
        uint id = nextCellSurfel(cell, index, link);
        vec4 sphere = u_surfels_vec4[id * 3u];
        vec3 delta = position - sphere.xyz;
        float dist2 = dot(delta, delta);
        if (dist2 >= sphere.w * sphere.w || lighting[id].anchor.w != owner)
            continue;
        vec3 donor_normal = u_surfels_vec4[id * 3u + 1u].xyz;
        float separation = max(abs(dot(delta, normal)), abs(dot(delta, donor_normal)));
        float support = 1.0 - dist2 / (sphere.w * sphere.w);
        vec4 donor = u_surfels_vec4[id * 3u + 2u];
        float weight = support * support * smoothstep(0.8, 1.0, dot(normal, donor_normal)) *
            (1.0 - smoothstep(0.0, 0.12 * sphere.w, separation)) * min(donor.w / SURFEL_CONFIDENT_SAMPLES, 1.0);
        sum += vec4(donor.rgb, 1.0) * weight;
    }
    vec3 estimate = sum.rgb / max(sum.w, 1e-6);
    if (sum.w >= 0.5)
        return estimate;
    vec4 bounce = lookupVolume(position, normal, true);
    return mix(bounce.rgb * bounce.a, estimate, smoothstep(0.0, 0.5, sum.w));
}

vec4 surfelGISegment(vec3 origin, vec3 direction, float limit, uint target, float ray_weight)
{
    rayQueryEXT query;
    rayQueryInitializeEXT(query, u_tlas, gl_RayFlagsNoneEXT, RT_MASK_STATIC | RT_MASK_DYNAMIC,
        origin, 0.0001, direction, limit);
    RT_PROCEED(query)
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT)
        return vec4(surfelGISky(direction), limit);
    float hit = rayQueryGetIntersectionTEXT(query, true);
    if (!rayQueryGetIntersectionFrontFaceEXT(query, true))
        return vec4(0.0, 0.0, 0.0, -max(hit, 0.0001));
    uint material = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true) +
        rayQueryGetIntersectionGeometryIndexEXT(query, true);
    RTGeometry geometry = u_rt_geometries[material];
    uint first = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)) * 3u;
    uint va = getRTIndex(geometry, first);
    uint vb = getRTIndex(geometry, first + 1u);
    uint vc = getRTIndex(geometry, first + 2u);
    mat3 transform = mat3(rayQueryGetIntersectionWorldToObjectEXT(query, true));
    vec3 first_position = getRTVertexPosition(geometry, va);
    vec3 normal = cross(getRTVertexPosition(geometry, vb) - first_position,
                        getRTVertexPosition(geometry, vc) - first_position) * transform;
    normal = dot(normal, normal) > 1e-16 ? normalize(normal) : -direction;
    if (dot(normal, direction) > 0.0)
        normal = -normal;
    uvec2 cached = u_surfel_gi_materials[material];
    vec3 outgoing = unpackSurfelGIRGB(cached.y);
    if (cached.x != 0u)
    {
        vec3 segment, shadow_light;
        vec3 position = origin + hit * direction;
        vec3 shadow_origin = position + normal * max(0.0002, dot(abs(position), abs(normal)) * 2.38419e-7);
        vec3 diffuse = unpackSurfelGIRGB(cached.x);
        outgoing += diffuse * (surfelGIDirect(position, normal, shadow_origin, segment, shadow_light) +
            surfelGIHitBounce(position, normal, geometry.m_owner));
        if (any(notEqual(segment, vec3(0.0))))
        {
            vec3 delta = clamp(outgoing + diffuse * shadow_light, 0.0, 64.0) - clamp(outgoing, 0.0, 64.0);
            uint packed = packSurfelGIRGB(delta * ray_weight);
            if (packed != 0u)
            {
                uint phase = target < probeRayCount() ? 0u : 1u;
                uint index = atomicAdd(shadow_dispatch[phase].w, 1u);
                if ((index & 63u) == 0u)
                    atomicAdd(shadow_dispatch[phase].x, 1u);
                uint base = phase == 0u ? 0u : probeRayCount();
                shadow_rays[base + index] = SurfelGIShadow(shadow_origin, target, segment, packed);
            }
        }
    }
    return vec4(clamp(outgoing, 0.0, 64.0), hit);
}
