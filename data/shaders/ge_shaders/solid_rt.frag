// Ray traced reflection of the solid materials (GEVulkanDrawCall::
// renderSolidRT), a fullscreen draw at the end of the lighting pass like
// solid_ssr.frag, but the reflected ray is traced in the ray traced scene
// (GEVulkanRTScene, an acceleration structure of the static nodes in world
// space and instances of other ones) instead of the screen, so it has no
// history and no screen edges.
//
// The hit is lit like solid.frag + deferred_pbr.frag: albedo of the mesh
// textures (with the hue, vertex color and uv shift of the node, which are
// refreshed every frame by GEVulkanRTScene::update), PBR data of its third
// texture, the sun with a shadow ray, the ambient from the diffuse (irradiance)
// and specular cubemaps of the environment (skytop * ambient color without
// IBL). Point lights are not done yet. A ray without a hit gets the specular
// IBL (same as solid_ssr.frag), a not glossy pixel is discarded, so the
// specular IBL which is already in the lit color stays (the output is
// premultiplied, blended by m_alphablend: src + dst * (1 - alpha)).
//
// The index of the geometry of a hit is the custom index of its instance (the
// first GERTGeometry of the instance) plus the geometry index of the hit in
// the BLAS of the instance. Vertices of an instance which is not in world space
// are in the space of its BLAS, so the normal is transformed to world space by
// the world to object matrix of the hit.
//
// Set 0 is the one of GEVulkanRTScene (acceleration structure, data buffer,
// geometry buffer), set 1 the camera and the global light, set 2 GVDFP_HDR
// (g-buffer, same as lighting), set 3 the environment and set 4 the mesh
// textures
#extension GL_EXT_ray_query : require

#define MESH_TEXTURES_SET 4

layout(set = 0, binding = 0) uniform accelerationStructureEXT u_tlas;
// Vertices (8 words: position, color, normal, 2 half uv) of all geometries,
// then their indices
layout(set = 0, binding = 1, std430) readonly buffer RTData
{
    uint u_rt_data[];
};
// Same as GERTGeometry
struct RTGeometry
{
    uint m_vertex_offset;
    uint m_index_offset;
    uint m_material_id;
    uint m_shading_type;
    vec2 m_uv_shift;
    float m_hue;
    uint m_vertex_color;
};
layout(set = 0, binding = 2, std430) readonly buffer RTGeometries
{
    RTGeometry u_rt_geometries[];
};
layout(set = 2, binding = 1) uniform sampler2D u_normal;
layout(set = 2, binding = 2) uniform sampler2D u_depth;
layout(set = 3, binding = 0) uniform samplerCube u_diffuse;
layout(set = 3, binding = 1) uniform samplerCube u_specular;
// Only to share pbr_light.glsl, the shadow of a hit is a ray
layout(set = 3, binding = 3) uniform sampler2DArrayShadow u_shadow_map;

layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/pbr_utils.glsl"
#include "utils/unproject_position.glsl"
#include "../utils/decodeNormal.frag"
#include "utils/get_vertex_color.glsl"
#include "utils/sample_mesh_texture.glsl"
#include "../utils/rgb_conversion.frag"
#include "utils/global_light_data.glsl"
#include "utils/pbr_light.glsl"

const uint RT_SHADING_SOLID = 0;
const uint RT_VERTEX_WORDS = 8;

const float RT_RAY_MIN = 0.02;
const float RT_RAY_MAX = 2000.0;

vec4 getRTVertexColor(uint vertex_index, uint geometry_vertex_offset)
{
    return getVertexColor(
        u_rt_data[(geometry_vertex_offset + vertex_index) * RT_VERTEX_WORDS + 3]);
}

vec2 getRTVertexUV(uint vertex_index, uint geometry_vertex_offset)
{
    return unpackHalf2x16(
        u_rt_data[(geometry_vertex_offset + vertex_index) * RT_VERTEX_WORDS + 7]);
}

vec3 getRTVertexNormal(uint vertex_index, uint geometry_vertex_offset)
{
    uint base = (geometry_vertex_offset + vertex_index) * RT_VERTEX_WORDS + 4u;
    return vec3(uintBitsToFloat(u_rt_data[base]),
        uintBitsToFloat(u_rt_data[base + 1u]),
        uintBitsToFloat(u_rt_data[base + 2u]));
}

// Whether the sun is blocked at a world position (a ray query can't be a
// parameter of a function, but it can be a local one)
float getSunShadow(vec3 position, vec3 normal, vec3 sun_direction)
{
    rayQueryEXT shadow_query;
    rayQueryInitializeEXT(shadow_query, u_tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 0xFF,
        position + normal * RT_RAY_MIN, RT_RAY_MIN, sun_direction,
        RT_RAY_MAX);
    while (rayQueryProceedEXT(shadow_query)) {}
    return rayQueryGetIntersectionTypeEXT(shadow_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

// The lit (hdr) color of the closest hit, the values of the hit are
// parameters because a ray query can't be one
vec3 shadeHit(uint geometry_index, uint primitive, vec2 b, float hit_distance,
              mat3 world_to_object, vec3 ray_origin, vec3 ray_direction)
{
    vec3 bary = vec3(1.0 - b.x - b.y, b.x, b.y);

    RTGeometry geometry = u_rt_geometries[geometry_index];
    uint index_base = geometry.m_index_offset + primitive * 3u;
    uvec3 idx = uvec3(u_rt_data[index_base], u_rt_data[index_base + 1u],
        u_rt_data[index_base + 2u]);

    vec2 uv = getRTVertexUV(idx.x, geometry.m_vertex_offset) * bary.x +
        getRTVertexUV(idx.y, geometry.m_vertex_offset) * bary.y +
        getRTVertexUV(idx.z, geometry.m_vertex_offset) * bary.z;
    uv += geometry.m_uv_shift;
    vec4 vertex_color =
        getRTVertexColor(idx.x, geometry.m_vertex_offset) * bary.x +
        getRTVertexColor(idx.y, geometry.m_vertex_offset) * bary.y +
        getRTVertexColor(idx.z, geometry.m_vertex_offset) * bary.z;
    vertex_color *= getVertexColor(geometry.m_vertex_color);
    vec3 normal = normalize(
        getRTVertexNormal(idx.x, geometry.m_vertex_offset) * bary.x +
        getRTVertexNormal(idx.y, geometry.m_vertex_offset) * bary.y +
        getRTVertexNormal(idx.z, geometry.m_vertex_offset) * bary.z);
    // The inverse transpose of object to world (row vector times matrix)
    normal = normalize(normal * world_to_object);
    vec3 eyedir = -ray_direction;
    // Two sided
    if (dot(normal, eyedir) < 0.0)
        normal = -normal;

    // No derivatives in a ray query, the farther the hit the blurrier
    float lod = clamp(log2(1.0 + hit_distance * 0.5), 0.0, 8.0);
    int base = int(geometry.m_material_id) * TOTAL_MESH_TEXTURE_LAYER;
    vec4 tex = textureLod(f_mesh_textures[GE_SAMPLE_TEX_INDEX(base)], uv, lod);
    // Same as solid.frag
    if (geometry.m_hue > 0.0)
    {
        float mask = tex.a;
        vec3 old_hsv = rgbToHsv(tex.rgb);
        float mask_step = step(mask, 0.5);
        float saturation = mask * 2.5;
        vec2 new_xy = mix(vec2(old_hsv.x, old_hsv.y),
            vec2(geometry.m_hue, max(old_hsv.y, saturation)),
            vec2(mask_step, mask_step));
        tex = vec4(hsvToRgb(vec3(new_xy.x, new_xy.y, old_hsv.z)), 1.0);
    }
    vec3 diffuse_color = tex.rgb * vertex_color.rgb;
    // x: gloss, y: metallic, z: emissive
    vec3 pbr = textureLod(f_mesh_textures[GE_SAMPLE_TEX_INDEX(base + 2)], uv,
        lod).xyz;
    float perceptual_roughness = 1.0 - pbr.x;

    vec3 world_position = ray_origin + ray_direction * hit_distance;
    vec3 sun_direction = normalize(u_global_light.m_sun_direction);
    float shadow = 1.0;
    if (u_shadow_size != 0 && dot(normal, sun_direction) > 0.0)
        shadow = getSunShadow(world_position, normal, sun_direction);

    // Same as handlePBRDeferred, in world space (no sun disk, no ssao)
    vec3 irradiance = vec3(0.0);
    vec3 radiance = vec3(0.0);
    if (u_ibl)
    {
        vec3 reflection = reflect(ray_direction, normal);
        irradiance = texture(u_diffuse, normal).rgb;
        radiance = textureLod(u_specular, reflection,
            perceptual_roughness * u_specular_levels_minus_one).rgb;
    }
    vec3 hdr = PBRSunAmbientEmitLight(normal, eyedir, sun_direction,
        diffuse_color, irradiance, radiance, u_global_light.m_sun_color,
        u_global_light.m_ambient_color, perceptual_roughness, pbr.y, pbr.z,
        shadow, vec3(1.0), vec3(1.0));

    // TODO: point lights (accumulateLights of the lighting pass, in world
    // space, with the shadows of the lights as a ray too)
    return hdr;
}

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 normal_data = texelFetch(u_normal, pixel, 0);
    // Gloss (specval of the OpenGL renderer), the cheapest rejection first
    float specval = normal_data.z;
    float depth = texelFetch(u_depth, pixel, 0).x;
    if (specval < 0.5 || depth == 0.0)
        discard;

    vec3 xpos = getPosFromUVDepth(vec3(gl_FragCoord.xy, depth),
        u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
    vec3 eyedir = -normalize(xpos);
    vec3 view_normal = DecodeNormal(normal_data.xy);
    vec3 normal = (u_camera.m_view_matrix * vec4(view_normal, 0.0)).xyz;
    vec3 reflected = reflect(-eyedir, normal);

    // The scene is in world space
    vec3 world_position = (u_camera.m_inverse_view_matrix *
        vec4(xpos, 1.0)).xyz;
    vec3 world_reflection = normalize((u_camera.m_inverse_view_matrix *
        vec4(reflected, 0.0)).xyz);
    float cosine = dot(reflected, eyedir);

    rayQueryEXT ray_query;
    rayQueryInitializeEXT(ray_query, u_tlas, gl_RayFlagsOpaqueEXT, 0xFF,
        world_position, RT_RAY_MIN, world_reflection, RT_RAY_MAX);
    // Everything is opaque, so no candidate is generated
    while (rayQueryProceedEXT(ray_query)) {}
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT)
    {
        // Same as the rays of solid_ssr.frag that can't be traced, the
        // specular IBL, and without IBL the lit color stays
        if (u_ibl)
        {
            float perceptual_roughness = 1.0 - specval;
            float miss_weight = 0.35 * smoothstep(0.2, 0.5, cosine);
            vec3 ibl = textureLod(u_specular, world_reflection,
                perceptual_roughness * u_specular_levels_minus_one).rgb;
            o_color = vec4(ibl * miss_weight, miss_weight);
            return;
        }
        discard;
    }

    // Glossy 0.5 - 1.0 to 0.0 - 1.0
    float weight = (specval - 0.5) * 2.0;
    uint geometry_index =
        uint(rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true)) +
        uint(rayQueryGetIntersectionGeometryIndexEXT(ray_query, true));
    vec3 hit_color = shadeHit(geometry_index,
        uint(rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true)),
        rayQueryGetIntersectionBarycentricsEXT(ray_query, true),
        rayQueryGetIntersectionTEXT(ray_query, true),
        mat3(rayQueryGetIntersectionWorldToObjectEXT(ray_query, true)),
        world_position, world_reflection);
    o_color = vec4(hit_color * weight, weight);
}
