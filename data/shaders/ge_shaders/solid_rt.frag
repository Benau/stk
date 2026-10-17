// Ray traced reflection of the solid materials (GEVulkanDrawCall::
// renderSolidRT), a fullscreen draw at the end of the lighting pass like
// solid_ssr.frag, but the reflected ray is traced in the ray traced scene
// (GEVulkanRTScene, one acceleration structure of the static nodes in world
// space) instead of the screen, so it has no history and no screen edges.
//
// The hit is shaded by its albedo for now (mesh textures of the hit
// material, vertex color and the uv shift of the node), the lighting of the
// hit is not done yet. A ray without a hit, or a not glossy pixel, is
// discarded so the specular IBL which is already in the lit color stays (the
// output is premultiplied, blended by m_alphablend: src + dst * (1 - alpha)).
//
// Set 0 is the one of GEVulkanRTScene (acceleration structure, data buffer,
// geometry buffer), set 1 the camera, set 2 GVDFP_HDR (g-buffer, same as
// lighting), set 3 the environment (unused) and set 4 the mesh textures
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

layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/pbr_utils.glsl"
#include "utils/unproject_position.glsl"
#include "../utils/decodeNormal.frag"
#include "utils/get_vertex_color.glsl"
#include "utils/sample_mesh_texture.glsl"

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

// The albedo of the closest hit, no lighting yet (a ray query can't be a
// parameter of a function, so the values of the hit are)
vec3 shadeHit(uint geometry_index, uint primitive, vec2 b, float hit_distance)
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

    // No derivatives in a ray query, the farther the hit the blurrier
    float lod = clamp(log2(1.0 + hit_distance * 0.5), 0.0, 8.0);
    int base = int(geometry.m_material_id) * TOTAL_MESH_TEXTURE_LAYER;
    vec4 tex = textureLod(f_mesh_textures[GE_SAMPLE_TEX_INDEX(base)], uv, lod);
    return tex.rgb * vertex_color.rgb;
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

    rayQueryEXT ray_query;
    rayQueryInitializeEXT(ray_query, u_tlas, gl_RayFlagsOpaqueEXT, 0xFF,
        world_position, RT_RAY_MIN, world_reflection, RT_RAY_MAX);
    // Everything is opaque, so no candidate is generated
    while (rayQueryProceedEXT(ray_query)) {}
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT)
        discard;

    // Glossy 0.5 - 1.0 to 0.0 - 1.0
    float weight = (specval - 0.5) * 2.0;
    vec3 hit_color = shadeHit(
        uint(rayQueryGetIntersectionGeometryIndexEXT(ray_query, true)),
        uint(rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true)),
        rayQueryGetIntersectionBarycentricsEXT(ray_query, true),
        rayQueryGetIntersectionTEXT(ray_query, true));
    o_color = vec4(hit_color * weight, weight);
}
