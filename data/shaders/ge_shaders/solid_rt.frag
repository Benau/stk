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
// The vertices and indices of a geometry are in the data buffer (GERTVertex, 32
// bit indices) or, if the geometry is of a dynamic node, in the mesh cache of
// the driver (GEVulkanMeshCache, the same as the raster pass uses, 16 bit
// indices), see m_flags. A vertex is the same 8 words in both: position,
// packed normal, linear vertex color, 2 half uvs, 2 half uvs of the second uv
// and packed tangent (in world space, the ones of the mesh cache are in the
// space of the BLAS of the instance).
//
// Set 0 is the one of GEVulkanRTScene (acceleration structure, data buffer,
// geometry buffer, mesh cache), set 1 the camera and the global light, set 2 GVDFP_HDR
// (g-buffer, same as lighting), set 3 the environment and set 4 the mesh
// textures
#extension GL_EXT_ray_query : require

#define MESH_TEXTURES_SET 4

layout(set = 0, binding = 0) uniform accelerationStructureEXT u_tlas;
// Vertices (8 words, see above) of all geometries, then their indices
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
    float m_uv_shift_x;
    float m_uv_shift_y;
    float m_hue;
    uint m_vertex_color;
    uint m_flags;
};
layout(set = 0, binding = 2, std430) readonly buffer RTGeometries
{
    RTGeometry u_rt_geometries[];
};
// The mesh cache of the driver, vertices, then indices
layout(set = 0, binding = 3, std430) readonly buffer RTMeshCache
{
    uint u_mesh_cache[];
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
#include "utils/rt_utils.glsl"

// Same as GERTShadingType
const uint RT_SHADING_SOLID = 0;
const uint RT_SHADING_NORMALMAP = 1;
const uint RT_SHADING_DECAL = 2;
const uint RT_SHADING_SPLATTING = 3;
const uint RT_VERTEX_WORDS = 8;
// Same as GERTGeometry flags
const uint RT_GEOMETRY_CACHE_VERTEX = 1;
const uint RT_GEOMETRY_CACHE_INDEX = 2;

const float RT_RAY_MIN = 0.02;
const float RT_RAY_MAX = 2000.0;

uint getRTIndex(RTGeometry geometry, uint i)
{
    uint at = geometry.m_index_offset + i;
    if ((geometry.m_flags & RT_GEOMETRY_CACHE_INDEX) == 0u)
        return u_rt_data[at];
    uint word = u_mesh_cache[at >> 1u];
    return (at & 1u) == 0u ? (word & 0xFFFFu) : (word >> 16u);
}

// Words of a vertex, the same in the data buffer and in the mesh cache
const uint RT_VERTEX_NORMAL = 3;
const uint RT_VERTEX_COLOR = 4;
const uint RT_VERTEX_UV = 5;
const uint RT_VERTEX_UV_TWO = 6;
const uint RT_VERTEX_TANGENT = 7;

uint getRTVertexWord(RTGeometry geometry, uint vertex_index, uint word)
{
    uint at = (geometry.m_vertex_offset + vertex_index) * RT_VERTEX_WORDS +
        word;
    if ((geometry.m_flags & RT_GEOMETRY_CACHE_VERTEX) != 0u)
        return u_mesh_cache[at];
    return u_rt_data[at];
}

vec4 getRTVertexColor(RTGeometry geometry, uint vertex_index)
{
    return getVertexColor(getRTVertexWord(geometry, vertex_index,
        RT_VERTEX_COLOR));
}

vec2 getRTVertexUV(RTGeometry geometry, uint vertex_index)
{
    return unpackHalf2x16(getRTVertexWord(geometry, vertex_index,
        RT_VERTEX_UV));
}

vec2 getRTVertexUVTwo(RTGeometry geometry, uint vertex_index)
{
    return unpackHalf2x16(getRTVertexWord(geometry, vertex_index,
        RT_VERTEX_UV_TWO));
}

vec3 getRTVertexNormal(RTGeometry geometry, uint vertex_index)
{
    return unpackSnorm1010102(getRTVertexWord(geometry, vertex_index,
        RT_VERTEX_NORMAL)).xyz;
}

// xyz in the space of the BLAS, w is the sign of the bitangent
vec4 getRTVertexTangent(RTGeometry geometry, uint vertex_index)
{
    return unpackSnorm1010102(getRTVertexWord(geometry, vertex_index,
        RT_VERTEX_TANGENT));
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

vec4 sampleHitTexture(int base, int layer, vec2 uv, float lod)
{
    return textureLod(f_mesh_textures[GE_SAMPLE_TEX_INDEX(base + layer)], uv,
        lod);
}

// The albedo of solid.frag and normalmap.frag (hue, vertex color and the
// texture)
vec3 getSolidAlbedo(int base, vec2 uv, float lod, float hue,
                    vec4 vertex_color)
{
    vec4 tex = sampleHitTexture(base, 0, uv, lod);
    if (hue > 0.0)
    {
        float mask = tex.a;
        vec3 old_hsv = rgbToHsv(tex.rgb);
        float mask_step = step(mask, 0.5);
        float saturation = mask * 2.5;
        vec2 new_xy = mix(vec2(old_hsv.x, old_hsv.y),
            vec2(hue, max(old_hsv.y, saturation)),
            vec2(mask_step, mask_step));
        tex = vec4(hsvToRgb(vec3(new_xy.x, new_xy.y, old_hsv.z)), 1.0);
    }
    return tex.rgb * vertex_color.rgb;
}

// splatting.frag, the 4 detail layers (2 to 5) are mixed by the splatting map
// (layer 1, second uv). The low resolution sampling of the far distance
// (mitigation of repetitive patterns) uses the distance to the camera, and as
// its uv is the half one, it's one level less blurry
vec3 getSplattingAlbedo(int base, vec2 uv, vec2 uv_two, float lod,
                        float camera_distance)
{
    float mitigation = clamp(pow(camera_distance * 0.01, 2.0), 0.0, 1.0);
    float low_lod = max(lod - 1.0, 0.0);
    vec4 splatting = sampleHitTexture(base, 1, uv_two, lod);
    vec4 detail0 = mix(sampleHitTexture(base, 2, uv, lod),
        sampleHitTexture(base, 2, uv * 0.5, low_lod), mitigation);
    vec4 detail1 = mix(sampleHitTexture(base, 3, uv, lod),
        sampleHitTexture(base, 3, uv * 0.5, low_lod), mitigation);
    vec4 detail2 = mix(sampleHitTexture(base, 4, uv, lod),
        sampleHitTexture(base, 4, uv * 0.5, low_lod), mitigation);
    vec4 detail3 = mix(sampleHitTexture(base, 5, uv, lod),
        sampleHitTexture(base, 5, uv * 0.5, low_lod), mitigation);
    return (splatting.r * detail0 + splatting.g * detail1 +
        splatting.b * detail2 +
        max(0.0, (1.0 - splatting.r - splatting.g - splatting.b)) *
        detail3).xyz;
}

// The material of the hit like the fragment shader of the geometry writes it to
// the g-buffer: the albedo, x: gloss, y: metallic, z: emissive of pbr, and the
// (not two sided yet) normal, which the shader can change. uv is the first
// one (with the uv shift of the node), uv_two the second one, which has no
// shift
// tangent is in world space, w is the sign of the bitangent
void getHitMaterial(RTGeometry geometry, int base, vec2 uv, vec2 uv_two,
                    vec4 vertex_color, vec4 tangent, float lod,
                    float camera_distance, out vec3 diffuse_color,
                    out vec3 pbr, inout vec3 normal)
{
    if (geometry.m_shading_type == RT_SHADING_DECAL)
    {
        // decal.frag, the second layer is over the first one, no vertex
        // color and no hue
        vec4 color = sampleHitTexture(base, 0, uv, lod);
        vec4 layer_two = sampleHitTexture(base, 1, uv_two, lod);
        diffuse_color = layer_two.a * layer_two.rgb +
            color.rgb * (1.0 - layer_two.a);
        pbr = sampleHitTexture(base, 2, uv, lod).xyz;
        return;
    }
    if (geometry.m_shading_type == RT_SHADING_SPLATTING)
    {
        // No gloss, metallic and emissive
        diffuse_color = getSplattingAlbedo(base, uv, uv_two, lod,
            camera_distance);
        pbr = vec3(0.0);
        return;
    }
    // solid.frag and normalmap.frag
    diffuse_color = getSolidAlbedo(base, uv, lod, geometry.m_hue,
        vertex_color);
    pbr = sampleHitTexture(base, 2, uv, lod).xyz;
    if (geometry.m_shading_type == RT_SHADING_NORMALMAP &&
        dot(tangent.xyz, tangent.xyz) > 0.0)
    {
        // normalmap.frag, the same tangent space (not orthogonalized)
        vec3 tangent_space_normal =
            2.0 * sampleHitTexture(base, 3, uv, lod).xyz - 1.0;
        vec3 frag_tangent = normalize(tangent.xyz);
        vec3 frag_normal = normalize(normal);
        vec3 frag_bitangent =
            normalize(cross(frag_normal, tangent.xyz) * tangent.w);
        normal = normalize(mat3(frag_tangent, frag_bitangent, frag_normal) *
            tangent_space_normal);
    }
}

// The lit (hdr) color of the closest hit, the values of the hit are
// parameters because a ray query can't be one
vec3 shadeHit(uint geometry_index, uint primitive, vec2 b, float hit_distance,
              mat3 world_to_object, mat3 object_to_world, vec3 ray_origin,
              vec3 ray_direction)
{
    vec3 bary = vec3(1.0 - b.x - b.y, b.x, b.y);

    RTGeometry geometry = u_rt_geometries[geometry_index];
    uvec3 idx = uvec3(getRTIndex(geometry, primitive * 3u),
        getRTIndex(geometry, primitive * 3u + 1u),
        getRTIndex(geometry, primitive * 3u + 2u));

    vec2 uv = getRTVertexUV(geometry, idx.x) * bary.x +
        getRTVertexUV(geometry, idx.y) * bary.y +
        getRTVertexUV(geometry, idx.z) * bary.z;
    vec2 uv_two = getRTVertexUVTwo(geometry, idx.x) * bary.x +
        getRTVertexUVTwo(geometry, idx.y) * bary.y +
        getRTVertexUVTwo(geometry, idx.z) * bary.z;
    uv += vec2(geometry.m_uv_shift_x, geometry.m_uv_shift_y);
    vec4 vertex_color =
        getRTVertexColor(geometry, idx.x) * bary.x +
        getRTVertexColor(geometry, idx.y) * bary.y +
        getRTVertexColor(geometry, idx.z) * bary.z;
    vertex_color *= getVertexColor(geometry.m_vertex_color);
    vec3 normal = normalize(
        getRTVertexNormal(geometry, idx.x) * bary.x +
        getRTVertexNormal(geometry, idx.y) * bary.y +
        getRTVertexNormal(geometry, idx.z) * bary.z);
    // The inverse transpose of object to world (row vector times matrix)
    normal = normalize(normal * world_to_object);
    vec3 eyedir = -ray_direction;
    // The tangents of a dynamic geometry are in the space of its BLAS, the sign
    // of the bitangent is the one of the first vertex
    vec4 tangent = getRTVertexTangent(geometry, idx.x) * bary.x +
        getRTVertexTangent(geometry, idx.y) * bary.y +
        getRTVertexTangent(geometry, idx.z) * bary.z;
    tangent.xyz = object_to_world * tangent.xyz;
    tangent.w = getRTVertexTangent(geometry, idx.x).w >= 0.0 ? 1.0 : -1.0;

    // No derivatives in a ray query, the farther the hit the blurrier
    float lod = clamp(log2(1.0 + hit_distance * 0.5), 0.0, 8.0);
    int base = int(geometry.m_material_id) * TOTAL_MESH_TEXTURE_LAYER;
    vec3 diffuse_color;
    vec3 pbr;
    getHitMaterial(geometry, base, uv, uv_two, vertex_color, tangent, lod,
        distance(ray_origin + ray_direction * hit_distance,
        u_camera.m_inverse_view_matrix[3].xyz), diffuse_color, pbr, normal);
    // Two sided
    if (dot(normal, eyedir) < 0.0)
        normal = -normal;
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
        mat3(rayQueryGetIntersectionObjectToWorldEXT(ray_query, true)),
        world_position, world_reflection);
    o_color = vec4(hit_color * weight, weight);
}
