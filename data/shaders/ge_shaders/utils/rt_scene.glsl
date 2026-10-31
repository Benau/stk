// The ray traced scene (GEVulkanRTScene) as seen by a shader: its bindings
// (set RT_SCENE_SET, 0 by default), the geometry data of a hit and the ray
// queries shared by the passes which trace it (solid_rt.frag,
// displace_mask_rt.frag, the ray traced shadows).
//
// The shader which includes this file must have, before it, the include of
// utils/constants_utils.glsl, utils/get_vertex_color.glsl,
// utils/sample_mesh_texture.glsl (with MESH_TEXTURES_SET defined) and
// utils/rt_utils.glsl, and keep #extension GL_EXT_ray_query : require in its
// own file (GEVulkanShaderManager looks for it there to use GLSL 4.60)

// The set of the scene in the pipeline layout of the pass, 0 if it's not defined
#ifndef RT_SCENE_SET
#define RT_SCENE_SET 0
#endif
layout(set = RT_SCENE_SET, binding = 0) uniform accelerationStructureEXT u_tlas;
// Vertices (8 words, see above) of all geometries, then their indices
layout(set = RT_SCENE_SET, binding = 1, std430) readonly buffer RTData
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
layout(set = RT_SCENE_SET, binding = 2, std430) readonly buffer RTGeometries
{
    RTGeometry u_rt_geometries[];
};
// The mesh cache of the driver, vertices, then indices
layout(set = RT_SCENE_SET, binding = 3, std430) readonly buffer RTMeshCache
{
    uint u_mesh_cache[];
};

// The instances of the TLAS (the buffer it's built from, see
// VkAccelerationStructureInstanceKHR) for an effect which is attached to a
// triangle (the surfels of GEVulkanSurfelGI) to follow its node between the
// frames. The key of the node of an instance is its slot in the buffer and is
// stored in the instanceShaderBindingTableRecordOffset of the instance (a ray
// query can read it, see surfel_gi_rt.glsl). The static geometry is already in
// world space, it has the key 0 and there is nothing to read for it. A slot
// which is not used in the frame has the static BLAS with the mask 0 and the
// key 0, so it can't be hit
const uint RT_INSTANCE_CAPACITY = 2048u;
// The 24 bits of instanceCustomIndex and
// instanceShaderBindingTableRecordOffset (the rest is the mask and the flags)
const uint RT_INSTANCE_BITS_MASK = 0xFFFFFFu;
struct RTInstance
{
    // Row major 3x4, the translation is the w of the rows
    vec4 m_transform[3];
    uint m_custom_index_mask;
    uint m_sbt_offset_flags;
    // The address of the BLAS, not read
    uvec2 m_blas_address;
};
layout(set = RT_SCENE_SET, binding = 4, std430) readonly buffer RTInstances
{
    RTInstance u_rt_instances[];
};

// Same as GERTShadingType
const uint RT_SHADING_SOLID = 0;
const uint RT_SHADING_NORMALMAP = 1;
const uint RT_SHADING_DECAL = 2;
const uint RT_SHADING_SPLATTING = 3;
const uint RT_SHADING_ALPHATEST = 4;
const uint RT_SHADING_UNLIT = 5;
const uint RT_SHADING_GRASS = 6;
const uint RT_VERTEX_WORDS = 8;
// Same as GERTGeometry flags
const uint RT_GEOMETRY_CACHE_VERTEX = 1;
const uint RT_GEOMETRY_CACHE_INDEX = 2;

const float RT_RAY_MIN = 0.02;
// No limit of the distance of a ray which shades a pixel (the reflection and
// the indirect light of the surfels): the camera can see up to its far plane
// (5500 in the tracks) and a ray has to find its surface at any distance
const float RT_RAY_MAX = 20000.0;
// Longest distance of a shadow ray, the shadow map of the sun has a range too
const float RT_SHADOW_MAX = 150.0;

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

// Position of a vertex in the space of the BLAS (the model space of a dynamic
// or a grass geometry, the world space of the static and the skinned ones)
vec3 getRTVertexPosition(RTGeometry geometry, uint vertex_index)
{
    const uint at = (geometry.m_vertex_offset + vertex_index) * RT_VERTEX_WORDS;
    if ((geometry.m_flags & RT_GEOMETRY_CACHE_VERTEX) != 0u)
    {
        return vec3(uintBitsToFloat(u_mesh_cache[at]),
            uintBitsToFloat(u_mesh_cache[at + 1u]),
            uintBitsToFloat(u_mesh_cache[at + 2u]));
    }
    return vec3(uintBitsToFloat(u_rt_data[at]),
        uintBitsToFloat(u_rt_data[at + 1u]), uintBitsToFloat(u_rt_data[at + 2u]));
}

vec4 sampleHitTexture(int base, int layer, vec2 uv, float lod)
{
#if defined(BIND_MESH_TEXTURES_AT_ONCE) || defined(MESH_TEXTURES_SET)
    return textureLod(f_mesh_textures[GE_SAMPLE_TEX_INDEX(base + layer)], uv,
        lod);
#else
    return vec4(0.0);
#endif
}

// A candidate hit of a geometry which isn't opaque (alphatest.frag and
// unlit.frag discard below 0.5 of the alpha of the texture and the vertex) is
// confirmed if it isn't discarded, other geometries are opaque, so they're not
// candidates
bool passesAlphaTest(uint geometry_index, uint primitive, vec2 b,
                     float hit_distance)
{
    RTGeometry geometry = u_rt_geometries[geometry_index];
    if (geometry.m_shading_type != RT_SHADING_ALPHATEST &&
        geometry.m_shading_type != RT_SHADING_UNLIT &&
        geometry.m_shading_type != RT_SHADING_GRASS)
        return true;
    vec3 bary = vec3(1.0 - b.x - b.y, b.x, b.y);
    uvec3 idx = uvec3(getRTIndex(geometry, primitive * 3u),
        getRTIndex(geometry, primitive * 3u + 1u),
        getRTIndex(geometry, primitive * 3u + 2u));
    vec2 uv = getRTVertexUV(geometry, idx.x) * bary.x +
        getRTVertexUV(geometry, idx.y) * bary.y +
        getRTVertexUV(geometry, idx.z) * bary.z +
        vec2(geometry.m_uv_shift_x, geometry.m_uv_shift_y);
    // grass.vert has a white vertex color (the vertex color of the mesh is the
    // weight of the wind)
    float vertex_alpha = 1.0;
    if (geometry.m_shading_type != RT_SHADING_GRASS)
    {
        vertex_alpha = (getRTVertexColor(geometry, idx.x).a * bary.x +
            getRTVertexColor(geometry, idx.y).a * bary.y +
            getRTVertexColor(geometry, idx.z).a * bary.z) *
            getVertexColor(geometry.m_vertex_color).a;
    }
    float lod = clamp(log2(1.0 + hit_distance * 0.5), 0.0, 8.0);
    int base = int(geometry.m_material_id) * TOTAL_MESH_TEXTURE_LAYER;
    return sampleHitTexture(base, 0, uv, min(lod, 1.0)).a * vertex_alpha >= 0.5;
}

// A ray query can't be a parameter of a function, so it's a macro, the ray
// goes through the candidates which are discarded
#define RT_PROCEED(query) \
    while (rayQueryProceedEXT(query)) \
    { \
        if (rayQueryGetIntersectionTypeEXT(query, false) == \
            gl_RayQueryCandidateIntersectionTriangleEXT && \
            passesAlphaTest( \
            uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, false)) + \
            uint(rayQueryGetIntersectionGeometryIndexEXT(query, false)), \
            uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false)), \
            rayQueryGetIntersectionBarycentricsEXT(query, false), \
            rayQueryGetIntersectionTEXT(query, false))) \
            rayQueryConfirmIntersectionEXT(query); \
    }

// Same range as the shadow map of the sun (getShadowFactor of pbr_light.glsl):
// faded to lit from the distance 130 and no ray at all after 150
const float RT_SHADOW_FADE_START = 130.0;
const float RT_SHADOW_FADE_END = 150.0;

// Whether the sun is blocked at a world position (1 is lit), camera_distance
// is the distance of the position from the camera (a ray query can't be a
// parameter of a function, but it can be a local one).
float getSunShadow(vec3 position, vec3 normal, vec3 sun_direction,
                   float camera_distance)
{
    if (camera_distance >= RT_SHADOW_FADE_END)
        return 1.0;
    rayQueryEXT shadow_query;
    rayQueryInitializeEXT(shadow_query, u_tlas,
        gl_RayFlagsTerminateOnFirstHitEXT |
        gl_RayFlagsCullFrontFacingTrianglesEXT, 0xFF,
        position + normal * RT_RAY_MIN, RT_RAY_MIN, sun_direction,
        RT_SHADOW_MAX);
    RT_PROCEED(shadow_query)
    float visibility = rayQueryGetIntersectionTypeEXT(shadow_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
    return max(visibility,
        smoothstep(RT_SHADOW_FADE_START, RT_SHADOW_FADE_END, camera_distance));
}

// Whether a point light or a spot light is blocked at a world position (1 is
// lit): a hard shadow, one ray to the center of the light. The ray stops a bit
// before the light, so the mesh of a lamp doesn't shadow it. The caller
// already knows that the light reaches the position (range, cone and facing),
// see rtShadowLightReaches of utils/rt_shadow_layer.glsl
float getLightShadow(vec3 position, vec3 normal, vec3 light_position,
                     float camera_distance)
{
    if (camera_distance >= RT_SHADOW_FADE_END)
        return 1.0;
    vec3 to_light = light_position - position;
    float light_distance = length(to_light);
    rayQueryEXT shadow_query;
    rayQueryInitializeEXT(shadow_query, u_tlas,
        gl_RayFlagsTerminateOnFirstHitEXT |
        gl_RayFlagsCullFrontFacingTrianglesEXT, 0xFF,
        position + normal * RT_RAY_MIN, RT_RAY_MIN,
        to_light / max(light_distance, 1.0e-4),
        max(light_distance - 0.1, RT_RAY_MIN * 2.0));
    RT_PROCEED(shadow_query)
    float visibility = rayQueryGetIntersectionTypeEXT(shadow_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
    // Fades to lit like the sun, a light has a range anyway
    return max(visibility,
        smoothstep(RT_SHADOW_FADE_START, RT_SHADOW_FADE_END, camera_distance));
}
