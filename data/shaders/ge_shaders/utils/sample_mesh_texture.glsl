#ifdef BIND_MESH_TEXTURES_AT_ONCE
// The includer can change the set (the ray traced pass has it as set 4)
#ifndef MESH_TEXTURES_SET
#define MESH_TEXTURES_SET 0
#endif
layout(set = MESH_TEXTURES_SET, binding = 0) uniform sampler2D f_mesh_textures[];

// Workaround for the drivers (Turnip) which can't preload the descriptors when
// the index comes from a varying: in the fragment shaders of the draws the
// material id is a push constant, so it's uniform for the draw and it doesn't
// need nonuniformEXT (each draw has only 1 material). Others using this file
// (compute shaders, the ray traced pass) keep the id they pass.
// The includer can have its own push constant block (only 1 per stage), then
// it adds "layout(offset = 32) int m_material_id;" to it and defines
// GE_MATERIAL_ID to the member, like displace_mask.frag
#if defined(MATERIAL_ID_UNIFORM) && defined(GE_FRAGMENT_SHADER)
#ifndef GE_MATERIAL_ID
layout(push_constant) uniform MaterialPushConstants
{
    layout(offset = 32) int m_material_id;
} u_material_push_constants;
#define GE_MATERIAL_ID u_material_push_constants.m_material_id
#endif
#define GE_MESH_TEX_INDEX(i) (i)
#define GE_MESH_MATERIAL_ID(material_id) (GE_MATERIAL_ID)
#else
#define GE_MESH_TEX_INDEX(i) GE_SAMPLE_TEX_INDEX(i)
#define GE_MESH_MATERIAL_ID(material_id) (material_id)
#endif

vec4 sampleMeshTexture0(int material_id, vec2 uv)
{
    int id = (TOTAL_MESH_TEXTURE_LAYER * GE_MESH_MATERIAL_ID(material_id)) + 0;
    return texture(f_mesh_textures[GE_MESH_TEX_INDEX(id)], uv);
}

vec4 sampleMeshTexture1(int material_id, vec2 uv)
{
    int id = (TOTAL_MESH_TEXTURE_LAYER * GE_MESH_MATERIAL_ID(material_id)) + 1;
    return texture(f_mesh_textures[GE_MESH_TEX_INDEX(id)], uv);
}

vec4 sampleMeshTexture2(int material_id, vec2 uv)
{
    int id = (TOTAL_MESH_TEXTURE_LAYER * GE_MESH_MATERIAL_ID(material_id)) + 2;
    return texture(f_mesh_textures[GE_MESH_TEX_INDEX(id)], uv);
}

vec4 sampleMeshTexture3(int material_id, vec2 uv)
{
    int id = (TOTAL_MESH_TEXTURE_LAYER * GE_MESH_MATERIAL_ID(material_id)) + 3;
    return texture(f_mesh_textures[GE_MESH_TEX_INDEX(id)], uv);
}

vec4 sampleMeshTexture4(int material_id, vec2 uv)
{
    int id = (TOTAL_MESH_TEXTURE_LAYER * GE_MESH_MATERIAL_ID(material_id)) + 4;
    return texture(f_mesh_textures[GE_MESH_TEX_INDEX(id)], uv);
}

vec4 sampleMeshTexture5(int material_id, vec2 uv)
{
    int id = (TOTAL_MESH_TEXTURE_LAYER * GE_MESH_MATERIAL_ID(material_id)) + 5;
    return texture(f_mesh_textures[GE_MESH_TEX_INDEX(id)], uv);
}

vec4 sampleMeshTexture6(int material_id, vec2 uv)
{
    int id = (TOTAL_MESH_TEXTURE_LAYER * GE_MESH_MATERIAL_ID(material_id)) + 6;
    return texture(f_mesh_textures[GE_MESH_TEX_INDEX(id)], uv);
}

vec4 sampleMeshTexture7(int material_id, vec2 uv)
{
    int id = (TOTAL_MESH_TEXTURE_LAYER * GE_MESH_MATERIAL_ID(material_id)) + 7;
    return texture(f_mesh_textures[GE_MESH_TEX_INDEX(id)], uv);
}
#else
layout(binding = 0) uniform sampler2D f_mesh_texture_0;
layout(binding = 1) uniform sampler2D f_mesh_texture_1;
#ifdef PBR_ENABLED
layout(binding = 2) uniform sampler2D f_mesh_texture_2;
layout(binding = 3) uniform sampler2D f_mesh_texture_3;
layout(binding = 4) uniform sampler2D f_mesh_texture_4;
layout(binding = 5) uniform sampler2D f_mesh_texture_5;
layout(binding = 6) uniform sampler2D f_mesh_texture_6;
layout(binding = 7) uniform sampler2D f_mesh_texture_7;
#endif

vec4 sampleMeshTexture0(int material_id, vec2 uv)
{
    return texture(f_mesh_texture_0, uv);
}

vec4 sampleMeshTexture1(int material_id, vec2 uv)
{
    return texture(f_mesh_texture_1, uv);
}

#ifdef PBR_ENABLED
vec4 sampleMeshTexture2(int material_id, vec2 uv)
{
    return texture(f_mesh_texture_2, uv);
}

vec4 sampleMeshTexture3(int material_id, vec2 uv)
{
    return texture(f_mesh_texture_3, uv);
}

vec4 sampleMeshTexture4(int material_id, vec2 uv)
{
    return texture(f_mesh_texture_4, uv);
}

vec4 sampleMeshTexture5(int material_id, vec2 uv)
{
    return texture(f_mesh_texture_5, uv);
}

vec4 sampleMeshTexture6(int material_id, vec2 uv)
{
    return texture(f_mesh_texture_6, uv);
}

vec4 sampleMeshTexture7(int material_id, vec2 uv)
{
    return texture(f_mesh_texture_7, uv);
}
#endif
#endif
