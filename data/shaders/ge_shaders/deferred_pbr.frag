#ifdef SPLIT
layout(binding = 0) uniform sampler2D u_color;
layout(binding = 1) uniform sampler2D u_normal;
layout(binding = 2) uniform sampler2D u_depth;
// Full size ambient occlusion of GEVulkanSSAO, used if u_ssao (a transparent
// texture is bound if not)
layout(binding = 3) uniform sampler2D u_ssao_texture;
// Visibility of the sun (layer 0) and of the point lights and spot lights
// (layer 1 + the index of the light) of GEVulkanRTShadow, used if
// u_rt_shadow_type has them (a dummy texture is bound if not)
layout(binding = 4) uniform sampler2DArray u_rt_shadow_texture;
// Irradiance of GEVulkanSurfelGI (alpha is its coverage), used if u_surfel_gi (a
// transparent texture is bound if not)
layout(binding = 5) uniform sampler2D u_surfel_gi_texture;
// Reflection of the glossy solid materials of GEVulkanSolidReflection, used if
// u_solid_reflection_image (a transparent texture is bound if not)
layout(binding = 6) uniform sampler2D u_solid_reflection_texture;
// Enables the filter of it in utils/pbr_light.glsl
#define HAS_RT_SHADOW_TEXTURE
#define GE_LOAD_GBUFFER(tex) texelFetch(tex, ivec2(gl_FragCoord.xy), 0)
#else
layout (input_attachment_index = 0, binding = 0) uniform subpassInput u_color;
layout (input_attachment_index = 1, binding = 1) uniform subpassInput u_normal;
layout (input_attachment_index = 2, binding = 2) uniform subpassInput u_depth;
#define GE_LOAD_GBUFFER(tex) subpassLoad(tex)
#endif

layout(location = 0) out vec4 o_color;

layout(push_constant) uniform Constants
{
    int m_fullscreen_light_count;
    // GEVulkanSolidReflection traced the viewport in this frame (only used
    // with u_solid_reflection_image)
    uint m_solid_reflection;
} u_push_constants;

#include "utils/unproject_position.glsl"
#include "utils/handle_pbr.glsl"
#include "../utils/decodeNormal.frag"

void main()
{
    float depth = GE_LOAD_GBUFFER(u_depth).x;
    if (depth == 0.0)
        discard;
    vec3 diffuse_color = GE_LOAD_GBUFFER(u_color).xyz;
    vec3 pbr = vec3(GE_LOAD_GBUFFER(u_normal).zw, GE_LOAD_GBUFFER(u_color).w);
    vec3 world_normal = DecodeNormal(GE_LOAD_GBUFFER(u_normal).xy);
    vec3 xpos = getPosFromUVDepth(vec3(gl_FragCoord.xy, depth),
        u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
    vec4 world_position = vec4(0.0);
    if (u_shadow_size != 0)
        world_position = u_camera.m_inverse_view_matrix * vec4(xpos, 1.0);
    vec3 eyedir = -normalize(xpos);
    vec3 normal = (u_camera.m_view_matrix * vec4(world_normal, 0.0)).xyz;
    float ao = 1.0;
    float rt_shadow = 1.0;
    vec4 gi = vec4(0.0);
    // The specular radiance which replaces the specular IBL
    bool has_reflection = false;
    vec3 reflection = vec3(0.0);
#ifdef SPLIT
    // Only the glossy pixels have it (GEVulkanSolidReflection doesn't write
    // the others)
    if (u_solid_reflection_image && u_push_constants.m_solid_reflection != 0u &&
        pbr.x >= SOLID_REFLECTION_MIN_GLOSS)
    {
        has_reflection = true;
        reflection = texelFetch(u_solid_reflection_texture,
            ivec2(gl_FragCoord.xy), 0).rgb;
    }
    if (u_ssao)
        ao = texelFetch(u_ssao_texture, ivec2(gl_FragCoord.xy), 0).x;
    if (u_surfel_gi)
        gi = texelFetch(u_surfel_gi_texture, ivec2(gl_FragCoord.xy), 0);
    if ((u_rt_shadow_type & GST_SUN) != 0)
    {
        rt_shadow = sampleRTShadow(xpos, normal, 0);
    }
#endif
    vec3 hdr = handlePBRDeferred(diffuse_color, pbr, world_normal, eyedir,
        normal, 1.0 - pbr.x, world_position, xpos.z, ao, rt_shadow, gi,
        has_reflection, reflection);
    hdr += accumulateLights(u_push_constants.m_fullscreen_light_count,
        diffuse_color, normal, xpos, eyedir, 1.0 - pbr.x, pbr.y,
        world_position.xyz);
    hdr = handleFog(hdr, xpos);
    o_color = vec4(hdr, 1.0);
}
