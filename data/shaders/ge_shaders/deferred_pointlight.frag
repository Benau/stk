#ifdef SPLIT
layout(binding = 0) uniform sampler2D u_color;
layout(binding = 1) uniform sampler2D u_normal;
layout(binding = 2) uniform sampler2D u_depth;
// Of GEVulkanRTShadow (the same descriptor set as deferred_pbr.frag), used if
// u_rt_shadow_type has the point lights
layout(binding = 4) uniform sampler2DArray u_rt_shadow_texture;
// Enables the filter of it in utils/pbr_light.glsl
#define HAS_RT_SHADOW_TEXTURE
#define GE_LOAD_GBUFFER(tex) texelFetch(tex, ivec2(gl_FragCoord.xy), 0)
#else
layout (input_attachment_index = 0, binding = 0) uniform subpassInput u_color;
layout (input_attachment_index = 1, binding = 1) uniform subpassInput u_normal;
layout (input_attachment_index = 2, binding = 2) uniform subpassInput u_depth;
#define GE_LOAD_GBUFFER(tex) subpassLoad(tex)
#endif

layout(location = 0) flat in int light_idx;

layout(location = 0) out vec4 o_color;

#include "utils/unproject_position.glsl"
#include "utils/handle_pbr.glsl"
#include "../utils/decodeNormal.frag"

void main()
{
    float depth = GE_LOAD_GBUFFER(u_depth).x;
    if (depth == 0.0)
    {
        o_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
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
    vec3 light = calculateLight(light_idx, diffuse_color, normal, xpos,
        eyedir, 1.0 - pbr.x, pbr.y, world_position.xyz);
    o_color = vec4(light, 1.0);
}
