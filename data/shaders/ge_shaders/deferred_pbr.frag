#ifdef SPLIT
layout(binding = 0) uniform sampler2D u_color;
layout(binding = 1) uniform sampler2D u_normal;
layout(binding = 2) uniform sampler2D u_depth;
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
    vec3 hdr = handlePBRDeferred(diffuse_color, pbr, world_normal, eyedir,
        normal, 1.0 - pbr.x, world_position, xpos.z);
    hdr += accumulateLights(u_push_constants.m_fullscreen_light_count,
        diffuse_color, normal, xpos, eyedir, 1.0 - pbr.x, pbr.y,
        world_position.xyz);
    hdr = handleFog(hdr, xpos);
    o_color = vec4(hdr, 1.0);
}
