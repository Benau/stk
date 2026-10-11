// Screen space reflection of the solid materials (GEVulkanDrawCall::
// renderSolidSSR), a fullscreen draw at the end of the lighting pass (after
// the sky, before the glow outline and light scattering are added).
//
// u_history is the tonemapped output of the previous frame (the attachment of
// the tonemap pass, so it has the bloom, the ghost and transparent materials
// too), converted back to hdr. The traced color replaces the (already lit)
// color of the glossy pixel by the weight of SSRBlendWeight, so the output is
// premultiplied (blended by m_alphablend: src + dst * (1 - alpha)). Pixels
// without a hit, or not glossy, are discarded.
//
// Set 0 is the one of GEVulkanHiZDepth::getRenderingDescriptorSet() (same
// layout as the displace pass, the first image is the history instead of the
// current frame, and the hiz is a transparent texture if u_hiz_iterations
// is 0), set 1 the camera and set 2 GVDFP_HDR (g-buffer, same as lighting)
layout(set = 0, binding = 0) uniform sampler2D u_history;
layout(set = 0, binding = 1) uniform sampler2DShadow u_depth_shadow;
layout(set = 0, binding = 2) uniform sampler2D u_hiz_depth;
layout(set = 2, binding = 1) uniform sampler2D u_normal;
layout(set = 2, binding = 2) uniform sampler2D u_depth;

layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/unproject_position.glsl"
#include "../utils/decodeNormal.frag"

#define SSR_HIZ
#include "../utils/ssr.glsl"

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
    vec3 normal = (u_camera.m_view_matrix *
        vec4(DecodeNormal(normal_data.xy), 0.0)).xyz;
    vec3 reflected = reflect(-eyedir, normal);
    // Disable raycasts towards camera
    float cosine = dot(reflected, eyedir);
    if (!SSRIsReflective(specval, cosine))
        discard;

    vec2 viewport_scale = u_camera.m_viewport.zw / u_camera.m_screensize.xy;
    vec2 viewport_offset = u_camera.m_viewport.xy / u_camera.m_screensize.xy;
    vec2 coords;
    if (u_hiz_iterations == 0)
    {
        coords = RayCast(reflected, xpos, u_camera.m_projection_matrix,
            viewport_scale, viewport_offset, u_depth_shadow);
    }
    else
    {
        vec3 start_ss;
        if (!traceHiZView(xpos, reflected, u_camera.m_projection_matrix,
            coords, start_ss))
            discard;
        coords = coords * viewport_scale + viewport_offset;
    }
    // Never outside of the viewport (the area of the camera, with splitscreen
    // it's a part of the FBO)
    vec2 viewport_coords = (coords - viewport_offset) / viewport_scale;
    if (viewport_coords.x < 0. || viewport_coords.x > 1. ||
        viewport_coords.y < 0. || viewport_coords.y > 1.)
        discard;

    // Disable raycasts onto another reflective surface
    float mirror = texelFetch(u_normal,
        ivec2(coords * u_camera.m_screensize.xy), 0).z;
    float weight = SSRBlendWeight(GetEdgeFade(coords, viewport_scale,
        viewport_offset), cosine, mirror, specval);
    if (weight <= 0.0)
        discard;

    vec3 hdr = inverseConvertColor(texture(u_history, coords).rgb);
    o_color = vec4(hdr * weight, weight);
}
