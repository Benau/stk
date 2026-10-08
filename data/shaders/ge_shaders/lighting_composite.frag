#include "utils/constants_utils.glsl"

// Both are always bound (the unused ones with a transparent texture), but
// only the enabled ones (specialization constants) are used
layout(binding = 0) uniform sampler2D u_glow;
layout(binding = 1) uniform sampler2D u_glow_blur;
// Half size, blurred
layout(binding = 2) uniform sampler2D u_scatter_blur;

layout(push_constant) uniform PushConstants
{
    // The ones which are rendered this frame for the draw call (viewport),
    // images of the others have stale data: 1 = glow outline, 2 = scatter
    uint m_active;
} pc;

layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"

// Strength of the glow added to hdr (before tonemap)
const float GLOW_INTENSITY = 1.0;

// Additive fullscreen draw at the end of the lighting pass (after the sky),
// the glow outline and the light scattering use the same pipeline because
// they are both a blurred image which is added to hdr
void main()
{
    vec3 add = vec3(0.0);
    if (u_glow_outline && (pc.m_active & 1u) != 0u)
    {
        // Original glow, rgb is the color of the meshes and alpha is 1.0
        // where they are
        vec4 original = texelFetch(u_glow, ivec2(gl_FragCoord.xy), 0);
        vec2 uv = gl_FragCoord.xy / vec2(textureSize(u_glow, 0));
        vec3 blurred = textureLod(u_glow_blur, uv, 0.0).rgb;
        // Blurred glow minus the original one: nothing is added where the
        // meshes are, so they are above their own glow
        add += blurred * (1.0 - original.a);// * GLOW_INTENSITY;
    }
    if (u_light_scatter && (pc.m_active & 2u) != 0u)
    {
        // Position in the half size image, never read outside of the area of
        // the viewport (camera) of the pixel so another viewport (splitscreen)
        // doesn't leak in by the linear filter
        vec2 pos = gl_FragCoord.xy * 0.5;
        vec2 lo = u_camera.m_viewport.xy * 0.5 + 0.5;
        vec2 hi = (u_camera.m_viewport.xy + u_camera.m_viewport.zw) * 0.5 -
            0.5;
        // The OpenGL renderer adds it to the color after the fog too
        add += textureLod(u_scatter_blur,
            clamp(pos, lo, max(lo, hi)) / vec2(textureSize(u_scatter_blur, 0)),
            0.0).rgb;
    }
    // Alpha of hdr is the coverage if it's rendered to an offscreen rtt
    o_color = vec4(add, clamp(max(add.r, max(add.g, add.b)), 0.0, 1.0));
}
