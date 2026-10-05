layout(binding = 0) uniform sampler2D u_glow;
layout(binding = 1) uniform sampler2D u_glow_blur;

layout(location = 0) out vec4 o_color;

// Strength of the glow added to hdr (before tonemap)
const float GLOW_INTENSITY = 1.0;

void main()
{
    // Original glow, rgb is the color of the meshes and alpha is 1.0 where
    // they are
    vec4 original = texelFetch(u_glow, ivec2(gl_FragCoord.xy), 0);
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(u_glow, 0));
    vec3 blurred = textureLod(u_glow_blur, uv, 0.0).rgb;
    // Blurred glow minus the original one: nothing is added where the meshes
    // are, so they are above their own glow
    vec3 glow = blurred * (1.0 - original.a);// * GLOW_INTENSITY;
    // Alpha of hdr is the coverage if it's rendered to an offscreen rtt
    o_color = vec4(glow, clamp(max(glow.r, max(glow.g, glow.b)), 0.0, 1.0));
}
