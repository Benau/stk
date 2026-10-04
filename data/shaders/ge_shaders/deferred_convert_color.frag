#ifdef SPLIT
layout(binding = 0) uniform sampler2D u_hdr;
#define GE_LOAD_HDR(tex) texelFetch(tex, ivec2(gl_FragCoord.xy), 0)
#else
layout (input_attachment_index = 0, binding = 0) uniform subpassInput u_hdr;
#define GE_LOAD_HDR(tex) subpassLoad(tex)
#endif

layout(location = 0) out vec4 o_color;

#include "utils/constants_utils.glsl"

void main()
{
    vec4 hdr = GE_LOAD_HDR(u_hdr);
    vec4 color = vec4(convertColor(hdr.xyz), 1.0);
    // Offscreen RTT has an HDR attachment with alpha (never B10G11R11), which
    // is the clear color (transparent) where nothing is drawn, lighting writes
    // 1.0 for pixels with geometry (point lights are additive so it can be
    // above 1.0)
    if (u_offscreen_rtt)
        color.a = clamp(hdr.a, 0.0, 1.0);
    o_color = color;
}
