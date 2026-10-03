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
    o_color = vec4(convertColor(GE_LOAD_HDR(u_hdr).xyz), 1.0);
}
