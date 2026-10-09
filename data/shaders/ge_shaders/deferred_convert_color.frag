#ifdef SPLIT
layout(binding = 0) uniform sampler2D u_hdr;
// Blurred bloom (half size, the first level of its chain), always bound with a
// transparent texture if it's not enabled, and only used if u_bloom
layout(binding = 1) uniform sampler2D u_bloom_blur;
// The widest level of the same chain (eighth size, same as the 128 texture of
// the OpenGL bloom), only used if u_bloom_blend
layout(binding = 2) uniform sampler2D u_bloom_wide;
// GEConfig::m_bloom_blend_texture (lens dust), mapped on the whole viewport and
// only used if u_bloom_blend
layout(binding = 3) uniform sampler2D u_bloom_blend_tex;
#define GE_LOAD_HDR(tex) texelFetch(tex, ivec2(gl_FragCoord.xy), 0)
#else
layout (input_attachment_index = 0, binding = 0) uniform subpassInput u_hdr;
#define GE_LOAD_HDR(tex) subpassLoad(tex)
#endif

#ifdef SPLIT
layout(push_constant) uniform PushConstants
{
    // The viewport (camera) in pixels (x0, y0, x1, y1 exclusive), for the
    // blend texture
    ivec4 m_viewport_rect;
    // The area (x0, y0, x1, y1 exclusive, in pixels of the half size bloom
    // image) of the viewport which has valid bloom this frame, the same one
    // which the blur of GEVulkanBloom stays inside. Empty if it doesn't have
    // any (the image has stale data), only used if u_bloom
    ivec4 m_bloom_rect;
    // Same for the eighth size image, empty if the viewport is too small
    // for it, only used if u_bloom_blend
    ivec4 m_wide_rect;
} pc;
#endif

layout(location = 0) out vec4 o_color;

#include "utils/constants_utils.glsl"

// Strength of the bloom added to hdr (before tonemap)
const float BLOOM_INTENSITY = 1.75;
// Strength of the blend texture (lens dust) times the widest level
const float BLOOM_BLEND_INTENSITY = 4.0;

#ifdef SPLIT
bool isEmptyRect(ivec4 rect)
{
    return rect.z <= rect.x || rect.w <= rect.y;
}

// pos is in pixels of the image of the sampler, never reading outside of rect
// (area of the viewport) so another viewport (splitscreen) doesn't leak in
vec3 sampleInside(sampler2D tex, ivec4 rect, vec2 pos)
{
    vec2 lo = vec2(rect.xy) + 0.5;
    vec2 hi = vec2(rect.zw) - 0.5;
    return textureLod(tex, clamp(pos, lo, hi) / vec2(textureSize(tex, 0)),
        0.0).rgb;
}

// Bloom is added in the same pass as tonemap so hdr is only read once more
// (it's a tent filter of 4 bilinear taps, the bloom is at half size)
vec3 sampleBloom()
{
    if (isEmptyRect(pc.m_bloom_rect))
        return vec3(0.0);
    // The pixel in half size image coordinates
    vec2 p = gl_FragCoord.xy * 0.5;
    vec3 sum = sampleInside(u_bloom_blur, pc.m_bloom_rect,
        p + vec2(-0.5, -0.5));
    sum += sampleInside(u_bloom_blur, pc.m_bloom_rect, p + vec2( 0.5, -0.5));
    sum += sampleInside(u_bloom_blur, pc.m_bloom_rect, p + vec2(-0.5,  0.5));
    sum += sampleInside(u_bloom_blur, pc.m_bloom_rect, p + vec2( 0.5,  0.5));
    return sum * 0.25;
}

// The blend texture is multiplied by the widest blur, so it shows around the
// bright lights and not on top of them (a single bilinear tap of an eighth
// size image is smooth enough)
vec3 sampleBloomBlend()
{
    if (isEmptyRect(pc.m_wide_rect) || isEmptyRect(pc.m_viewport_rect))
        return vec3(0.0);
    vec3 wide = sampleInside(u_bloom_wide, pc.m_wide_rect,
        gl_FragCoord.xy * 0.125);
    vec2 uv = (gl_FragCoord.xy - vec2(pc.m_viewport_rect.xy)) /
        vec2(pc.m_viewport_rect.zw - pc.m_viewport_rect.xy);
    return texture(u_bloom_blend_tex, uv).rgb * wide * BLOOM_BLEND_INTENSITY;
}
#endif

void main()
{
    vec4 hdr = GE_LOAD_HDR(u_hdr);
#ifdef SPLIT
    if (u_bloom)
    {
        hdr.rgb += sampleBloom() * BLOOM_INTENSITY;
        if (u_bloom_blend)
            hdr.rgb += sampleBloomBlend();
    }
#endif
    vec4 color = vec4(convertColor(hdr.xyz), 1.0);
    // Offscreen RTT has an HDR attachment with alpha (never B10G11R11), which
    // is the clear color (transparent) where nothing is drawn, lighting writes
    // 1.0 for pixels with geometry (point lights are additive so it can be
    // above 1.0)
    if (u_offscreen_rtt)
        color.a = clamp(hdr.a, 0.0, 1.0);
    o_color = color;
}
