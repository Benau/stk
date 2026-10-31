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
// Full size ambient occlusion of GEVulkanSSAO, only sampled by the debug view
// (pc.m_ssao_debug and u_ssao), a transparent texture is bound if there is
// none
layout(binding = 4) uniform sampler2D u_ssao_texture;
// Half size irradiance of GEVulkanSurfelGI, only sampled by its debug view
// (pc.m_gi_debug and u_surfel_gi), a transparent texture is bound if there is
// none
layout(binding = 5) uniform sampler2D u_gi_texture;
// Half size indirect light of GEVulkanSSAO (SSIL), multiplied by the albedo of
// the g-buffer (binding 7) and added to hdr, only used if u_ssil (the ambient
// occlusion and the surfel global illumination have to be on), a transparent
// texture is bound if there is none. It's also added to the debug view of the
// global illumination
layout(binding = 6) uniform sampler2D u_ssil_texture;
// The color (albedo) attachment of the g-buffer
layout(binding = 7) uniform sampler2D u_albedo_texture;
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
    // Non zero if the final image is the ambient occlusion of GEVulkanSSAO
    // instead of the tonemapped lighting (GEConfig::m_ssao_debug, the "SSAO
    // viz" of the debug menu), only used if u_ssao
    int m_ssao_debug;
    // Same for the irradiance of GEVulkanSurfelGI and the indirect light of
    // GEVulkanSSAO (GEConfig::m_gi_debug, the "GI viz" of the debug menu),
    // only used if u_surfel_gi
    int m_gi_debug;
} pc;
#endif

layout(location = 0) out vec4 o_color;

#include "utils/constants_utils.glsl"

// Strength of the bloom added to hdr (before tonemap)
const float BLOOM_INTENSITY = 1.75;
// Strength of the blend texture (lens dust) times the widest level
const float BLOOM_BLEND_INTENSITY = 4.0;
// Strength of the indirect light of the ambient occlusion added to hdr
const float SSIL_INTENSITY = 1.0;

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

// The indirect light of GEVulkanSSAO, which the samples of ssil.comp computed
// with the light of this frame: it's multiplied by the albedo of the receiver
// (the same as the ambient light of the lighting pass) and it's added to hdr
// before the bloom and the tonemap. The image is half size and blurred, so a
// bilinear sample is enough, clamped inside the half size viewport so another
// viewport (splitscreen) doesn't leak in
vec3 sampleSsil()
{
    const ivec4 rect = pc.m_viewport_rect;
    if (isEmptyRect(rect))
        return vec3(0.0);
    const ivec4 half_rect = ivec4(rect.xy >> 1, (rect.zw + 1) >> 1);
    // The position of the pixel in the half size image (0.25 is the center of
    // the 2x2 block which it was evaluated at)
    vec2 pos = gl_FragCoord.xy * 0.5 + 0.25;
    pos = clamp(pos, vec2(half_rect.xy) + 0.5, vec2(half_rect.zw) - 0.5);
    const vec3 indirect = textureLod(u_ssil_texture,
        pos / vec2(textureSize(u_ssil_texture, 0)), 0.0).rgb;
    return indirect * texelFetch(u_albedo_texture,
        ivec2(gl_FragCoord.xy), 0).xyz;
}
#endif

void main()
{
#ifdef SPLIT
    // Debug view: the ambient occlusion of the image, without the bloom and
    // the tonemap (the same as the OpenGL renderer). A disabled SSAO has a 1x1
    // transparent texture bound, which the fetch could go above
    if (pc.m_ssao_debug != 0 && u_ssao)
    {
        o_color = vec4(vec3(texelFetch(u_ssao_texture,
            ivec2(gl_FragCoord.xy), 0).x), 1.0);
        return;
    }
    // Debug view: the irradiance of the surfels, as the lighting pass uses it
    // (the image is half size, so it's fetched at half coordinates like
    // lighting does). The indirect light of the ambient occlusion is added to
    // it when it's enabled, so it shows the whole indirect light of the frame.
    // The values are hdr and a sunlit surface is far above 1, so they are
    // tonemapped (Reinhard) and gamma encoded, otherwise almost everything is
    // black or clips to white
    if (pc.m_gi_debug != 0 && u_surfel_gi)
    {
        vec3 gi = texelFetch(u_gi_texture, min(ivec2(gl_FragCoord.xy *
            0.5), textureSize(u_gi_texture, 0) - 1), 0).xyz;
        if (u_ssil)
        {
            gi += texelFetch(u_ssil_texture, min(ivec2(gl_FragCoord.xy *
                0.5), textureSize(u_ssil_texture, 0) - 1), 0).rgb;
        }
        o_color = vec4(pow(gi / (1.0 + gi), vec3(1.0 / 2.2)), 1.0);
        return;
    }
#endif
    vec4 hdr = GE_LOAD_HDR(u_hdr);
#ifdef SPLIT
    // The indirect light of the occlusion (bound to both its switch and the
    // global illumination), before the bloom so the bright parts of it glow
    // too
    if (u_ssil)
        hdr.rgb += sampleSsil() * SSIL_INTENSITY;
    if (u_bloom)
    {
        hdr.rgb += sampleBloom() * BLOOM_INTENSITY;
        if (u_bloom_blend)
            hdr.rgb += sampleBloomBlend();
    }
#endif
    vec4 color = vec4(convertColor(hdr.xyz), 1.0);
    // An offscreen RTT can have an HDR attachment with alpha, which is the
    // clear color (transparent) where nothing is drawn, lighting writes 1.0
    // for pixels with geometry (point lights are additive so it can be above
    // 1.0). For a HDR attachment without alpha (like the one of the default
    // framebuffer usually) this is always 1.0
    color.a = clamp(hdr.a, 0.0, 1.0);
    o_color = color;
}
