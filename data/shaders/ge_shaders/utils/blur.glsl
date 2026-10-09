// Shared by the compute blur shaders of GEVulkanPostProcessing effects (glow
// outline, light scattering and bloom), the wrappers (blur_hdr.comp and
// blur_unorm.comp) only choose the format of the storage image.
//
// The including shader must define BLUR_FORMAT (the format qualifier of the
// storage image) and then calls blurMain() from main().
//
// Every mode reads only inside m_src_rect, which is the rect of one viewport
// (camera) in the source image: the sample position is clamped to the center
// of the first / last pixel of it, so bilinear filtering never mixes with the
// pixels outside (another viewport, or the stale pixels left by the last
// frame) and nothing wraps around the border.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) uniform sampler2D u_src;
// Can be read too (BLUR_MODE_UP_ADD blends with what is already in it)
layout(set = 0, binding = 1, BLUR_FORMAT) uniform image2D u_dst;

const uint BLUR_MODE_DOWN = 0;
const uint BLUR_MODE_UP = 1;
const uint BLUR_MODE_UP_ADD = 2;
const uint BLUR_MODE_GAUSSIAN = 3;
const uint BLUR_MODE_BLOOM_PREFILTER = 4;
const uint BLUR_MODE_STREAK = 5;

layout(push_constant) uniform PushConstants
{
    // x0, y0, x1, y1 (exclusive) in pixels of u_src, samples are clamped
    // inside of it
    ivec4 m_src_rect;
    // Pixels of u_dst to write
    ivec4 m_dst_rect;
    // BLUR_MODE_GAUSSIAN only, (1, 0) = horizontal, (0, 1) = vertical
    ivec2 m_direction;
    uint m_mode;
    // Multiplies the distance of the samples (1.0 at the reference height of
    // the viewport), so the blur has the same width relative to the viewport
    float m_scale;
    // BLUR_MODE_UP_ADD: how much of the upscaled lower level is in the result
    // (the rest is what u_dst has already)
    float m_blend;
} pc;

// pos is in pixels of u_src, bilinear sampling
vec4 sampleSrc(vec2 pos)
{
    vec2 lo = vec2(pc.m_src_rect.xy) + 0.5;
    vec2 hi = max(lo, vec2(pc.m_src_rect.zw) - 0.5);
    return textureLod(u_src, clamp(pos, lo, hi) / vec2(textureSize(u_src, 0)),
        0.0);
}

// ----------------------------------------------------------------------------
// Dual filter ("Bandwidth-Efficient Rendering", Marius Bjorge, ARM, SIGGRAPH
// 2015) downscale, center is in pixels of u_dst (which is half the size of
// u_src)
vec4 dualDown(vec2 center)
{
    vec2 p = center * 2.0;
    vec4 sum = sampleSrc(p) * 4.0;
    sum += sampleSrc(p + vec2(-1.0, -1.0) * pc.m_scale);
    sum += sampleSrc(p + vec2( 1.0,  1.0) * pc.m_scale);
    sum += sampleSrc(p + vec2( 1.0, -1.0) * pc.m_scale);
    sum += sampleSrc(p + vec2(-1.0,  1.0) * pc.m_scale);
    return sum / 8.0;
}

// ----------------------------------------------------------------------------
// Dual filter upscale, center is in pixels of u_dst (which is twice the size
// of u_src)
vec4 dualUp(vec2 center)
{
    vec2 p = center * 0.5;
    vec4 sum = sampleSrc(p + vec2(-1.0,  0.0) * pc.m_scale);
    sum += sampleSrc(p + vec2(-0.5,  0.5) * pc.m_scale) * 2.0;
    sum += sampleSrc(p + vec2( 0.0,  1.0) * pc.m_scale);
    sum += sampleSrc(p + vec2( 0.5,  0.5) * pc.m_scale) * 2.0;
    sum += sampleSrc(p + vec2( 1.0,  0.0) * pc.m_scale);
    sum += sampleSrc(p + vec2( 0.5, -0.5) * pc.m_scale) * 2.0;
    sum += sampleSrc(p + vec2( 0.0, -1.0) * pc.m_scale);
    sum += sampleSrc(p + vec2(-0.5, -0.5) * pc.m_scale) * 2.0;
    return sum / 12.0;
}

// ----------------------------------------------------------------------------
// One direction of a gaussian blur (sigma 5) with 7 taps (the linear sampling
// optimization of the same slides: two neighbor pixels in a bilinear tap),
// same size of u_src and u_dst
const float GAUSSIAN_SIGMA = 5.0;
const float GAUSSIAN_W0 = 1.0 / (sqrt(2.0 * 3.14) * GAUSSIAN_SIGMA);

float gaussianWeight(float i)
{
    return GAUSSIAN_W0 *
        exp(-0.5 * i * i / (GAUSSIAN_SIGMA * GAUSSIAN_SIGMA));
}

vec4 gaussian(vec2 center)
{
    const float w1 = gaussianWeight(1.0), w2 = gaussianWeight(2.0),
        w3 = gaussianWeight(3.0), w4 = gaussianWeight(4.0),
        w5 = gaussianWeight(5.0);
    // Merged taps: the weight is the sum and the offset is in the middle
    // weighted by them
    const float a = w1 + w2, b = w3 + w4;
    const float oa = (1.0 * w1 + 2.0 * w2) / a;
    const float ob = (3.0 * w3 + 4.0 * w4) / b;
    vec2 dir = vec2(pc.m_direction) * pc.m_scale;

    vec4 sum = sampleSrc(center) * GAUSSIAN_W0;
    sum += sampleSrc(center - dir * oa) * a;
    sum += sampleSrc(center + dir * oa) * a;
    sum += sampleSrc(center - dir * ob) * b;
    sum += sampleSrc(center + dir * ob) * b;
    sum += sampleSrc(center - dir * 5.0) * w5;
    sum += sampleSrc(center + dir * 5.0) * w5;
    return sum;
}

// ----------------------------------------------------------------------------
// Horizontal streak of the bloom (the lens flare of OpenGL, lensblend.frag, is
// its bloom which is only blurred in this direction): a gaussian with sigma
// 2.5 pixels of u_src (m_scale of them in distance, like the other modes) with
// the same 7 taps as gaussian(), cut at 2 sigma, and weights which add up to
// 1.0 so the energy of the source is the same
const float STREAK_SIGMA = 2.5;

float streakWeight(float i)
{
    return exp(-0.5 * i * i / (STREAK_SIGMA * STREAK_SIGMA));
}

vec4 streak(vec2 center)
{
    const float w1 = streakWeight(1.0), w2 = streakWeight(2.0),
        w3 = streakWeight(3.0), w4 = streakWeight(4.0),
        w5 = streakWeight(5.0);
    const float a = w1 + w2, b = w3 + w4;
    const float oa = (1.0 * w1 + 2.0 * w2) / a;
    const float ob = (3.0 * w3 + 4.0 * w4) / b;
    const float total = 1.0 + 2.0 * (a + b + w5);
    vec2 dir = vec2(pc.m_scale, 0.0);

    vec4 sum = sampleSrc(center);
    sum += sampleSrc(center - dir * oa) * a;
    sum += sampleSrc(center + dir * oa) * a;
    sum += sampleSrc(center - dir * ob) * b;
    sum += sampleSrc(center + dir * ob) * b;
    sum += sampleSrc(center - dir * 5.0) * w5;
    sum += sampleSrc(center + dir * 5.0) * w5;
    return sum / total;
}

// ----------------------------------------------------------------------------
// First level of the bloom: u_src is the finished hdr color, u_dst is half of
// its size. All the pixels of the bright part of the 4x4 area (of every 2x2
// pixels, so nothing is skipped at any resolution) go in by the 13 tap filter
// of "Next Generation Post Processing in Call of Duty: Advanced Warfare", with
// the weighting of Karis (the brightness of every group of 4 taps is divided
// down) so one very bright pixel of hdr doesn't flicker when it moves.
//
// The threshold is the one of the OpenGL renderer (bloom.frag): the brightness
// goes through smoothstep(LOW, HIGH, brightness) and the chromaticity stays
// the same (the color is scaled), so what goes in the blur is never brighter
// than 1.0 and keeps its saturation: a lamp of 20.0 and one of 2.0 both glow
// orange, they don't turn white in the tonemap like what a subtracted
// threshold (color - 1.0, which keeps all the energy of the hdr) adds to it.
const float BLOOM_THRESHOLD_LOW = 1.0;
const float BLOOM_THRESHOLD_HIGH = 3.0;

vec3 bloomThreshold(vec3 c)
{
    float b = max(c.r, max(c.g, c.b));
    float t = smoothstep(BLOOM_THRESHOLD_LOW, BLOOM_THRESHOLD_HIGH, b);
    return c * (t / max(b, 0.0001));
}

// Threshold of one group of taps, then its weight of the average (Karis)
void bloomAccumulate(inout vec3 sum, inout float total, vec3 group,
                     float weight)
{
    vec3 c = bloomThreshold(group);
    float luma = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float w = weight / (1.0 + luma);
    sum += c * w;
    total += w;
}

vec4 bloomPrefilter(vec2 center)
{
    // The center of the 2x2 pixels of u_src, which is a corner between pixels
    // so the taps at whole pixel offsets are bilinear averages of 2x2 pixels.
    // The offsets are multiplied by m_scale (1.0 is the 4x4 pixels of the
    // original filter) so the area which is averaged before the threshold has
    // the same size relative to the viewport at any resolution: a thin bright
    // line (the ring of a wheel) doesn't lose its peak in a bigger average of
    // the pixels around it when there are more pixels
    vec2 p = center * 2.0;
    vec3 a = sampleSrc(p + vec2(-2.0, -2.0) * pc.m_scale).rgb;
    vec3 b = sampleSrc(p + vec2( 0.0, -2.0) * pc.m_scale).rgb;
    vec3 c = sampleSrc(p + vec2( 2.0, -2.0) * pc.m_scale).rgb;
    vec3 d = sampleSrc(p + vec2(-1.0, -1.0) * pc.m_scale).rgb;
    vec3 e = sampleSrc(p + vec2( 1.0, -1.0) * pc.m_scale).rgb;
    vec3 f = sampleSrc(p + vec2(-2.0,  0.0) * pc.m_scale).rgb;
    vec3 g = sampleSrc(p).rgb;
    vec3 h = sampleSrc(p + vec2( 2.0,  0.0) * pc.m_scale).rgb;
    vec3 i = sampleSrc(p + vec2(-1.0,  1.0) * pc.m_scale).rgb;
    vec3 j = sampleSrc(p + vec2( 1.0,  1.0) * pc.m_scale).rgb;
    vec3 k = sampleSrc(p + vec2(-2.0,  2.0) * pc.m_scale).rgb;
    vec3 l = sampleSrc(p + vec2( 0.0,  2.0) * pc.m_scale).rgb;
    vec3 m = sampleSrc(p + vec2( 2.0,  2.0) * pc.m_scale).rgb;

    // The middle box has half of the weight and the four corner boxes (which
    // overlap) share the other half. No arrays: whole arrays are a problem for
    // the translation to MSL (MoltenVK)
    vec3 sum = vec3(0.0);
    float total = 0.0;
    bloomAccumulate(sum, total, (d + e + i + j) * 0.25, 0.5);
    bloomAccumulate(sum, total, (a + b + f + g) * 0.25, 0.125);
    bloomAccumulate(sum, total, (b + c + g + h) * 0.25, 0.125);
    bloomAccumulate(sum, total, (f + g + k + l) * 0.25, 0.125);
    bloomAccumulate(sum, total, (g + h + l + m) * 0.25, 0.125);
    return vec4(sum / total, 1.0);
}

// ----------------------------------------------------------------------------
void blurMain()
{
    ivec2 dst = pc.m_dst_rect.xy + ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(dst, pc.m_dst_rect.zw)))
        return;
    vec2 center = vec2(dst) + 0.5;
    vec4 result;
    switch (pc.m_mode)
    {
    case BLUR_MODE_DOWN:
        result = dualDown(center);
        break;
    case BLUR_MODE_UP:
        result = dualUp(center);
        break;
    case BLUR_MODE_UP_ADD:
        result = mix(imageLoad(u_dst, dst), dualUp(center), pc.m_blend);
        break;
    case BLUR_MODE_GAUSSIAN:
        result = gaussian(center);
        break;
    case BLUR_MODE_STREAK:
        result = streak(center);
        break;
    default:
        result = bloomPrefilter(center);
        break;
    }
    imageStore(u_dst, dst, result);
}
