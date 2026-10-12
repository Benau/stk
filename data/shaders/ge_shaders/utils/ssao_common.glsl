// Shared by the compute shaders of GEVulkanSSAO (ssao_depth.comp,
// ssao_depth_down.comp, ssao.comp, ssao_blur.comp and ssao_upsample.comp)
layout(push_constant) uniform PushConstants
{
    // x0, y0, x1, y1 (exclusive) in pixels of the g-buffer: the viewport of
    // one camera
    ivec4 m_rect;
    // x: radius of the effect in world units, y: sigma (intensity), z: k
    // (power), w: height of the viewport relative to the reference height of
    // the other effects (so the width of the blur is the same part of the
    // screen at any resolution or render scale)
    vec4 m_params;
    // x: level of the source (ssao_depth_down.comp), y and z: direction of
    // the blur (ssao_blur.comp)
    ivec4 m_misc;
} pc;

#ifdef SHADER_STORAGE_IMAGE_EXTENDED_FORMATS
#define SSAO_AO_FORMAT r8
#else
#define SSAO_AO_FORMAT rgba8
#endif

// View space depth of the pixels without geometry (the sky)
const float SSAO_SKY_Z = 1.0e6;

// Pixels of the rect at a level of the depth chain (x1 and y1 are exclusive)
ivec4 ssaoLevelRect(ivec4 rect, int level)
{
    int round_up = (1 << level) - 1;
    return ivec4(rect.xy >> level, (rect.zw + round_up) >> level);
}

// Pixels of the half size AO images which cover the rect (x1 and y1 are
// exclusive)
ivec4 ssaoHalfRect(ivec4 rect)
{
    return ivec4(rect.xy >> 1, (rect.zw + 1) >> 1);
}

// Linear depth of the four pixels of a 2x2 block for the next level of the
// depth chain: the average, but the surfaces which are not close to the
// farthest one (further than the radius of the effect) have less weight, so
// the depth of a foreground object doesn't leak over what is behind it. Same
// as XeGTAO's depth MIP filter
float ssaoDepthFilter(float d0, float d1, float d2, float d3)
{
    float max_d = max(max(d0, d1), max(d2, d3));
    float range = 0.75 * pc.m_params.x;
    float falloff_range = 0.615 * range;
    float falloff_mul = -1.0 / falloff_range;
    float falloff_add = (range - falloff_range) / falloff_range + 1.0;
    float w0 = clamp((max_d - d0) * falloff_mul + falloff_add, 0.0, 1.0);
    float w1 = clamp((max_d - d1) * falloff_mul + falloff_add, 0.0, 1.0);
    float w2 = clamp((max_d - d2) * falloff_mul + falloff_add, 0.0, 1.0);
    float w3 = clamp((max_d - d3) * falloff_mul + falloff_add, 0.0, 1.0);
    return (w0 * d0 + w1 * d1 + w2 * d2 + w3 * d3) / (w0 + w1 + w2 + w3);
}

// Weight of a neighbour in the blur and the upsample: 1 if it's on the same
// surface and 0 over a depth edge (relative to the radius of the effect and
// the distance, so it doesn't depend on the scale of the scene)
float ssaoDepthWeight(float center, float other)
{
    float tolerance = 0.2 * pc.m_params.x + 0.02 * min(center, 1000.0);
    return max(0.0, 1.0 - abs(center - other) / tolerance);
}
