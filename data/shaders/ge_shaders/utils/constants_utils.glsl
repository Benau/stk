const uint GST_SUN = 1;
const uint GST_POINTLIGHT = 1 << 1;
const uint GST_COMBINED = GST_SUN | GST_POINTLIGHT;

layout (constant_id = 0) const bool u_ibl = true;
layout (constant_id = 1) const float u_specular_levels_minus_one = 0.0;
layout (constant_id = 2) const bool u_deferred = false;
layout (constant_id = 3) const bool u_offscreen_rtt = false;
layout (constant_id = 4) const bool u_ssr = false;
layout (constant_id = 5) const uint u_hiz_iterations = 0;
layout (constant_id = 6) const uint u_shadow_size = 0;
layout (constant_id = 7) const uint u_shadow_type = 0;
layout (constant_id = 8) const uint u_point_shadow_limit = 0;
// Of the split deferred FBO (see lighting_composite.frag)
layout (constant_id = 9) const bool u_glow_outline = false;
layout (constant_id = 10) const bool u_light_scatter = false;
// Of deferred_convert_color.frag, bloom is added to hdr before tonemap
layout (constant_id = 11) const bool u_bloom = false;
// The blend texture (lens dust) of the bloom is also added (needs u_bloom)
layout (constant_id = 12) const bool u_bloom_blend = false;

vec3 convertColor(vec3 input_color)
{
    if (u_ibl)
    {
        return (input_color * (6.5 * input_color + 0.45)) /
            (input_color * (5.0 * input_color + 1.75) + 0.05);
    }
    else
    {
        return (input_color * (7.0 * input_color + 0.75)) /
            (input_color * (5.0 * input_color + 1.75) + 0.05);
    }
}

// Inverse of convertColor (for each channel y = x * (a * x + b) /
// (x * (5 * x + 1.75) + 0.05), so (5 * y - a) * x^2 + (1.75 * y - b) * x +
// 0.05 * y = 0, which has a single positive root), used to read hdr back from
// a tonemapped color. The curve is clamped by an 8 bit unorm output (1.0 is
// reached with x = 0.55 or 0.90 if u_ibl), anything brighter is lost, it needs
// a float output to reach the asymptote of the curve (a / 5)
vec3 inverseConvertColor(vec3 y)
{
    float a = u_ibl ? 6.5 : 7.0;
    float b = u_ibl ? 0.45 : 0.75;
    // The asymptote can't be reached, this keeps the result finite (about
    // 1000 at most) if the output has a higher range than unorm
    y = clamp(y, vec3(0.0), vec3(a * 0.2 - 0.0004));
    vec3 qa = 5.0 * y - a;
    vec3 qb = 1.75 * y - b;
    vec3 qc = 0.05 * y;
    vec3 d = sqrt(max(qb * qb - 4.0 * qa * qc, vec3(0.0)));
    return (2.0 * qc) / max(d - qb, vec3(1e-20));
}
