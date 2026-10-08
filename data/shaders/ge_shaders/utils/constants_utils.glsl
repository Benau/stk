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
