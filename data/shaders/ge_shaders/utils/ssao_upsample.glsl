// Last step of GEVulkanSSAO: the full size ambient occlusion which the
// lighting pass samples, bilinear from the half size one but every sample
// is weighted by the difference of its depth to the depth of the pixel, so
// it's not taken from the other side of an edge. Pixels of the sky are 1.0
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

#include "ssao_common.glsl"

layout(set = 0, binding = 0) uniform sampler2D u_ao;
layout(set = 0, binding = 1) uniform sampler2D u_depth;
#ifdef SSIL_UPSAMPLE
layout(set = 0, binding = 2, rgba16f) uniform writeonly image2D u_out;
layout(set = 0, binding = 4) uniform sampler2D u_normal;
layout(set = 0, binding = 5) uniform sampler2D u_color;
const vec3 NEUTRAL = vec3(0.0);
#else
layout(set = 0, binding = 2, SSAO_AO_FORMAT) uniform writeonly image2D u_out;
const vec3 NEUTRAL = vec3(1.0);
#endif

void main()
{
    const ivec2 pixel = pc.m_rect.xy + ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, pc.m_rect.zw)))
        return;

    float center_z = texelFetch(u_depth, pixel, 0).x;
    float radius = ssaoContactRadius(pixel);
    if (center_z >= 0.5 * SSAO_SKY_Z || radius <= 0.0)
    {
        imageStore(u_out, pixel, vec4(NEUTRAL, 1.0));
        return;
    }

    const ivec4 half_rect = ssaoHalfRect(pc.m_rect);
    if (any(lessThanEqual(half_rect.zw, half_rect.xy)))
    {
        imageStore(u_out, pixel, vec4(NEUTRAL, 1.0));
        return;
    }
    // Center of the pixel in the half size image, where the center of a
    // texel is the middle of its 2x2 block
    vec2 position = vec2(pixel) * 0.5;
    ivec2 base = ivec2(floor(position));
    vec2 f = position - vec2(base);

    vec3 sum = vec3(0.0);
    float weight_sum = 0.0;
    for (int i = 0; i < 4; i++)
    {
        ivec2 offset = ivec2(i & 1, i >> 1);
        ivec2 q = clamp(base + offset, half_rect.xy, half_rect.zw - 1);
        vec3 value = texelFetch(u_ao, q, 0).rgb;
        ivec2 depth_pixel = clamp(q * 2, pc.m_rect.xy, pc.m_rect.zw - 1);
        float z = texelFetch(u_depth, depth_pixel, 0).x;
        float bilinear = (offset.x == 0 ? 1.0 - f.x : f.x) *
            (offset.y == 0 ? 1.0 - f.y : f.y);
        float w = bilinear * ssaoDepthWeight(center_z, z, radius) *
            ssaoContactWeight(radius, ssaoContactRadius(depth_pixel));
        sum += w * value;
        weight_sum += w;
    }
    vec3 result = weight_sum > 1.0e-3 ? sum / weight_sum : NEUTRAL;
#ifdef SSIL_UPSAMPLE
    float metallic = texelFetch(u_normal, pixel, 0).w;
    result *= texelFetch(u_color, pixel, 0).rgb * (1.0 - metallic);
#else
    result = result.rrr;
#endif
    if (pc.m_misc.w != 0)
    {
        vec2 contact = texelFetch(u_surfel_contact, pixel, 0).rg;
        if (contact.x > 0.0)
            result = mix(NEUTRAL, result, clamp(contact.y, 0.0, 1.0));
    }
    imageStore(u_out, pixel, vec4(result, 1.0));
}
