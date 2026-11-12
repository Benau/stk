#ifdef SPLIT
layout(binding = 0) uniform sampler2D u_color;
layout(binding = 1) uniform sampler2D u_normal;
layout(binding = 2) uniform sampler2D u_depth;
// Full size ambient occlusion of GEVulkanSSAO, used if u_ssao (a transparent
// texture is bound if not)
layout(binding = 3) uniform sampler2D u_ssao_texture;
// Visibility of the sun (layer 0) and of the point lights and spot lights
// (layer 1 + the index of the light) of GEVulkanRTShadow, used if
// u_rt_shadow_type has them (a dummy texture is bound if not)
layout(binding = 4) uniform sampler2DArray u_rt_shadow_texture;
// Irradiance of GEVulkanSurfelGI (alpha is its coverage), used if u_surfel_gi (a
// transparent texture is bound if not)
layout(binding = 5) uniform sampler2D u_surfel_gi_texture;
// Reflection of the glossy solid materials of GEVulkanSolidReflection, used if
// u_solid_reflection_image (a transparent texture is bound if not)
layout(binding = 6) uniform sampler2D u_solid_reflection_texture;
// Enables the filter of it in utils/pbr_light.glsl
#define HAS_RT_SHADOW_TEXTURE
#define GE_LOAD_GBUFFER(tex) texelFetch(tex, ivec2(gl_FragCoord.xy), 0)
#else
layout (input_attachment_index = 0, binding = 0) uniform subpassInput u_color;
layout (input_attachment_index = 1, binding = 1) uniform subpassInput u_normal;
layout (input_attachment_index = 2, binding = 2) uniform subpassInput u_depth;
#define GE_LOAD_GBUFFER(tex) subpassLoad(tex)
#endif

layout(location = 0) out vec4 o_color;

layout(push_constant) uniform Constants
{
    int m_fullscreen_light_count;
    // GEVulkanSolidReflection traced the viewport in this frame (only used
    // with u_solid_reflection_image)
    uint m_solid_reflection;
} u_push_constants;

#include "utils/unproject_position.glsl"
#include "utils/handle_pbr.glsl"
#include "../utils/decodeNormal.frag"

#ifdef SPLIT
// ----------------------------------------------------------------------------
// The stochastic reflection of a pixel (u_stochastic_reflection, see
// utils/stochastic_reflection.glsl), the radiance premultiplied by its
// coverage. The target is half size and its texel t is the reflection of the
// pixel 2t, so an even pixel is at a texel and an odd one half way between 2
// (or 4): the bilinear weights of the texels around the pixel, times an edge
// stopping weight (the pixel of the texel is in the viewport, glossy, near the
// plane of this pixel and of a similar normal), so the reflection of another
// surface (in front or behind, or a pixel which reflects nothing) is never
// mixed in. found is false if no texel is of this surface (the specular IBL is
// used then)
vec4 getStochasticReflection(ivec2 pixel, vec3 xpos, vec3 normal,
                             vec3 world_normal, out bool found)
{
    ivec2 base = pixel / 2;
    vec2 f = vec2(pixel - base * 2) * 0.5;
    ivec2 size = textureSize(u_solid_reflection_texture, 0);
    ivec4 rect = ivec4(ivec2(u_camera.m_viewport.xy),
        ivec2(u_camera.m_viewport.xy + u_camera.m_viewport.zw));
    float tolerance = 0.05 + 0.02 * abs(xpos.z);
    vec4 sum = vec4(0.0);
    float weight_sum = 0.0;
    for (int i = 0; i < 4; i++)
    {
        ivec2 offset = ivec2(i & 1, i >> 1);
        float weight = (offset.x == 1 ? f.x : 1.0 - f.x) *
            (offset.y == 1 ? f.y : 1.0 - f.y);
        if (weight <= 0.0)
            continue;
        ivec2 texel = base + offset;
        ivec2 texel_pixel = texel * 2;
        if (any(greaterThanEqual(texel, size)) ||
            any(lessThan(texel_pixel, rect.xy)) ||
            any(greaterThanEqual(texel_pixel, rect.zw)))
            continue;
        vec4 normal_data = texelFetch(u_normal, texel_pixel, 0);
        float depth = texelFetch(u_depth, texel_pixel, 0).x;
        if (normal_data.z < SOLID_REFLECTION_MIN_GLOSS || depth == 0.0)
            continue;
        vec3 texel_xpos = getPosFromUVDepth(vec3(vec2(texel_pixel) + 0.5,
            depth), u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
        weight *= max(1.0 - abs(dot(normal, texel_xpos - xpos)) / tolerance,
            0.0);
        weight *= pow(max(dot(DecodeNormal(normal_data.xy), world_normal),
            0.0), 16.0);
        if (weight <= 0.0)
            continue;
        sum += texelFetch(u_solid_reflection_texture, texel, 0) * weight;
        weight_sum += weight;
    }
    found = weight_sum > 1e-4;
    return found ? sum / weight_sum : vec4(0.0);
}   // getStochasticReflection
#endif

void main()
{
    float depth = GE_LOAD_GBUFFER(u_depth).x;
    if (depth == 0.0)
        discard;
    vec3 diffuse_color = GE_LOAD_GBUFFER(u_color).xyz;
    vec3 pbr = vec3(GE_LOAD_GBUFFER(u_normal).zw, GE_LOAD_GBUFFER(u_color).w);
    vec3 world_normal = DecodeNormal(GE_LOAD_GBUFFER(u_normal).xy);
    vec3 xpos = getPosFromUVDepth(vec3(gl_FragCoord.xy, depth),
        u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
    vec4 world_position = vec4(0.0);
    if (u_shadow_size != 0)
        world_position = u_camera.m_inverse_view_matrix * vec4(xpos, 1.0);
    vec3 eyedir = -normalize(xpos);
    vec3 normal = (u_camera.m_view_matrix * vec4(world_normal, 0.0)).xyz;
    float ao = 1.0;
    float rt_shadow = 1.0;
    vec4 gi = vec4(0.0);
    // The specular radiance which replaces the specular IBL
    bool has_reflection = false;
    vec4 reflection = vec4(0.0);
#ifdef SPLIT
    // Only the glossy pixels have it (GEVulkanSolidReflection doesn't write
    // the others)
    if (u_solid_reflection_image && u_push_constants.m_solid_reflection != 0u &&
        pbr.x >= SOLID_REFLECTION_MIN_GLOSS)
    {
        if (u_stochastic_reflection)
        {
            reflection = getStochasticReflection(ivec2(gl_FragCoord.xy), xpos,
                normal, world_normal, has_reflection);
        }
        else
        {
            has_reflection = true;
            reflection = vec4(texelFetch(u_solid_reflection_texture,
                ivec2(gl_FragCoord.xy), 0).rgb, 1.0);
        }
    }
    if (u_ssao)
        ao = texelFetch(u_ssao_texture, ivec2(gl_FragCoord.xy), 0).x;
    if (u_surfel_gi)
        gi = texelFetch(u_surfel_gi_texture, ivec2(gl_FragCoord.xy), 0);
    if ((u_rt_shadow_type & GST_SUN) != 0)
    {
        rt_shadow = sampleRTShadow(xpos, normal, 0);
    }
#endif
    vec3 hdr = handlePBRDeferred(diffuse_color, pbr, world_normal, eyedir,
        normal, 1.0 - pbr.x, world_position, xpos.z, ao, rt_shadow, gi,
        has_reflection, reflection);
    hdr += accumulateLights(u_push_constants.m_fullscreen_light_count,
        diffuse_color, normal, xpos, eyedir, 1.0 - pbr.x, pbr.y,
        world_position.xyz);
    hdr = handleFog(hdr, xpos);
    o_color = vec4(hdr, 1.0);
}
