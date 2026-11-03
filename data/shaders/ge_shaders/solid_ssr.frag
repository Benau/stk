// Screen space specular global illumination of the solid materials
// (GEVulkanDrawCall::renderSolidSSR), a fullscreen draw of the reflection pass
// of GEVulkanDeferredFBOSplit, which renders at a fraction of the size of the
// FBO (u_reflection_scale, a quarter for the screen space reflection). The
// spatial resolve of the pass (solid_reflection_resolve.frag) filters this
// target at its size and the lighting calculation samples the resolved one at
// its base level with the hardware bilinear filter (utils/handle_pbr.glsl),
// like the composite of FidelityFX SSSR.
//
// It's the algorithm of FidelityFX SSSR (GPUOpen): a sample of the lobe of the
// specular BRDF (sampleSpecularLobe, a GGX visible normal distribution) is
// traced over the hierarchical depth (traceHiZView) and the hit is validated
// (validateSSRHit: screen edges, background, backface and the thickness of the
// surface), so a rough reflection is a stochastic estimate of the radiance of
// the lobe instead of a single mirror ray. The radiance of the hit is stored
// raw, like the trace of FidelityFX SSSR: the samples have no BRDF weight (the
// Fresnel of each one would be noise in the reflection), the lighting
// calculation multiplies the average by the split-sum BRDF of the surface once,
// like the composite of FidelityFX SSSR (ApplyReflections.hlsl).
//
// The radiance of a hit is the tonemapped output of the previous frame
// (u_history, the attachment of the tonemap pass, so it has the bloom, the
// ghost and the transparent materials too, converted back to hdr): the
// lighting pass of the current frame is not finished, so the reflected color is
// the one of the last frame. FidelityFX lights its own frame and denoises it
// temporally; there is no reflection history here, so a rough lobe is sampled
// more than once (u_reflection_samples) and the target is filtered by the
// spatial resolve of the pass (solid_reflection_resolve.frag, the prefilter of
// the denoiser of FidelityFX, DNSR). The rays without a valid hit are not
// resampled in the environment here: the lighting calculation blends the
// prefiltered specular IBL of the lobe by the coverage at full resolution,
// which has no noise (FidelityFX samples the environment for every ray because
// its temporal accumulation removes the noise).
//
// The lit color already has the specular IBL of the environment, which the
// reflection replaces. The output of this pass is the traced radiance
// premultiplied by its coverage (rgb * alpha) and the coverage (alpha, the
// confidence of the rays over their count) in the target, so the spatial
// resolve which averages it is the average of the radiance weighted by the
// coverage: the lighting calculation samples the resolved target with the
// hardware bilinear filter and mixes the result with the specular IBL it
// computes at full resolution, so the reduced size trace never blurs the IBL.
// A not glossy pixel is discarded, its coverage stays 0 and the lit color
// keeps the IBL.
//
// Set 0 is the one of GEVulkanHiZDepth::getRenderingDescriptorSet() (same
// layout as the displace pass, the first image is the history instead of the
// current frame, and the hiz is a transparent texture if u_hiz_iterations
// is 0), set 1 the camera and set 2 GVDFP_HDR (g-buffer, same as lighting)
layout(set = 0, binding = 0) uniform sampler2D u_history;
layout(set = 0, binding = 1) uniform sampler2DShadow u_depth_shadow;
layout(set = 0, binding = 2) uniform sampler2D u_hiz_depth;
layout(set = 2, binding = 1) uniform sampler2D u_normal;
layout(set = 2, binding = 2) uniform sampler2D u_depth;

layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/pbr_utils.glsl"
#include "utils/unproject_position.glsl"
#include "../utils/decodeNormal.frag"

// Iteration limit of the hiz trace (0 uses the depth buffer)
const uint u_hiz_iterations =
    u_solid_reflection == SOLID_REFLECTION_SSR_HIZ400 ? 400u :
    u_solid_reflection == SOLID_REFLECTION_SSR_HIZ200 ? 200u :
    u_solid_reflection == SOLID_REFLECTION_SSR_HIZ100 ? 100u : 0u;
// Rays per pixel: FidelityFX SSR denoises one ray temporally, so here the
// quality of the trace is also the number of samples of a rough lobe
const uint u_reflection_samples =
    u_solid_reflection == SOLID_REFLECTION_SSR_HIZ400 ? 16u :
    u_solid_reflection == SOLID_REFLECTION_SSR_HIZ200 ? 12u :
    u_solid_reflection == SOLID_REFLECTION_SSR_HIZ100 ? 8u : 4u;

#define SSR_HIZ
#include "../utils/ssr.glsl"
#include "utils/specular_gi.glsl"

// The confidence of a screen space hit in [0, 1], the FFX_SSSR_ValidateHit of
// FidelityFX SSSR:
// hit_uv        : uv of the hit inside the viewport
// origin_uv     : uv of the reflected pixel inside the viewport
// hit_depth     : depth (reverse depth) of the surface at the hit
// ray_depth     : depth of the ray at the hit, negative if the trace can't
//                 report it (the thickness is not tested then)
// hit_normal    : world space normal of the surface at the hit
// ray_direction : world space direction of the reflected ray
float validateSSRHit(vec2 hit_uv, vec2 origin_uv, float hit_depth,
                     float ray_depth, vec3 hit_normal, vec3 ray_direction)
{
    vec2 viewport_size = u_camera.m_viewport.zw;
    // Reject hits outside the view frustum
    if (any(lessThan(hit_uv, vec2(0.0))) ||
        any(greaterThan(hit_uv, vec2(1.0))))
        return 0.0;
    // Reject the hit if the ray didn't advance significantly, it's an
    // immediate self reflection
    if (all(lessThan(abs(hit_uv - origin_uv), 2.0 / viewport_size)))
        return 0.0;
    // The background (the sky) has no radiance of a surface (reverse depth: 0
    // is the far plane)
    if (hit_depth <= 0.0)
        return 0.0;
    // A surface hit from its back is not what the ray reflects
    if (dot(hit_normal, ray_direction) > 0.0)
        return 0.0;
    // Fade out the hits near the borders, the trace ends at them
    vec2 fov = 0.05 * vec2(viewport_size.y / viewport_size.x, 1.0);
    vec2 border = smoothstep(vec2(0.0), fov, hit_uv) *
        (1.0 - smoothstep(1.0 - fov, vec2(1.0), hit_uv));
    float confidence = border.x * border.y;
    // The ray must be within the thickness of the surface (the same one the
    // trace uses to pass behind it): FidelityFX fades the hit by the distance
    // the ray went behind the surface
    if (ray_depth >= 0.0)
    {
        vec3 pixel = vec3(u_camera.m_viewport.xy + hit_uv * viewport_size,
            hit_depth);
        vec3 view_surface = getPosFromUVDepth(pixel, u_camera.m_viewport,
            u_camera.m_inverse_projection_matrix);
        vec3 view_ray = getPosFromUVDepth(vec3(pixel.xy, ray_depth),
            u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
        float thickness = SSR_THICKNESS_ABS +
            SSR_THICKNESS_REL * abs(view_surface.z);
        float distance = length(view_surface - view_ray);
        float depth_confidence = 1.0 - smoothstep(0.0, thickness, distance);
        confidence *= depth_confidence * depth_confidence;
    }
    return confidence;
}

void main()
{
    // The g-buffer is full size, this pass is not
    ivec2 pixel = min(ivec2(gl_FragCoord.xy * float(u_reflection_scale)),
        ivec2(u_camera.m_screensize.xy) - 1);
    vec4 normal_data = texelFetch(u_normal, pixel, 0);
    // Gloss (specval of the OpenGL renderer), the cheapest rejection first
    float specval = normal_data.z;
    float depth = texelFetch(u_depth, pixel, 0).x;
    if (specval < 0.5 || depth == 0.0)
        discard;

    ReflectionSurface surface = getReflectionSurface(pixel, normal_data,
        depth);

    vec2 viewport_size = u_camera.m_viewport.zw;
    vec2 viewport_scale = viewport_size / u_camera.m_screensize.xy;
    vec2 viewport_offset = u_camera.m_viewport.xy / u_camera.m_screensize.xy;
    vec2 origin_uv = (vec2(pixel) + 0.5 - u_camera.m_viewport.xy) /
        viewport_size;

    vec3 traced_sum = vec3(0.0);
    float traced_confidence = 0.0;
    for (uint i = 0u; i < u_reflection_samples; i++)
    {
        // The sample has no BRDF weight, see the header
        vec3 direction = sampleSpecularLobe(surface.m_world_eyedir,
            surface.m_perceptual_roughness, getReflectionSample(pixel, i),
            surface.m_world_normal);
        // The trace is in view space
        vec3 reflected = (u_camera.m_view_matrix *
            vec4(direction, 0.0)).xyz;
        vec2 hit_uv;
        // Negative: the trace of the depth buffer has no ray depth
        float ray_depth = -1.0;
        if (u_hiz_iterations == 0)
        {
            hit_uv = RayCast(reflected, surface.m_view_position,
                u_camera.m_projection_matrix, viewport_scale, viewport_offset,
                u_depth_shadow);
            // The trace of the depth buffer returns a full screen uv
            hit_uv = (hit_uv - viewport_offset) / viewport_scale;
        }
        else
        {
            vec3 start_ss;
            if (!traceHiZView(surface.m_view_position, reflected,
                u_camera.m_projection_matrix, hit_uv, ray_depth, start_ss))
                continue;
        }
        // The full screen texel of the hit (the g-buffer and the history are
        // the size of the FBO, which has the viewports of the split screen)
        ivec2 hit_pixel = clamp(ivec2(hit_uv * viewport_size), ivec2(0),
            ivec2(viewport_size) - 1) + ivec2(u_camera.m_viewport.xy);
        float hit_depth = texelFetch(u_depth, hit_pixel, 0).x;
        vec3 hit_normal = DecodeNormal(texelFetch(u_normal, hit_pixel, 0).xy);
        float confidence = validateSSRHit(hit_uv, origin_uv, hit_depth,
            ray_depth, hit_normal, direction);
        if (confidence <= 0.0)
            continue;
        // The reflected color is the tonemapped one of the previous frame,
        // converted back to hdr
        vec2 history_uv = hit_uv * viewport_scale + viewport_offset;
        vec3 hdr = inverseConvertColor(texture(u_history, history_uv).rgb);
        // Raw radiance: the split-sum BRDF is applied by the lighting
        // calculation, like the composite of FidelityFX SSSR
        traced_sum += hdr * confidence;
        traced_confidence += confidence;
    }

    // FidelityFX blends the environment with the reflection by the confidence
    // of the hit (lerp(environment, reflection, confidence)): a pixel without
    // a valid hit keeps the specular IBL of the lit color. The lighting
    // calculation does that blend at full resolution, this stores both
    // premultiplied (the spatial resolve of the pass averages the radiance
    // weighted by the coverage)
    float traced_coverage = traced_confidence / float(u_reflection_samples);
    vec3 traced_radiance = traced_sum / max(traced_confidence, 1e-5);
    o_color = vec4(traced_radiance * traced_coverage, traced_coverage);
}
