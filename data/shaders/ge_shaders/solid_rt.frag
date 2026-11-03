// Ray traced reflection of the solid materials (GEVulkanDrawCall::
// renderSolidRT), a fullscreen draw of the reflection pass of
// GEVulkanDeferredFBOSplit like solid_ssr.frag, but the reflected ray is
// traced in the ray traced scene (GEVulkanRTScene, an acceleration structure
// of the static nodes in world space and instances of other ones) instead of
// the screen, so it has no history and no screen edges. The pass renders at
// half of the size of the FBO (u_reflection_scale), the spatial resolve of the
// pass (solid_reflection_resolve.frag) filters this target at its size and the
// lighting calculation samples the resolved one at its base level with the
// hardware bilinear filter (utils/handle_pbr.glsl).
//
// The algorithm is the same as the screen space reflection (solid_ssr.frag,
// the one of FidelityFX SSSR): a sample of the lobe of the specular BRDF
// (sampleSpecularLobe, a GGX visible normal distribution) is traced and the
// radiance of the hit is stored raw, without a BRDF weight, like the trace of
// FidelityFX SSSR. The ray traced scene has no screen edges, so a ray only
// misses the sky. There is no reflection history here, so a rough lobe is
// sampled several times and the target is filtered by the spatial resolve of
// the pass (solid_reflection_resolve.frag, the prefilter of the denoiser of
// FidelityFX, DNSR), like solid_ssr.frag: the rays without a hit keep the
// prefiltered specular IBL of the lobe, which the lighting calculation blends
// by the coverage at full resolution.
//
// The hit is shaded by utils/rt_reflection.glsl (shared with the reflection of
// the displace materials). The output is the same as solid_ssr.frag: the
// traced radiance premultiplied by its coverage (rgb * alpha) and the coverage
// in alpha, so the spatial resolve averages the radiance weighted by the
// coverage, which the lighting calculation samples and mixes with the
// specular IBL of the environment (the lit color already has it) at full
// resolution. A ray without a hit and a not glossy pixel are discarded, their
// coverage stays 0 and the lit color keeps the IBL.
//
// Set 0 is the one of GEVulkanRTScene (acceleration structure, data buffer,
// geometry buffer, mesh cache), set 1 the camera and the global light, set 2 GVDFP_HDR
// (g-buffer, same as lighting), set 3 the environment and set 4 the mesh
// textures
#extension GL_EXT_ray_query : require

#define MESH_TEXTURES_SET 4

// x, y: normal, z: gloss, w: metallic
layout(set = 2, binding = 1) uniform sampler2D u_normal;
layout(set = 2, binding = 2) uniform sampler2D u_depth;
layout(set = 3, binding = 0) uniform samplerCube u_diffuse;
layout(set = 3, binding = 1) uniform samplerCube u_specular;
// Only to share pbr_light.glsl, the shadow of a hit is a ray
layout(set = 3, binding = 3) uniform sampler2DArrayShadow u_shadow_map;
layout(set = 3, binding = 5) uniform sampler2DShadow u_omni_shadow_map;

layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/pbr_utils.glsl"
#include "utils/unproject_position.glsl"
#include "../utils/decodeNormal.frag"
#include "utils/get_vertex_color.glsl"
#include "utils/sample_mesh_texture.glsl"
#include "../utils/rgb_conversion.frag"
#include "utils/global_light_data.glsl"
#include "utils/pbr_light.glsl"
#include "utils/rt_utils.glsl"
#include "utils/rt_scene.glsl"
#include "utils/specular_gi.glsl"

#include "utils/rt_reflection.glsl"

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

    // Rays per pixel: the denoiser of FidelityFX SSSR is temporal and there is
    // no reflection history here, so a rough lobe is sampled as many times as
    // the screen space reflection at its highest quality (solid_ssr.frag)
    const uint u_reflection_samples = 16u;
    // The scene is in world space
    vec3 traced_sum = vec3(0.0);
    float traced_confidence = 0.0;
    for (uint i = 0u; i < u_reflection_samples; i++)
    {
        // The sample has no BRDF weight, see the header
        vec3 direction = sampleSpecularLobe(surface.m_world_eyedir,
            surface.m_perceptual_roughness, getReflectionSample(pixel, i),
            surface.m_world_normal);
        vec3 hit_color;
        if (!traceReflection(surface.m_world_position, direction,
            surface.m_perceptual_roughness * 4.0, hit_color))
            continue;
        // Raw radiance: the split-sum BRDF is applied by the lighting
        // calculation, like the composite of FidelityFX SSSR
        traced_sum += hit_color;
        traced_confidence += 1.0;
    }
    // Same as solid_ssr.frag: a ray without a hit keeps the prefiltered
    // specular IBL, and so does the whole pixel if none of the samples hit.
    // The lighting calculation does that blend by the coverage at full
    // resolution
    if (traced_confidence <= 0.0)
        discard;
    float traced_coverage = traced_confidence / float(u_reflection_samples);
    vec3 traced_radiance = traced_sum / max(traced_confidence, 1e-5);
    o_color = vec4(traced_radiance * traced_coverage, traced_coverage);
}
