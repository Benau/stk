// The surface, the samples and the sampling of a specular reflection, shared
// by the screen space reflection (solid_ssr.frag) and the ray traced one
// (solid_rt.frag): both trace the direction of a sample of the specular lobe
// (sampleSpecularLobe, utils/pbr_utils.glsl) and store the raw radiance of the
// hit premultiplied by the confidence of the trace (the algorithm of
// FidelityFX SSSR, https://github.com/GPUOpen-Effects/FidelityFX-SSSR) in the
// target of the reflection pass, which the lighting calculation multiplies by
// the split-sum BRDF and blends with the prefiltered specular IBL of the
// environment by the coverage (utils/handle_pbr.glsl).
//
// The shader which includes this file must have, before it:
// - the includes of utils/camera.glsl, utils/unproject_position.glsl and
//   ../utils/decodeNormal.frag

#include "random.glsl"

// The surface of a pixel of the g-buffer which reflects, in the spaces of the
// reflections: the screen space trace is in view space, the ray traced scene
// in world space
struct ReflectionSurface
{
    vec3 m_view_position;
    // Direction from the surface toward the eye
    vec3 m_view_eyedir;
    vec3 m_view_normal;
    vec3 m_world_position;
    vec3 m_world_eyedir;
    vec3 m_world_normal;
    // 1 - gloss, the roughness of the PBR data
    float m_perceptual_roughness;
};

// normal_data is the texel of the g-buffer (x, y: normal, z: gloss, w:
// metallic), depth the reverse depth of the pixel (0 is the sky)
ReflectionSurface getReflectionSurface(ivec2 pixel, vec4 normal_data,
                                       float depth)
{
    ReflectionSurface surface;
    surface.m_view_position = getPosFromUVDepth(
        vec3(vec2(pixel) + 0.5, depth), u_camera.m_viewport,
        u_camera.m_inverse_projection_matrix);
    surface.m_view_eyedir = -normalize(surface.m_view_position);
    // The normal of the g-buffer is in world space
    vec3 world_normal = DecodeNormal(normal_data.xy);
    surface.m_view_normal = (u_camera.m_view_matrix *
        vec4(world_normal, 0.0)).xyz;
    surface.m_world_position = (u_camera.m_inverse_view_matrix *
        vec4(surface.m_view_position, 1.0)).xyz;
    surface.m_world_eyedir = normalize((u_camera.m_inverse_view_matrix *
        vec4(surface.m_view_eyedir, 0.0)).xyz);
    surface.m_world_normal = world_normal;
    surface.m_perceptual_roughness = 1.0 - normal_data.z;
    return surface;
}

// A uniform point of [0, 1)² for a sample of the lobe: the R2 sequence
// rotated by a per-pixel random offset (Cranley-Patterson), so the samples of
// a pixel are stratified and decorrelated from the pixels next to it. The
// offset is hashed, not a sequence keyed by the pixel: FidelityFX SSSR uses a
// blue noise texture, and a low discrepancy offset of the pixels next to each
// other shows diagonal bands in the reflection
vec2 getReflectionSample(ivec2 pixel, uint index)
{
    uint seed = uint(pixel.x) | (uint(pixel.y) << 16);
    seed = hashUint(seed);
    vec2 offset = vec2(float(hashUint(seed ^ 0x9e3779b9u)),
        float(hashUint(seed ^ 0x85ebca6bu))) * (1.0 / 4294967296.0);
    return fract(r2Sequence(index) + offset);
}
