// The stochastic reflection of the glossy solid materials
// (GEVulkanSolidReflection with GEConfig::m_stochastic_reflection), shared by
// its passes. It's the algorithm of FidelityFX SSSR: a rough surface doesn't
// reflect a single mirror ray, a few samples of the lobe of its specular BRDF
// (a GGX visible normal distribution) are traced and their radiance is stored
// raw, the lighting pass applies the split-sum BRDF of the surface to the
// average once (handlePBRDeferred), like the composite of FidelityFX SSSR. The
// trace is the screen space one over the depth buffer or the HiZ depth
// (stochastic_reflection_ssr.comp) or the ray traced one
// (stochastic_reflection_rt.comp).
//
// FidelityFX traces one ray per pixel and denoises it temporally. There is no
// reflection history here, so the passes are:
//
//   classify : stochastic_reflection_classify.comp, the glossy texels of the
//              half size target are added to a list (one texel is the 2x2
//              pixels of the g-buffer whose top left pixel, its
//              representative pixel getStochasticReflectionPixel, is
//              traced)
//   trace    : the texels of the list (indirect dispatch), a few rays each
//              (more for a rougher surface, one for a mirror), into level 0
//              of the trace target. The rays of a texel are traced by 8
//              threads (STOCHASTIC_REFLECTION_THREADS) at the same time and
//              summed: a trace is latency bound, a thread which traces its
//              rays one after another waits for each of them (3 times slower
//              with the ray traced scene, twice with 400 iterations of HiZ)
//   mipmap   : stochastic_reflection_mipmap.comp, levels 1 to 3 of the trace
//              target, the average of the glossy texels only
//   resolve  : stochastic_reflection_resolve.comp, the texels of the list
//              (one thread each, its own indirect dispatch), the spatial
//              filter of the trace (the
//              prefilter of the reflection denoiser of FidelityFX, DNSR),
//              into the resolved target which the lighting pass reads
//
// The trace target stores the radiance premultiplied by its coverage (rgb *
// alpha, the alpha is the confidence of the rays over their count) so the
// averages of the mipmap and the resolve are the averages of the radiance
// weighted by the coverage, and the lighting pass mixes the specular IBL of
// the pixel with the reflection by the coverage at full resolution (a ray
// without a hit is the environment, the prefiltered IBL has no noise). It
// needs the alpha, so the targets are RGBA16F (half size: 2 bytes per pixel
// of the screen).
//
// Only the glossy texels are traced. The mipmap and the resolve test the
// g-buffer of a texel (isStochasticReflectionTexel, the same test as the
// classify), so they never average a texel which wasn't traced in this frame,
// but a tap of the resolve is a bilinear sample of 4 texels, whose neighbors
// of the tested one can be any texel: the trace target is cleared every frame
// (GEVulkanSolidReflection::render), so they are 0 and never the output of an
// older frame (or the memory of another image, which can be NaN). The resolved
// target isn't cleared: the lighting pass only reads its glossy texels, which
// the resolve writes every frame (it's cleared once when it's created).
//
// The shader which includes it defines SOLID_REFLECTION_SET if it's not 0, and
// includes utils/constants_utils.glsl, utils/pbr_utils.glsl,
// utils/camera.glsl (if it uses getStochasticReflectionSurface),
// utils/unproject_position.glsl and ../utils/decodeNormal.frag before it

#ifndef SOLID_REFLECTION_SET
#define SOLID_REFLECTION_SET 0
#endif

// Levels of the trace target (the mipmap writes 1 to 3)
const int STOCHASTIC_REFLECTION_LEVELS = 4;

// Threads which trace the rays of a texel together, the y of the indirect
// dispatch of the trace (a group of 64 threads has 64 / 8 texels, so the 8
// groups in y have the 64 texels of the x of the dispatch)
const uint STOCHASTIC_REFLECTION_THREADS = 8u;

// The glossy texels (packStochasticReflectionTexel), m_trace_dispatch is the
// indirect dispatch of the trace (x groups of 64 texels, y is
// STOCHASTIC_REFLECTION_THREADS) and m_resolve_dispatch the one of the
// resolve (x groups of 64 texels, one thread each). y and z are written by
// GEVulkanSolidReflection before the classify
layout(std430, set = SOLID_REFLECTION_SET, binding = 1) buffer
    StochasticReflectionList
{
    uint m_count;
    uint m_trace_dispatch[3];
    uint m_resolve_dispatch[3];
    uint m_padding;
    uint m_texels[];
} u_reflection_list;

// ----------------------------------------------------------------------------
uint packStochasticReflectionTexel(ivec2 texel)
{
    return uint(texel.x) | (uint(texel.y) << 16u);
}   // packStochasticReflectionTexel

// ----------------------------------------------------------------------------
ivec2 unpackStochasticReflectionTexel(uint packed)
{
    return ivec2(packed & 0xFFFFu, packed >> 16u);
}   // unpackStochasticReflectionTexel

// ----------------------------------------------------------------------------
// The texel of the thread of the resolve, false if it has none (the last
// group isn't full)
bool getStochasticReflectionTexel(out ivec2 texel)
{
    texel = ivec2(0);
    if (gl_GlobalInvocationID.x >= u_reflection_list.m_count)
        return false;
    texel = unpackStochasticReflectionTexel(
        u_reflection_list.m_texels[gl_GlobalInvocationID.x]);
    return true;
}   // getStochasticReflectionTexel

// ----------------------------------------------------------------------------
// The pixel of the g-buffer which a texel of the half size targets traces.
// It's the top left one of its 2x2 pixels, so a pixel of the screen with even
// coordinates is exactly at a texel and an odd one half way between two, which
// the upsample of the lighting pass uses (deferred_pbr.frag)
ivec2 getStochasticReflectionPixel(ivec2 texel)
{
    return texel * 2;
}   // getStochasticReflectionPixel

// ----------------------------------------------------------------------------
// The texel is traced: its pixel is in the viewport rect (x0, y0, x1
// exclusive, y1 exclusive, in pixels; a texel belongs to the viewport of its
// pixel, never to two), glossy and not the sky (reverse z, depth 0). The same
// test for the classify, the mipmap, the resolve and the upsample of the
// lighting pass
bool isStochasticReflectionTexel(ivec2 texel, ivec4 rect, sampler2D normal,
                                 sampler2D depth)
{
    ivec2 pixel = getStochasticReflectionPixel(texel);
    if (any(lessThan(pixel, rect.xy)) || any(greaterThanEqual(pixel, rect.zw)))
        return false;
    return texelFetch(normal, pixel, 0).z >= SOLID_REFLECTION_MIN_GLOSS &&
        texelFetch(depth, pixel, 0).x != 0.0;
}   // isStochasticReflectionTexel

// ----------------------------------------------------------------------------
// The viewport rect of the camera in pixels (x0, y0, x1, y1 exclusive)
ivec4 getStochasticReflectionRect(vec4 viewport)
{
    ivec2 offset = ivec2(viewport.xy);
    return ivec4(offset, offset + ivec2(viewport.zw));
}   // getStochasticReflectionRect

// ----------------------------------------------------------------------------
// The reflection fades in from the lowest gloss which reflects, so a surface
// of a gloss just above it doesn't switch from the IBL to the traced
// radiance (the coverage is multiplied by it)
float getStochasticReflectionFade(float gloss)
{
    return smoothstep(SOLID_REFLECTION_MIN_GLOSS,
        SOLID_REFLECTION_MIN_GLOSS + 0.1, gloss);
}   // getStochasticReflectionFade

// ----------------------------------------------------------------------------
// Rays of a texel: the lobe of a mirror is a single direction, one ray has all
// of it, the roughest surface which reflects (perceptual roughness 0.5) gets
// max_rays
uint getStochasticReflectionRays(float perceptual_roughness, uint max_rays)
{
    float rays = ceil(float(max_rays) * perceptual_roughness /
        (1.0 - SOLID_REFLECTION_MIN_GLOSS));
    return clamp(uint(rays), 1u, max_rays);
}   // getStochasticReflectionRays

// ----------------------------------------------------------------------------
// PCG2D of Jarzynski and Olano 2020, "Hash Functions for GPU Rendering"
uvec2 hashStochasticReflection(uvec2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    return v;
}   // hashStochasticReflection

// ----------------------------------------------------------------------------
// The i-th uniform point in [0, 1)² of the rays of a texel: the R2 sequence
// (the 2D golden ratio, the points of a texel are spread evenly whatever their
// count) shifted by a random offset per texel (Cranley-Patterson rotation),
// so neighboring texels trace different directions which the resolve
// averages. It doesn't change with the frame: without a temporal filter a new
// pattern every frame would flicker, this one stays still with the camera
vec2 getStochasticReflectionSample(ivec2 texel, uint i)
{
    vec2 offset = vec2(hashStochasticReflection(uvec2(texel)) >> 8u) /
        16777216.0;
    return fract(offset + float(i) * vec2(0.7548776662, 0.5698402910));
}   // getStochasticReflectionSample

// ----------------------------------------------------------------------------
// The alpha (the roughness squared) of the lobe of a reflection. It's not
// perceptualRoughnessToRoughness: the minimum roughness of it keeps the
// highlights of the lights from being too small, but a lobe of that roughness
// is a few degrees wide, so a mirror (gloss 1) would reflect a noisy image
// (its reflection isn't filtered, see stochastic_reflection_resolve.comp)
float getStochasticReflectionAlpha(float perceptual_roughness)
{
    return perceptual_roughness * perceptual_roughness;
}   // getStochasticReflectionAlpha

// ----------------------------------------------------------------------------
// A world space direction of the lobe of the specular BRDF: the reflection of
// the view around a half vector of the visible normal distribution of GGX,
// sampled with the spherical caps of Eto and Tokuyoshi 2023, "Bounded VNDF
// Sampling for Smith-GGX Reflections" (fewer samples go below the surface
// than with the method of Heitz 2018). view is the direction from the surface
// to the eye, alpha the roughness (getStochasticReflectionAlpha), xi a
// uniform point in [0, 1)². The basis is the one of Duff et al. 2017,
// "Building an Orthonormal Basis, Revisited"
vec3 sampleStochasticReflection(vec3 view, vec3 normal, float alpha, vec2 xi)
{
    // A mirror, the lobe is a single direction (like the IsMirrorReflection
    // of FidelityFX SSSR)
    if (alpha < 1e-4)
        return reflect(-view, normal);
    float s = normal.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + normal.z);
    float b = normal.x * normal.y * a;
    vec3 t1 = vec3(1.0 + s * normal.x * normal.x * a, s * b, -s * normal.x);
    vec3 t2 = vec3(b, s + normal.y * normal.y * a, -normal.y);

    // The view in the space of the surface (z is the normal), above it (a
    // normal of the normal map can face away from the eye)
    vec3 i = vec3(dot(view, t1), dot(view, t2), dot(view, normal));
    i.z = max(i.z, 1e-4);
    i = normalize(i);

    // The view in the space where GGX of alpha is the unit hemisphere
    vec3 i_std = normalize(vec3(i.xy * alpha, i.z));
    float phi = 2.0 * 3.14159265359 * xi.x;
    // The spherical cap of the visible normals whose reflection is above the
    // surface (equations 5 and 6 of the paper)
    float cap_s = 1.0 + length(i.xy);
    float a2 = alpha * alpha;
    float s2 = cap_s * cap_s;
    float k = (1.0 - a2) * s2 / (s2 + a2 * i.z * i.z);
    float cap_b = k * i_std.z;
    float z = (1.0 - xi.y) * (1.0 + cap_b) - cap_b;
    float sin_theta = sqrt(clamp(1.0 - z * z, 0.0, 1.0));
    vec3 o_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    // The half vector, back in the space of GGX of alpha
    vec3 m_std = i_std + o_std;
    vec3 m = normalize(vec3(m_std.xy * alpha, m_std.z));
    vec3 o = reflect(-i, m);
    return normalize(o.x * t1 + o.y * t2 + o.z * normal);
}   // sampleStochasticReflection

// ----------------------------------------------------------------------------
// The radiance of the hits of the rays of a texel, which the trace shaders
// accumulate (addStochasticReflectionRay) and store
// (getStochasticReflectionOutput)
struct StochasticReflectionSum
{
    // The radiance of the hits weighted by their confidence, and the
    // confidence of the hits (0 for a miss, 1 for a sure hit) in w
    vec4 m_radiance;
    // The rays which were traced (the ones below the surface aren't)
    float m_rays;
};

// ----------------------------------------------------------------------------
void addStochasticReflectionRay(inout StochasticReflectionSum sum,
                                vec3 radiance, float confidence)
{
    // A float target can't store more than 65504, a hit can be brighter
    // (an emissive surface, the sun)
    sum.m_radiance += vec4(min(radiance, vec3(1024.0)) * confidence,
        confidence);
    sum.m_rays += 1.0;
}   // addStochasticReflectionRay

// ----------------------------------------------------------------------------
// The output of the trace of a texel: the average radiance premultiplied by
// the coverage and the coverage, both faded by the gloss
vec4 getStochasticReflectionOutput(StochasticReflectionSum sum, float gloss)
{
    if (sum.m_rays <= 0.0)
        return vec4(0.0);
    return sum.m_radiance * (getStochasticReflectionFade(gloss) / sum.m_rays);
}   // getStochasticReflectionOutput

#ifdef STOCHASTIC_REFLECTION_SURFACE
// ----------------------------------------------------------------------------
// The surface of the pixel of a texel which the trace shaders trace from, the
// includer declares u_normal and u_depth (the g-buffer)
struct StochasticReflectionSurface
{
    vec3 m_view_position;
    vec3 m_world_position;
    // From the surface to the eye
    vec3 m_view_eyedir;
    vec3 m_world_eyedir;
    vec3 m_world_normal;
    float m_gloss;
    float m_perceptual_roughness;
};

// ----------------------------------------------------------------------------
StochasticReflectionSurface getStochasticReflectionSurface(ivec2 pixel)
{
    StochasticReflectionSurface surface;
    vec4 normal_data = texelFetch(u_normal, pixel, 0);
    float depth = texelFetch(u_depth, pixel, 0).x;
    surface.m_view_position = getPosFromUVDepth(vec3(vec2(pixel) + 0.5,
        depth), u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
    surface.m_world_position = (u_camera.m_inverse_view_matrix *
        vec4(surface.m_view_position, 1.0)).xyz;
    surface.m_view_eyedir = -normalize(surface.m_view_position);
    surface.m_world_eyedir = normalize((u_camera.m_inverse_view_matrix *
        vec4(surface.m_view_eyedir, 0.0)).xyz);
    surface.m_world_normal = DecodeNormal(normal_data.xy);
    surface.m_gloss = normal_data.z;
    surface.m_perceptual_roughness = 1.0 - normal_data.z;
    return surface;
}   // getStochasticReflectionSurface
#endif

#ifdef STOCHASTIC_REFLECTION_TRACE
// The main of the trace shaders, which define the trace of a ray after they
// include this: the confidence of the ray of the world space direction from
// the surface (0 if it has no hit) and the radiance of its hit (hdr). They
// declare u_trace (level 0 of the trace target) and u_max_rays (the rays of
// the roughest surface, see getStochasticReflectionRays) before, and have 64
// threads in a group
float traceStochasticReflectionRay(StochasticReflectionSurface surface,
                                   ivec2 pixel, vec3 direction,
                                   out vec3 hdr);

// The sums of the rays of each thread of the group
shared StochasticReflectionSum s_stochastic_sums[64];

// ----------------------------------------------------------------------------
void main()
{
    // The texel of the thread and the first of its rays, the next ones are
    // STOCHASTIC_REFLECTION_THREADS rays later
    const uint lane = gl_LocalInvocationID.x;
    const uint first_ray = lane % STOCHASTIC_REFLECTION_THREADS;
    const uint index = (gl_WorkGroupID.x * STOCHASTIC_REFLECTION_THREADS +
        gl_WorkGroupID.y) * (64u / STOCHASTIC_REFLECTION_THREADS) +
        lane / STOCHASTIC_REFLECTION_THREADS;
    const bool valid = index < u_reflection_list.m_count;
    ivec2 texel = ivec2(0);
    StochasticReflectionSurface surface;
    StochasticReflectionSum sum = StochasticReflectionSum(vec4(0.0), 0.0);
    if (valid)
    {
        texel = unpackStochasticReflectionTexel(
            u_reflection_list.m_texels[index]);
        ivec2 pixel = getStochasticReflectionPixel(texel);
        surface = getStochasticReflectionSurface(pixel);
        float alpha = getStochasticReflectionAlpha(
            surface.m_perceptual_roughness);
        uint rays = getStochasticReflectionRays(
            surface.m_perceptual_roughness, u_max_rays);
        for (uint i = first_ray; i < rays; i += STOCHASTIC_REFLECTION_THREADS)
        {
            vec3 direction = sampleStochasticReflection(
                surface.m_world_eyedir, surface.m_world_normal, alpha,
                getStochasticReflectionSample(texel, i));
            // Below the surface of the normal map, the ray would hit the
            // mesh itself
            if (dot(direction, surface.m_world_normal) <= 0.0)
                continue;
            vec3 hdr;
            float confidence = traceStochasticReflectionRay(surface, pixel,
                direction, hdr);
            addStochasticReflectionRay(sum, hdr, confidence);
        }
    }
    s_stochastic_sums[lane] = sum;
    barrier();
    // The first thread of the texel sums the rays of all of them
    if (valid && first_ray == 0u)
    {
        for (uint i = 1u; i < STOCHASTIC_REFLECTION_THREADS; i++)
        {
            sum.m_radiance += s_stochastic_sums[lane + i].m_radiance;
            sum.m_rays += s_stochastic_sums[lane + i].m_rays;
        }
        imageStore(u_trace, texel, getStochasticReflectionOutput(sum,
            surface.m_gloss));
    }
}   // main
#endif
