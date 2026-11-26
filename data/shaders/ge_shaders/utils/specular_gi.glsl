// The specular GI of the glossy solid materials
// (GEVulkanSolidReflection with GEConfig::m_specular_gi), shared by
// its passes. It's the algorithm of FidelityFX SSSR: a rough surface doesn't
// reflect a single mirror ray, a few samples of the lobe of its specular BRDF
// (a GGX visible normal distribution) are traced and their radiance is stored
// raw, the lighting pass applies the split-sum BRDF of the surface to the
// average once (handlePBRDeferred), like the composite of FidelityFX SSSR. The
// trace is the screen space one over the depth buffer or the HiZ depth
// (specular_gi_ssr.comp) or the ray traced one
// (specular_gi_rt.comp).
//
// FidelityFX traces one ray per pixel and denoises it temporally. Here a
// texel traces a few rays, so the reflection is usable in a single frame, and
// the passes are:
//
//   classify : specular_gi_classify.comp, the glossy texels of the
//              half size target are added to a list (one texel is the 2x2
//              pixels of the g-buffer whose top left pixel, its
//              representative pixel getSpecularGIPixel, is
//              traced)
//   trace    : the texels of the list (indirect dispatch), a few rays each
//              (more for a rougher surface, one for a mirror), into level 0
//              of the trace target. SSSR packs neighboring texels into the
//              same group; the scene ray tracer uses 8 threads per texel.
//   mipmap   : specular_gi_mipmap.comp, levels 1 to 3 of the trace
//              target, the average of the glossy texels only
//   resolve  : specular_gi_resolve.comp, the texels of the list
//              (one thread each, its own indirect dispatch), the spatial
//              filter of the trace (the
//              prefilter of the reflection denoiser of FidelityFX, DNSR),
//              into the resolved target which the lighting pass reads
//   temporal : specular_gi_temporal.comp, with the frame cache
//              (GEConfig::m_frame_cache) only, the texels of the list: the
//              resolved trace is mixed with the history of the last frames
//              (the reprojection and the temporal resolve of FidelityFX DNSR,
//              see there), the resolve writes into another target then and
//              this one writes the target of the lighting pass. The rays of
//              a texel are other ones in every frame
//              (getSpecularGISample), which the history averages
//
// The trace target stores the radiance premultiplied by its coverage (rgb *
// alpha, the alpha is the confidence of the rays over their count) so the
// averages of the mipmap and the resolve are the averages of the radiance
// weighted by the coverage. Screen-space rays fill their missing radiance
// from the skybox; a surfel estimate carries its own coverage. It
// needs the alpha, so the targets are RGBA16F (half size: 2 bytes per pixel
// of the screen).
//
// Only the glossy texels are traced. The mipmap and the resolve test the
// g-buffer of a texel (isSpecularGITexel, the same test as the
// classify), so they never average a texel which wasn't traced in this frame,
// but a tap of the resolve is a bilinear sample of 4 texels, whose neighbors
// of the tested one can be any texel: the trace target is cleared every frame
// (GEVulkanSolidReflection::render), so they are 0 and never the output of an
// older frame (or the memory of another image, which can be NaN). The resolved
// target isn't cleared: the lighting pass only reads its glossy texels, which
// the resolve writes every frame (it's cleared once when it's created).
//
// The shader which includes it defines SOLID_REFLECTION_SET and
// SOLID_REFLECTION_BINDING if they're not 0, and includes
// utils/constants_utils.glsl, utils/pbr_utils.glsl,
// utils/camera.glsl (if it uses getSpecularGISurface),
// utils/unproject_position.glsl and ../utils/decodeNormal.frag before it

#ifndef SOLID_REFLECTION_SET
#define SOLID_REFLECTION_SET 0
#endif
// The binding of the trace target (level 0), the list is the next one
#ifndef SOLID_REFLECTION_BINDING
#define SOLID_REFLECTION_BINDING 0
#endif

// Levels of the trace target (the mipmap writes 1 to 3)
const int SPECULAR_GI_LEVELS = 4;

// The height of a viewport which the sizes in texels of the filters are for,
// they are scaled by the height of the viewport, so a filter covers the same
// part of the screen whatever its resolution (REFERENCE_HEIGHT of
// GEVulkanPostProcessing)
const float SPECULAR_GI_REFERENCE_HEIGHT = 1080.0;

// Threads per texel: SSSR uses the resolve dispatch, RT the wider trace dispatch.
#ifdef SPECULAR_GI_SSSR
const uint SPECULAR_GI_THREADS = 1u;
#else
const uint SPECULAR_GI_THREADS = 8u;
#endif

// The glossy texels (packSpecularGITexel), m_trace_dispatch is the
// indirect dispatch of the trace (x groups of 64 texels, y is
// 8 for RT) and m_resolve_dispatch the one of the screen-space trace,
// resolve and temporal filter (x groups of 64 texels, one thread each). y
// and z are written by GEVulkanSpecularGI before the classify, and
// m_frame: the frame of the temporal filter, never 0, or 0 without it (the
// rays of a texel are the same in every frame then)
layout(std430, set = SOLID_REFLECTION_SET,
    binding = SOLID_REFLECTION_BINDING + 1) buffer SpecularGIList
{
    uint m_count;
    uint m_trace_dispatch[3];
    uint m_resolve_dispatch[3];
    uint m_frame;
    uint m_texels[];
} u_reflection_list;

// ----------------------------------------------------------------------------
uint packSpecularGITexel(ivec2 texel)
{
    return uint(texel.x) | (uint(texel.y) << 16u);
}   // packSpecularGITexel

// ----------------------------------------------------------------------------
ivec2 unpackSpecularGITexel(uint packed)
{
    return ivec2(packed & 0xFFFFu, packed >> 16u);
}   // unpackSpecularGITexel

// ----------------------------------------------------------------------------
// The texel of the thread of the resolve, false if it has none (the last
// group isn't full)
bool getSpecularGITexel(out ivec2 texel)
{
    texel = ivec2(0);
    if (gl_GlobalInvocationID.x >= u_reflection_list.m_count)
        return false;
    texel = unpackSpecularGITexel(
        u_reflection_list.m_texels[gl_GlobalInvocationID.x]);
    return true;
}   // getSpecularGITexel

// ----------------------------------------------------------------------------
// The pixel of the g-buffer which a texel of the half size targets traces.
// It's the top left one of its 2x2 pixels, so a pixel of the screen with even
// coordinates is exactly at a texel and an odd one half way between two, which
// the upsample of the lighting pass uses (deferred_pbr.frag)
ivec2 getSpecularGIPixel(ivec2 texel)
{
    return texel * 2;
}   // getSpecularGIPixel

// ----------------------------------------------------------------------------
// The texel is traced: its pixel is in the viewport rect (x0, y0, x1
// exclusive, y1 exclusive, in pixels; a texel belongs to the viewport of its
// pixel, never to two), glossy and not the sky (reverse z, depth 0). The same
// test for the classify, the mipmap, the resolve and the upsample of the
// lighting pass
bool isSpecularGITexel(ivec2 texel, ivec4 rect, sampler2D normal,
                                 sampler2D depth)
{
    ivec2 pixel = getSpecularGIPixel(texel);
    if (any(lessThan(pixel, rect.xy)) || any(greaterThanEqual(pixel, rect.zw)))
        return false;
    return usesSpecularGISSR(texelFetch(normal, pixel, 0).z) &&
        texelFetch(depth, pixel, 0).x != 0.0;
}   // isSpecularGITexel

// ----------------------------------------------------------------------------
// The viewport rect of the camera in pixels (x0, y0, x1, y1 exclusive)
ivec4 getSpecularGIRect(vec4 viewport)
{
    ivec2 offset = ivec2(viewport.xy);
    return ivec4(offset, offset + ivec2(viewport.zw));
}   // getSpecularGIRect

// ----------------------------------------------------------------------------
// Rays of a texel: the lobe of a mirror is a single direction, one ray has all
// of it, the roughest surface traced by SSSR gets max_rays
uint getSpecularGIRays(float perceptual_roughness, uint max_rays)
{
    float rays = ceil(float(max_rays) * perceptual_roughness /
        SPECULAR_GI_MAX_SSSR_ROUGHNESS);
    return clamp(uint(rays), 1u, max_rays);
}   // getSpecularGIRays

// ----------------------------------------------------------------------------
// PCG2D of Jarzynski and Olano 2020, "Hash Functions for GPU Rendering"
uvec2 hashSpecularGI(uvec2 v)
{
    v = v * 1664525u + 1013904223u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    v.x += v.y * 1664525u;
    v.y += v.x * 1664525u;
    v ^= v >> 16u;
    return v;
}   // hashSpecularGI

#ifdef SPECULAR_GI_SSSR
bool chooseSurfelSpecularGI(ivec2 texel, float perceptual_roughness)
{
    if (!u_surfel_gi || perceptual_roughness <= SPECULAR_GI_SSSR_ONLY_ROUGHNESS)
        return false;
    float probability = smoothstep(SPECULAR_GI_SSSR_ONLY_ROUGHNESS,
        SPECULAR_GI_SURFEL_ONLY_ROUGHNESS, perceptual_roughness);
    uint rank = hashSpecularGI(uvec2(texel)).y + u_reflection_list.m_frame * 0x9E3779B9u;
    return float(rank >> 8u) * (1.0 / 16777216.0) < probability;
}
#endif

// ----------------------------------------------------------------------------
// The i-th uniform point in [0, 1)² of the rays of a texel: the R2 sequence
// (the 2D golden ratio, the points of a texel are spread evenly whatever their
// count) shifted by a random offset per texel (Cranley-Patterson rotation),
// so neighboring texels trace different directions which the resolve
// averages. Without the temporal filter (m_frame 0) it doesn't change with the
// frame: a new pattern every frame would flicker, this one stays still with
// the camera. With it, the sequence of a texel goes on in the next frame (the
// rays of the frame f are the points f * rays to f * rays + rays - 1), so the
// history averages more and more points of the same well spread sequence. The
// sequence is in 32 bit fixed point there (its wrap is the fract, exactly in
// every frame)
vec2 getSpecularGISample(ivec2 texel, uint i, uint rays)
{
    uvec2 seed = hashSpecularGI(uvec2(texel));
    if (u_reflection_list.m_frame == 0u)
    {
        vec2 offset = vec2(seed >> 8u) / 16777216.0;
        return fract(offset + float(i) * vec2(0.7548776662, 0.5698402910));
    }
    // 0.7548776662 * 2^32 and 0.5698402910 * 2^32
    const uvec2 step = uvec2(3242174889u, 2447445414u);
    uvec2 point = seed + (u_reflection_list.m_frame * rays + i) * step;
    return vec2(point >> 8u) / 16777216.0;
}   // getSpecularGISample

// ----------------------------------------------------------------------------
// The alpha (the roughness squared) of the lobe of a reflection. It's not
// perceptualRoughnessToRoughness: the minimum roughness of it keeps the
// highlights of the lights from being too small, but a lobe of that roughness
// is a few degrees wide, so a mirror (gloss 1) would reflect a noisy image
// (its reflection isn't filtered, see specular_gi_resolve.comp)
float getSpecularGIAlpha(float perceptual_roughness)
{
    return perceptual_roughness * perceptual_roughness;
}   // getSpecularGIAlpha

// ----------------------------------------------------------------------------
// A world space direction of the lobe of the specular BRDF: the reflection of
// the view around a half vector of the visible normal distribution of GGX,
// sampled with the spherical caps of Eto and Tokuyoshi 2023, "Bounded VNDF
// Sampling for Smith-GGX Reflections" (fewer samples go below the surface
// than with the method of Heitz 2018). view is the direction from the surface
// to the eye, alpha the roughness (getSpecularGIAlpha), xi a
// uniform point in [0, 1)². The basis is the one of Duff et al. 2017,
// "Building an Orthonormal Basis, Revisited"
vec3 sampleSpecularGI(vec3 view, vec3 normal, float alpha, vec2 xi)
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
}   // sampleSpecularGI

// ----------------------------------------------------------------------------
// The radiance of the hits of the rays of a texel, which the trace shaders
// accumulate (addSpecularGIRay) and store
// (getSpecularGIOutput)
struct SpecularGISum
{
    // The radiance of the hits weighted by their confidence, and the
    // confidence of the hits (0 for a miss, 1 for a sure hit) in w
    vec4 m_radiance;
    // The rays which were traced (the ones below the surface aren't)
    float m_rays;
    // The distance of the hits weighted by their confidence (the length of
    // the ray of the reflection, for the temporal filter)
    float m_distance;
    // The confidence of the hits on a node which can move
    float m_moving;
#ifdef SPECULAR_GI_SSSR
    vec3 m_sky;
#endif
#ifdef SPECULAR_GI_KARIS
    // m_radiance with every ray weighted by 1 / (1 + its luminance) (the
    // Karis average, see addSpecularGIRay), and the sum of the
    // weights of all rays, hits and misses
    vec4 m_karis;
    float m_karis_weight;
#endif
};

// ----------------------------------------------------------------------------
void addSpecularGIRay(inout SpecularGISum sum,
                                vec3 radiance, float confidence,
                                float hit_distance, bool moving)
{
    // A float target can't store more than 65504, a hit can be brighter
    // (an emissive surface, the sun)
    vec3 hit = min(radiance, vec3(1024.0));
    sum.m_radiance += vec4(hit * confidence, confidence);
    sum.m_rays += 1.0;
    sum.m_distance += hit_distance * confidence;
    sum.m_moving += moving ? confidence : 0.0;
#ifdef SPECULAR_GI_KARIS
    float hit_weight = confidence / (1.0 + dot(hit,
        vec3(0.2126, 0.7152, 0.0722)));
    sum.m_karis += vec4(hit, 1.0) * hit_weight;
    sum.m_karis_weight += hit_weight + (1.0 - confidence) * 1.0;
#endif
}   // addSpecularGIRay

// ----------------------------------------------------------------------------
// The length of the rays of a texel to their hits (their average, weighted by
// the confidence of the hits), 0 if no ray has a hit. The temporal filter
// reprojects the reflected position with it. It's negative if a ray hit a node
// which can move: there are no motion vectors, the history of its reflection
// is somewhere else, so the temporal filter keeps few frames of it
float getSpecularGIRayLength(SpecularGISum sum)
{
    if (sum.m_radiance.w <= 0.0)
        return 0.0;
    float ray_length = max(sum.m_distance / sum.m_radiance.w, 1e-3);
    return sum.m_moving > 0.0 ? -ray_length : ray_length;
}   // getSpecularGIRayLength

// ----------------------------------------------------------------------------
// The output of the trace of a texel: the average radiance premultiplied by
// the coverage and the coverage
vec4 getSpecularGIOutput(SpecularGISum sum)
{
    if (sum.m_rays <= 0.0)
        return vec4(0.0);
#ifdef SPECULAR_GI_KARIS
    if (sum.m_karis_weight <= 0.0)
        return vec4(0.0);
    return sum.m_karis / sum.m_karis_weight;
#else
    return sum.m_radiance / sum.m_rays;
#endif
}   // getSpecularGIOutput

#ifdef SPECULAR_GI_SURFACE
// ----------------------------------------------------------------------------
// The surface of the pixel of a texel which the trace shaders trace from, the
// includer declares u_normal and u_depth (the g-buffer)
struct SpecularGISurface
{
    vec3 m_view_position;
    vec3 m_world_position;
    // From the surface to the eye
    vec3 m_view_eyedir;
    vec3 m_world_eyedir;
    vec3 m_world_normal;
    float m_perceptual_roughness;
};

// ----------------------------------------------------------------------------
SpecularGISurface getSpecularGISurface(ivec2 pixel)
{
    SpecularGISurface surface;
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
    surface.m_perceptual_roughness = 1.0 - normal_data.z;
    return surface;
}   // getSpecularGISurface
#endif

#ifdef SPECULAR_GI_TRACE
// The main of the trace shaders, which define the trace of a ray after they
// include this: the confidence of the ray of the world space direction from
// the surface (0 if it has no hit), the radiance of its hit (hdr), its
// distance from the surface (hit_distance, in world units) and if it's a node
// which can move (moving). They
// declare u_trace (level 0 of the trace target) and u_max_rays (the rays of
// the roughest surface, see getSpecularGIRays) before, and have 64
// threads in a group
float traceSpecularGIRay(SpecularGISurface surface,
                                   ivec2 pixel, vec3 direction,
                                   out vec3 hdr, out float hit_distance,
                                   out bool moving);

// The length of the rays of the texels (getSpecularGIRayLength),
// only written with the temporal filter. It's binding 7 of the set of the
// reflection, the screen space trace has it in GVDFP_HDR after the list
#ifndef SPECULAR_GI_RAY_LENGTH_BINDING
#define SPECULAR_GI_RAY_LENGTH_BINDING 7
#endif
layout(set = SOLID_REFLECTION_SET,
    binding = SPECULAR_GI_RAY_LENGTH_BINDING, r32f) uniform
    writeonly image2D u_ray_length;

// The sums of the rays of each thread of the group
shared SpecularGISum s_specular_gi_sums[64];

// ----------------------------------------------------------------------------
void main()
{
    // The texel of the thread and the first of its rays, the next ones are
    // SPECULAR_GI_THREADS rays later
    const uint lane = gl_LocalInvocationID.x;
    const uint first_ray = lane % SPECULAR_GI_THREADS;
    const uint index = (gl_WorkGroupID.x * SPECULAR_GI_THREADS +
        gl_WorkGroupID.y) * (64u / SPECULAR_GI_THREADS) +
        lane / SPECULAR_GI_THREADS;
    const bool valid = index < u_reflection_list.m_count;
    ivec2 texel = ivec2(0);
    SpecularGISurface surface;
#ifdef SPECULAR_GI_SSSR
    bool use_surfel = false;
#endif
#ifdef SPECULAR_GI_SSSR
    SpecularGISum sum = SpecularGISum(vec4(0.0), 0.0, 0.0, 0.0, vec3(0.0));
#elif defined(SPECULAR_GI_KARIS)
    SpecularGISum sum = SpecularGISum(vec4(0.0), 0.0,
        0.0, 0.0, vec4(0.0), 0.0);
#else
    SpecularGISum sum = SpecularGISum(vec4(0.0), 0.0,
        0.0, 0.0);
#endif
    if (valid)
    {
        texel = unpackSpecularGITexel(
            u_reflection_list.m_texels[index]);
        ivec2 pixel = getSpecularGIPixel(texel);
        surface = getSpecularGISurface(pixel);
        float alpha = getSpecularGIAlpha(
            surface.m_perceptual_roughness);
        uint rays = getSpecularGIRays(
            surface.m_perceptual_roughness, u_max_rays);
#ifdef SPECULAR_GI_SSSR
        use_surfel = chooseSurfelSpecularGI(texel, surface.m_perceptual_roughness);
        if (use_surfel)
            rays = 0u;
#endif
        for (uint i = first_ray; i < rays; i += SPECULAR_GI_THREADS)
        {
            vec3 direction = sampleSpecularGI(
                surface.m_world_eyedir, surface.m_world_normal, alpha,
                getSpecularGISample(texel, i, rays));
            // Below the surface of the normal map, the ray would hit the
            // mesh itself
            if (dot(direction, surface.m_world_normal) <= 0.0)
                continue;
            vec3 hdr;
            float hit_distance;
            bool moving;
            float confidence = traceSpecularGIRay(surface, pixel,
                direction, hdr, hit_distance, moving);
            addSpecularGIRay(sum, hdr, confidence, hit_distance,
                moving);
#ifdef SPECULAR_GI_SSSR
            if (confidence < 1.0)
                sum.m_sky += (1.0 - confidence) * textureLod(u_skybox, direction, 0.0).rgb;
#endif
        }
    }
    if (SPECULAR_GI_THREADS > 1u)
    {
        s_specular_gi_sums[lane] = sum;
        barrier();
    }
    // The first thread of the texel sums the rays of all of them
    if (valid && first_ray == 0u)
    {
        for (uint i = 1u; i < SPECULAR_GI_THREADS; i++)
        {
            sum.m_radiance += s_specular_gi_sums[lane + i].m_radiance;
            sum.m_rays += s_specular_gi_sums[lane + i].m_rays;
            sum.m_distance += s_specular_gi_sums[lane + i].m_distance;
            sum.m_moving += s_specular_gi_sums[lane + i].m_moving;
#ifdef SPECULAR_GI_KARIS
            sum.m_karis += s_specular_gi_sums[lane + i].m_karis;
            sum.m_karis_weight += s_specular_gi_sums[lane + i].m_karis_weight;
#endif
        }
        vec4 result = getSpecularGIOutput(sum);
        float ray_length = getSpecularGIRayLength(sum);
#ifdef SPECULAR_GI_SSSR
        if (use_surfel)
        {
            result = getSurfelSpecularGI(texelFetch(u_surfel_gi_texture, getSpecularGIPixel(texel), 0));
            ray_length = 0.001;
        }
        else if (sum.m_rays > 0.0)
        {
            float misses = sum.m_rays - sum.m_radiance.a;
            if (misses > 0.0)
            {
                vec3 direction = reflect(-surface.m_world_eyedir, surface.m_world_normal);
                result.rgb += (misses / sum.m_rays) * resolveSpecularGISky(sum.m_sky / misses,
                    surface.m_world_position, surface.m_world_normal, direction);
            }
            result.a = 1.0;
        }
#endif
        imageStore(u_trace, texel, result);
        if (u_reflection_list.m_frame != 0u)
        {
            imageStore(u_ray_length, texel, vec4(ray_length));
        }
    }
}   // main
#endif
