// The slices, sectors and visibility bitmasks of the estimator of
// GEVulkanSSAO, shared by its two shaders: ssao.comp (the visibility) and
// ssil.comp (the indirect light). They use the same sampling so an object which
// the occlusion sees through also lets its light through.
//
// The estimator is VBAO (Visibility Bitmask Ambient Occlusion, Therrien et al,
// "Screen Space Indirect Lighting with Visibility Bitmask"), also extended to
// indirect light by the same paper. Every slice around the view direction is
// divided into SECTOR_COUNT angular sectors, and a sample covers the sectors
// between its front horizon and the horizon of its back face (the sample is
// the surface of an object with a given thickness). This allows light to pass
// behind thin surfaces and takes several occluders of the same slice into
// account, unlike the two horizon angles of GTAO / SAO.
//
// Included after utils/ssao_common.glsl (pc, SSAO_SKY_Z) and by those two
// shaders only
#include "camera.glsl"

// Samples on each side of a slice, and sectors (bits) of the visibility mask
// of the upper hemisphere of a slice: two sectors for every sample (the same
// as Bevy), so a sample always covers at least one of them
const int SAMPLES_PER_SLICE_SIDE = 4;
const int SECTOR_COUNT = SAMPLES_PER_SLICE_SIDE * 2;
// Slices around the view direction: a slice and the one at PI from it are the
// same line, and both sides of every slice are sampled, so the half circle is
// enough
const int SLICE_COUNT = 8;
const float PI = 3.14159265;
const float HALF_PI = 0.5 * PI;
// A sample is the surface of an object whose thickness is this part of the
// radius (the paper uses a small constant, which is 0.05 to 0.25 of the radius
// in its examples), so it also works with a radius changed by the user
const float THICKNESS_RATIO = 0.2;
// Samples which are this close to the tangent plane of the surface (as a part
// of the radius) are ignored: they can't occlude, and the projection of their
// direction on the slice plane has an error which darkens the flat surfaces
const float NORMAL_BIAS_RATIO = 0.01;
// The offset (in pixels) of a sample which fetches level 0 of the depth chain
// is up to 2^3.3 = 10, the levels after it are used for the farther ones (the
// texels of level N cover 2^N pixels), the same as XeGTAO and Bevy
const float MIP_BIAS = 3.3;
const int MAX_MIP = 3;
// Largest radius on the screen, as a part of the height of the viewport (the
// samples of a very close surface would otherwise be all over the screen)
const float MAX_RADIUS_RATIO = 0.1;

// Radius of the effect on the screen in pixels, the farther the surface is
// the smaller it is (nothing is sampled if it's less than a pixel), and it's
// limited so the samples of a very close surface are not all over the screen
float radiusInPixels(float radius, float z)
{
    const float radius_px = radius * abs(u_camera.m_projection_matrix[1][1]) *
        0.5 * u_camera.m_viewport.w / z;
    return min(radius_px, MAX_RADIUS_RATIO * u_camera.m_viewport.w);
}

// View space ray (z of 1 or -1, so the position is the ray multiplied by the
// linear depth), linear in the position on the viewport
vec3 g_ray_origin;
vec3 g_ray_x;
vec3 g_ray_y;

vec3 unitDepthRay(vec2 ndc)
{
    vec4 v = u_camera.m_inverse_projection_matrix * vec4(ndc, 1.0, 1.0);
    vec3 p = v.xyz / v.w;
    return p / abs(p.z);
}

void initRays()
{
    g_ray_origin = unitDepthRay(vec2(0.0));
    g_ray_x = unitDepthRay(vec2(1.0, 0.0)) - g_ray_origin;
    g_ray_y = unitDepthRay(vec2(0.0, 1.0)) - g_ray_origin;
}

// Position is in pixels of the g-buffer, at the center of the pixel
vec3 viewRay(vec2 position)
{
    vec2 ndc = (position - u_camera.m_viewport.xy) /
        u_camera.m_viewport.zw * 2.0 - 1.0;
    return g_ray_origin + ndc.x * g_ray_x + ndc.y * g_ray_y;
}

// Index of a pixel of a 64x64 tile in a Hilbert curve, the same as the LUT of
// Bevy (https://www.shadertoy.com/view/3tB3z3), which the noise below uses to
// be well distributed in the screen space
uint hilbertIndex(uvec2 p)
{
    uint index = 0u;
    for (uint level = 32u; level > 0u; level >>= 1u)
    {
        const uint region_x = (p.x & level) != 0u ? 1u : 0u;
        const uint region_y = (p.y & level) != 0u ? 1u : 0u;
        index += level * level * ((3u * region_x) ^ region_y);
        if (region_y == 0u)
        {
            if (region_x == 1u)
            {
                p.x = 63u - p.x;
                p.y = 63u - p.y;
            }
            const uint swap = p.x;
            p.x = p.y;
            p.y = swap;
        }
    }
    return index;
}

// Rotation of the slices and offset of the samples of the pixel, from the R2
// sequence of Bevy
// http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences
vec2 loadNoise(ivec2 pixel)
{
    const uint index = hilbertIndex(uvec2(pixel) & 63u);
    return fract(0.5 + float(index) *
        vec2(0.75487766624669276005, 0.5698402909980532659114));
}

// Sets the sectors between the two horizons of a slice (both in [0, 1], where
// 0 and 1 are the horizons of the slice) at once, so any number of them costs
// the same. The same as insertBits() in the code of Bevy, but it never wraps
// around at the width of the mask, which would set the sectors of the other
// side
uint updateSectors(float min_horizon, float max_horizon,
                   float samples_per_slice, uint bitmask)
{
    const uint start = min(uint(min_horizon * samples_per_slice), 32u);
    const uint count = min(uint(ceil((max_horizon - min_horizon) *
        samples_per_slice)), 32u - start);
    if (count == 0u)
        return bitmask;
    return bitmask | ((0xFFFFFFFFu >> (32u - count)) << start);
}

// Marks the sectors which a sample covers: from the angle of its front face to
// the angle of its back face, relative to the view vector and to the normal of
// the slice (the same as processSample() in the code of Bevy). The sample is
// the surface of an object with a thickness along the view vector, so light
// can pass behind it
void processSample(vec3 delta_position, vec3 view, float thickness,
                   float sampling_direction, float n, float samples_per_slice,
                   inout uint bitmask)
{
    // The back face of the sample, behind it from the point of view of the
    // camera
    const vec3 delta_back = delta_position - view * thickness;
    const float front_length = length(delta_position);
    const float back_length = length(delta_back);
    if (front_length < 1.0e-5 || back_length < 1.0e-5)
        return;

    // Angles of the front and the back of the sample: unsigned, so the same
    // on both sides of the view vector, and in [0, 1] where 0 and 1 are the
    // horizons of the slice. They are then made relative to the normal (which
    // is at n) and to the side of the view vector of the sample
    vec2 horizons = vec2(
        acos(clamp(dot(delta_position / front_length, view), -1.0, 1.0)),
        acos(clamp(dot(delta_back / back_length, view), -1.0, 1.0))) *
        (1.0 / PI);
    horizons = clamp(vec2(n) - sampling_direction * horizons, 0.0, 1.0);
    // The samples of the other side of the view vector are seen in the
    // opposite direction, so their front is the far horizon
    if (sampling_direction >= 0.0)
        horizons = horizons.yx;

    bitmask = updateSectors(horizons.x, horizons.y, samples_per_slice, bitmask);
}

// The samples which are this close to the tangent plane of the surface don't
// occlude anything, they are on the surface itself. The direction of a sample
// projected on the slice plane is also where the error of the reconstruction
// of its position is the largest, so flat surfaces in the light would
// otherwise be darkened
bool isOccluder(vec3 delta, vec3 normal, float normal_bias)
{
    return length(delta) > 1.0e-5 && dot(delta, normal) >= normal_bias;
}

// View space position of the sample at the pixel (full size coordinates; the
// depth of the closest pixel of the rect is used if it's outside): it's
// reconstructed where the sample is asked for, with the depth of the texel of
// the level which was fetched (the same as the bilinear sampling of the depth
// chain in XeGTAO and Bevy) and not at the center of that texel
vec3 fetchSamplePosition(ivec2 sample_pixel, int mip, out float z)
{
    const ivec4 level_rect = ssaoLevelRect(pc.m_rect, mip);
    const ivec2 level_pixel = clamp(
        clamp(sample_pixel, pc.m_rect.xy, pc.m_rect.zw - 1) >> mip,
        level_rect.xy, level_rect.zw - 1);
    z = texelFetch(u_depth, level_pixel, mip).x;
    return viewRay(vec2(sample_pixel) + 0.5) * z;
}

// Direction of the samples of a slice (view space) and the angle of the
// normal in the slice, which is the center of its bitmask. omega is the
// direction of the slice in the viewport (pixels), where the samples of both
// sides are offset. Returns false if the slice is undefined for this pixel
bool vbaoSampleDirection(vec3 view, vec3 normal, vec2 omega, out float n)
{
    // The offset of the samples is along omega in pixels, and g_ray_x and
    // g_ray_y are the rays of the camera for a unit of NDC, so the direction
    // is the same combination of them (the Y flip and a rotation of the
    // projection are already in the rays)
    const vec2 ndc_dir = omega * (2.0 / u_camera.m_viewport.zw);
    const vec3 direction = ndc_dir.x * g_ray_x + ndc_dir.y * g_ray_y;
    // The tangent of the viewport at the pixel, the slice plane contains it
    // and the view vector
    const vec3 tangent = direction - view * dot(direction, view);
    const float tangent_len = length(tangent);
    // The direction of the samples is parallel to the view vector, the slice
    // plane is undefined
    if (tangent_len < 1.0e-6)
        return false;
    const vec3 slice_n = cross(tangent / tangent_len, view);
    // The normal projected in the slice plane: its angle relative to the view
    // vector is the center of the mask of the slice
    const vec3 proj_n = normal - slice_n * dot(normal, slice_n);
    const float proj_n_len = length(proj_n);
    // The slice plane is tangent to the surface, nothing is above it and the
    // angle of the normal is undefined
    if (proj_n_len < 1.0e-4)
        return false;
    const vec3 proj_normal = proj_n / proj_n_len;
    // The same as Bevy: where the normal points to in the slice, and the
    // angle of the normal relative to the view vector (clamped to the
    // hemisphere which is above the surface)
    const float sign_norm = sign(dot(tangent, proj_normal));
    const float cos_norm = clamp(dot(proj_normal, view), 0.0, 1.0);
    n = (HALF_PI - sign_norm * acos(cos_norm)) * (1.0 / PI);
    return true;
}
