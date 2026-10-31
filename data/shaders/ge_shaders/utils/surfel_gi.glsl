// The pool of surfels of the surfel based global illumination (GEVulkanSurfelGI,
// the same as SurfelPlus, Wang et al. 2025):
//
// surfel_gi_prepare.comp: clears the cells of the grid and the counters of the
// frame, and on the first frame the stack of the free surfels.
// surfel_gi_update.comp: the surfels which are alive follow the vertices of the
// triangles they are attached to (GEVulkanRTScene keeps the transformation of
// the node of every geometry, so a surfel of a kart follows it), update their
// radius from their distance to the camera, register themselves in the cells
// they cover (a surfel can be in several of the 27 cells around it) and request
// the rays of the frame.
// surfel_gi_cell_info.comp and surfel_gi_cell_to_surfel.comp: the offsets and
// the lists of the surfels of every cell.
// surfel_gi_trace.comp: a path of up to 3 segments per ray of the surfel, which
// ends at the first hit the field of the surfels covers (the light of the sun,
// the point lights and the sky on the way), with the directions of the rays
// sampled with the R3 low discrepancy sequence and guided by the irradiance map
// of the surfel. A hit which no surfel covers spawns a sleeping surfel there
// (W298 SurfelGI), so the field grows where the rays go and not only where the
// camera looks. The visibility of the sun at a surfel is traced by its first
// ray of the frame, and the hits of the paths take it from the field instead of
// tracing a shadow ray of their own.
// surfel_gi_integrate.comp: one 32 lane group per surfel which has rays: the
// multiscale mean estimator (Ray Tracing Gems chapter 25) of its rays, the
// accumulation of their radiance in its 6x6 irradiance map (which guides its
// rays) and of their distances in its 6x6 depth moments (the visibility of
// the surfel, DDGI and W298 SurfelGI).
// surfel_gi_share.comp: the radiance of the surfels of its cell which reach
// every surfel merged in the mean of its estimator (the irradiance sharing of
// GIBS), sampled with a bounded random walk of the list and with a window
// which follows the noise of the surfel.
// surfel_gi_generate.comp: the irradiance of the pixels (the weighted mean of
// the surfels of their cell), and the spawn of the surfels where the coverage
// is low (the pool is flooded at the start, then the surfels of the over
// covered pixels are removed and the holes are filled).
//
// The cells are camera relative: the inner cube of the grid (2 * m_half_d world
// units, m_delta per cell) is uniform, and the 6 frustums around it are larger
// and larger layers (the ratio SURFEL_CELL_P), so a surfel far from the camera
// is as large as its cell. A surfel keeps its history when the camera moves
// (the cells are only a way to find the surfels around a position), and the
// surfels of an object keep their radiance when it moves: a surface which turns
// under its surfels (a wheel) is not the surface their samples came from, so
// surfel_gi_update.comp resets the map and widens the estimator of the surfels
// it turns.
//
// There is no screen space temporal filter: the mean of a surfel is stable
// (the multiscale mean estimator) and it only changes when its lighting
// changes, so the irradiance of a pixel doesn't flicker. A new surfel is
// seeded with the irradiance of the surfels around it, so it doesn't pop.

// Same layout as GEVulkanSurfelGI::MSMEData (std430, 48 bytes)
struct MSMEData
{
    vec3 m_mean;
    // How much the mean is allowed to catch up with the samples, 0 is frozen
    // and 1 is a full update (see surfelMSME)
    float m_vbbr;
    vec3 m_short_mean;
    // Number of standard deviations the mean is from the short mean
    float m_inconsistency;
    vec3 m_variance;
    // Effective number of samples the mean averages (bounded by
    // SURFEL_MSME_N_MAX): the weight of a packet in the mean is
    // 1 / (m_count + 1) at least (see surfelMSME), and the relative noise of
    // the mean, which sets the sharing window, is the one of its rays over
    // the square root of it (see surfelMeanNoise)
    float m_count;
};

// Same layout as GEVulkanSurfelGI::Surfel (std430, 144 bytes)
struct Surfel
{
    // World position and radius of the surfel
    vec3 m_position;
    float m_radius;
    // World normal
    vec3 m_normal;
    // Key of the node of the triangle (0 for the static geometry, which is
    // already in world space and doesn't move), see utils/rt_scene.glsl
    uint m_node_key;
    // Index of the geometry in the instance of the node, index of the triangle
    // in the geometry, and the 2 barycentrics of the attachment (2 16 bit
    // unorms)
    uint m_geometry;
    uint m_primitive;
    uint m_bary;
    // Rays of the frame in the list of rays
    uint m_ray_offset;
    uint m_ray_count;
    // Frames before the surfel is recycled, reset when it's seen
    uint m_life;
    // Number of frames since the surfel was created (it's faded in)
    uint m_frame;
    uint m_status;
    MSMEData m_msme;
    // The mean of the samples of the surfel at the end of the previous frame
    // (surfel_gi_update.comp copies the mean of the estimator here at the start
    // of the frame, and surfel_gi_share.comp only writes the estimator), the
    // input of the irradiance of the pixels of surfel_gi_generate.comp and of
    // the field of surfelGatherIrradiance (the "radiance" of SurfelPlus)
    vec3 m_radiance;
    // Frame the surfel was last seen in
    uint m_last_seen;
    // floatBitsToUint of the sum of the 6x6 irradiance map of the surfel, with
    // 1 in its lowest bit when the map is full (every texel was hit): the rays
    // of the surfel are guided by it then (see surfelSampleDirection), and 0
    // means that the surfel was never integrated
    uint m_irradiance;
    // The visibility of the sun at the surfel, traced by its first ray of the
    // frame (-1 while it was never traced): the hits of the paths which the
    // field covers take it instead of tracing a shadow ray (see
    // surfelGatherIrradiance)
    float m_sun;
    // Rays the surfel traced since it was created, the index of the R3 low
    // discrepancy sequence of its next ray (see surfelR3): the sequences of the
    // rays of the frames continue each other, so the directions of its few
    // rays are as uniform as the ones of a fast sequence
    uint m_ray_index;
    uint m_pad;
};

// Same layout as GEVulkanSurfelGI::SurfelLight (std430, 64 bytes): the fields
// of a surfel which the passes which read every surfel of a cell need (the
// irradiance of the pixels, the sharing and the last hit of a path), so they
// don't read the whole 144 bytes Surfel (and its bounds) for each of them
struct SurfelLight
{
    // World position and radius of the surfel
    vec4 m_position_radius;
    // World normal and the number of frames since the surfel was created (it's
    // faded in)
    vec4 m_normal_frame;
    // The mean of the samples at the end of the previous frame (the "radiance"
    // of SurfelPlus)
    vec3 m_radiance;
    // The sums of the rows of its 6x6 irradiance map, 2 half floats each: the
    // first level of the walk of surfelSampleDirection
    uint m_row_sums_01;
    uint m_row_sums_23;
    uint m_row_sums_45;
    // The range of the depth moments of the surfel (the distances of its rays
    // are clamped to it), 0 while it has none (see surfelVisibility)
    float m_depth_range;
    // The visibility of the sun at the surfel (the direct light cache of the
    // reference article: a hit which the field covers takes the shadow of the
    // sun from its surfels instead of tracing a ray)
    float m_sun;
};

// Same layout as GEVulkanSurfelGI::SurfelCounter (std430, 32 bytes)
struct SurfelCounter
{
    // Number of surfels which are alive
    int m_alive;
    // Number of entries of the stack of the free surfels (u_dead[0..m_free-1])
    int m_free;
    // Number of (surfel, cell) pairs of the frame, used to compute the offsets
    // of the cells
    uint m_cell_surfels;
    // Number of rays of the frame
    uint m_rays;
    // Number of surfels which have rays in the frame (u_active)
    uint m_active;
    // Rays the surfels asked for in this frame and in the previous one (the
    // rays of a frame are scaled down when the previous one asked for more
    // than SURFEL_RAY_TARGET)
    uint m_ray_requests;
    uint m_last_ray_requests;
    // Sleeping surfels the tracing of the frame spawned
    uint m_sleeping_spawns;
};

struct CellInfo
{
    // First index of the surfels of the cell in the list, and their number
    uint m_offset;
    uint m_count;
};

layout(set = 2, binding = 0) uniform sampler2D u_normal;
layout(set = 2, binding = 1) uniform sampler2D u_depth;
layout(set = 2, binding = 2, rgba16f) uniform writeonly image2D u_gi;
layout(set = 2, binding = 3, std430) buffer SurfelBuffer
{
    Surfel u_surfels[];
};
layout(set = 2, binding = 4, std430) buffer SurfelCounterBuffer
{
    SurfelCounter u_counter;
};
// Stack of the ids of the surfels which are not alive
layout(set = 2, binding = 5, std430) buffer SurfelDeadBuffer
{
    uint u_dead[];
};
layout(set = 2, binding = 6, std430) buffer SurfelCellBuffer
{
    CellInfo u_cells[];
};
layout(set = 2, binding = 7, std430) buffer CellSurfelBuffer
{
    uint u_cell_surfels[];
};
// Rays of the frame: the id of the surfel of every ray (written by
// surfel_gi_update.comp, read by surfel_gi_trace.comp) and the radiance of the
// result
layout(set = 2, binding = 8, std430) buffer SurfelRayIdBuffer
{
    uint u_ray_ids[];
};
// The radiance of the rays of the frame, and the direction of the ray
// (surfel_gi_trace.comp writes them, surfel_gi_integrate.comp reads them): the
// radiance is the one of the hit, without the cosine and the pdf of the
// direction (the sample of the irradiance is radiance * cosine / (PI * pdf)),
// and the w is the octahedral uv of the direction of the ray (2 half floats,
// the same space as the map)
layout(set = 2, binding = 9, std430) buffer SurfelRayResultBuffer
{
    vec4 u_ray_results[];
};
// The 6x6 octahedral irradiance map of every surfel (3072x1536 pixels, a 6x6
// tile for every surfel of the pool): the radiance of the rays of a surfel is
// accumulated in the tile of its direction, so its bright directions are
// sampled more (the ray guiding of SurfelPlus, see surfel_gi_trace.comp)
layout(set = 2, binding = 10) uniform sampler2D u_surfel_irradiance;
layout(set = 2, binding = 11, r16f) uniform image2D u_surfel_irradiance_map;
// The compact copy of the fields of the surfels which the passes of the cells
// read (see SurfelLight), written by surfel_gi_update.comp,
// surfel_gi_integrate.comp and surfel_gi_generate.comp
layout(set = 2, binding = 12, std430) buffer SurfelLightBuffer
{
    SurfelLight u_surfel_lights[];
};
// The data of every ray of the frame (surfel_gi_trace.comp writes it,
// surfel_gi_integrate.comp reads it). x is the weight of the sample of the
// irradiance of the ray, cosine / (PI * pdf), the density of
// surfelSampleDirection which the direction of the ray was sampled with, so the
// sample is radiance * weight and the sample doesn't depend on the map (the
// integration writes the map after the rays were traced). y is the distance of
// the first hit of the ray (RT_RAY_MAX for a miss), the depth of the texel of
// its direction in the depth moments of the surfel
layout(set = 2, binding = 13, std430) buffer SurfelRayDataBuffer
{
    vec2 u_ray_data[];
};
// The VkDispatchIndirectCommands of the frame: the groups of the tracing (the
// number of rays, see surfel_gi_cell_info.comp), then the groups of the
// cooperative integration (x = the surfels which have rays, at most 256, y =
// ceil(x / 256), so a group is one surfel and its ID is x + y * 256), then the
// groups of the sharing (one 64 thread group for every surfel which has rays)
layout(set = 2, binding = 14, std430) buffer SurfelDispatchBuffer
{
    uint u_dispatch[];
};
// The depth moments of the surfels (rg32f, a 6x6 tile for every surfel of the
// same size as the irradiance map): the mean distance of the hits of a texel
// and its second moment. The visibility of a surfel from a point is the
// Chebyshev bound of its moments (DDGI, W298 SurfelGI), see surfelVisibility
layout(set = 2, binding = 15) uniform sampler2D u_surfel_depth;
layout(set = 2, binding = 16, rg32f) uniform image2D u_surfel_depth_map;
// The IDs of the surfels which have rays in the frame (SURFEL_NONE if a surfel
// was recycled after it asked for them), surfel_gi_update.comp writes them and
// surfel_gi_integrate.comp reads them: one group of integration for every of
// them instead of one for every slot of the pool
layout(set = 2, binding = 17, std430) buffer SurfelActiveBuffer
{
    uint u_active[];
};

// Same layout as GEVulkanSurfelGI::SurfelGIPushConstants (68 bytes)
layout(push_constant) uniform PushConstants
{
    // xyz world position of the camera of the first viewport, the center of
    // the inner cube of the grid
    vec4 m_origin;
    // x0, y0, x1, y1 (exclusive) in pixels of the g-buffer: the viewport
    ivec4 m_rect;
    // Frame of the pool
    uint m_frame;
    // 1 if the pool is not initialized (the first frame)
    uint m_first_frame;
    // Maximum number of rays of the frame
    uint m_max_rays;
    // Size of a cell of the inner cube of the grid, and half its size
    float m_delta;
    float m_half_d;
    // Vertical fov of the camera, in radians
    float m_fov;
    // Area of a surfel in pixels
    float m_surfel_size;
    // Size of the viewport, for the radius of the surfels
    uint m_full_width;
    uint m_full_height;
} pc;

// Same as GEVulkanSurfelGI
const uint SURFEL_MAX_COUNT = 131072u;
// Rays a surfel traces in a frame at most. The estimator merges the rays in
// packets (see surfel_gi_integrate.comp), so the few rays of a converged
// surfel would follow a change of its lighting slower with 8 rays than with
// 16: the reference gives every surfel which is not converged 64 rays. 16
// rays for the whole pool costs too much of the frame here (the tracing is the
// mass of the pass), 8 keeps the packets which follow a change of the lighting
const uint SURFEL_MAX_RAYS = 8u;
// Rays of a surfel which is warmed up (see SURFEL_WARMUP_FRAMES): its first
// frames are the ones which converge it, so it traces twice the rays of a
// converged surfel (the reference gives every surfel which is not converged
// 64 rays, which the frame of the game doesn't afford; the pool is only young
// after the camera moved into a new area, so the extra rays of the warm up are
// not the whole pool)
const uint SURFEL_WARMUP_RAYS = 16u;
// Rays of all the surfels of a frame at most, the surfels which don't get any
// keep their radiance (they're traced in the next frame). This is the only
// limit of the cost of a frame (the same as kMaxRayCount of the reference),
// and the size of the buffers of the rays: it must be the RAY_BUDGET of
// GEVulkanSurfelGI (MAX_SURFELS * WARMUP_RAYS), a larger budget here writes
// past the u_ray_ids, u_ray_results and u_ray_data buffers of the pool
const uint SURFEL_RAY_BUDGET = SURFEL_MAX_COUNT * SURFEL_WARMUP_RAYS;
// Rays the surfels ask for in a frame, before the adaptive ray counts of the
// frame before scale them down when the pool is full: the surfels which are
// not converged get up to 8 rays (see surfel_gi_update.comp), so the budget of
// a frame of the pool is the target, and a frame which asked for more is
// scaled down smoothly instead of dropping the rays of a part of the pool
const uint SURFEL_RAY_TARGET = 3u * SURFEL_MAX_COUNT;
// Frames a surfel is warmed up: a surfel which the camera just created has
// only the seed of the generate pass (the field around it, or the environment)
// as its value, so its first frames merge the samples with a large temporal
// window (see surfel_gi_integrate.comp). The 1% window of a converged surfel
// takes ~50 frames of 8 rays (a second) to leave a wrong seed, which the
// camera sees as a lagging value in every area it moves into; the progressive
// average of the warm up reaches the light of the surface in a few frames and
// keeps averaging the noise of the few rays, and the fade in of the generate
// pass (10 frames) hides the first frames
const uint SURFEL_WARMUP_FRAMES = 12u;
// Life of a surfel which the tracing spawned (a sleeping surfel): it's outside
// of the view, so only the paths of the surfels near its cell use it (and keep
// its life), and it's recycled after this many frames without a use
const uint SURFEL_SLEEPING_LIFE = 60u;
// Sleeping surfels the tracing of a frame spawns at most: the cells of the
// surfaces a bit outside of the view are filled in a few frames, and the pool
// is not taken from the surfels which the camera sees
const uint SURFEL_SLEEPING_SPAWNS = 512u;
// Window of the merge of the radiance of the neighbours in the mean of a
// surfel (see surfel_gi_share.comp) falls with the noise of its mean: a surfel
// which is not converged merges the field around it with a large window, so
// its mean takes the light of its area in a few frames instead of averaging
// the noise of its few rays for a second, and a converged surfel only smooths
// its own mean with a small one. The merge is after the estimator and it only
// changes its mean, so its variance, its catch up and its firefly clamp are
// the ones of the rays of the surfel: the sharing can't make the estimator
// trust a sample which is not from its rays
const float SURFEL_SHARING_WINDOW_CONVERGED = 0.02;
const float SURFEL_SHARING_WINDOW_NOISY = 0.15;
// Relative noise of the mean of a surfel (the standard error of its rays over
// its level, see surfelMeanNoise) which is the one of a converged surfel and
// the one of a surfel which is not converged: the noise of the sharing window
// and of the number of rays of the frame (see surfel_gi_update.comp)
const float SURFEL_NOISE_LOW = 0.03;
const float SURFEL_NOISE_HIGH = 0.2;
// Entries of the list of a cell which the gather of a pixel of
// surfel_gi_generate.comp walks at most. The list of a cell near the camera
// holds hundreds of surfels (the surfels are as large as a few pixels, the
// cell is 1.5 units), and the list is not ordered: a sample of it (or its
// first entries alone) leaves the pixels which that sample doesn't reach with
// no field, which the lighting pass reads as no ambient diffuse light (black
// patches near the camera). The walk covers the whole list up to this count,
// which is longer than the list of a cell in practice; the image of the pass
// is a quarter of the g-buffer, so the walk is still the cost of the image and
// not of the viewport (see surfel_gi_generate.comp)
const uint SURFEL_CELL_LIST_MAX = 256u;
// (surfel, cell) pairs of a frame at most (a surfel is in 27 cells at most)
const uint SURFEL_CELL_SURFEL_COUNT = SURFEL_MAX_COUNT * 32u;
// Entries of the list of a cell which surfelGatherIrradiance walks at most,
// starting from a random one and wrapping around (the list of a cell of a
// surface is long: the surfels which cover the point are a few of them, and
// the walk finds them where a random sample of the first entries is sparse),
// and the surfels which contribute to the field of a hit at most: the walk
// stops then, so a covered hit costs a bounded number of the values of the
// field and the estimator of the surfel averages them
const uint SURFEL_GATHER_WALK = 16u;
const uint SURFEL_GATHER_VALID = 4u;
// Entries of the list of the cell of a surfel which its irradiance sharing
// walks at most (from a random one, wrapping around), and the surfels which
// contribute to it at most: the neighbours of a surfel are the surfels of its
// own cell which reach it (a few of the hundreds of the list), so the sharing
// is a sampled merge, and the window of the merge (see surfel_gi_share.comp)
// makes the sample enough
const uint SURFEL_SHARE_WALK = 8u;
const uint SURFEL_SHARE_VALID = 4u;
// Distance of the hit of a ray of a surfel in the depth moments, in its
// radiuses (a miss is the range, so a direction of the sky which a texel never
// sees is not a surface near the surfel), and the floor of their variance in
// the same unit: a surface 5% of the radius away from the mean distance of a
// texel is as visible as one at the mean (the Chebyshev bound of
// surfelVisibility, which is not the one of a true distribution of the
// distances)
const float SURFEL_DEPTH_RANGE = 4.0;
const float SURFEL_DEPTH_VARIANCE = 0.05;
// Part of the directions of a ray which is sampled with the cosine weighted
// hemisphere when the irradiance map of the surfel guides them (the mixture of
// the guiding and the cosine, see surfelSampleDirection): 1/4 of the rays find
// the light the map doesn't have yet, and the weight of a sample of a texel
// which the map values the most is 1 / (1/4 + 3/4 * 27 / 36) = 2/3 of the one
// of a cosine weighted ray
const float SURFEL_GUIDE_MIX = 0.25;
// Relative noise of the samples of the estimator which the deviation of the
// firefly clamp has at least, and the deviation itself (see surfelMSME): a
// dark surfel has a plain deviation of a few 1e-5, which makes any sample of
// another surface a firefly and stops the estimator from following its light
const float SURFEL_MSME_RELATIVE_FLOOR = 0.03;
const float SURFEL_MSME_ABSOLUTE_FLOOR = 1.0e-4;
// Effective samples a mean averages at most: the weight of a packet in the
// mean is 1 / (n + 1) at least, so the mean follows a change of the lighting
// of the surfel in ~n / packets frames even when the estimator doesn't catch
// up, and the noise of a converged mean is the one of n rays
const float SURFEL_MSME_N_MAX = 64.0;
// Segments of a path of the trace of the rays: the light of the hit of the
// first segment which the field doesn't cover is the one of its own rays (the
// field has no value there), which is the indirect light two bounces farther
// (a corner which no surfel of the camera's rays reached), and a path which
// ends at a hit the field covers is as exact as the one-segment path. The
// paths are traced by the rays of the surfels, so their cost is the cost of
// the tracing
const uint SURFEL_PATH_SEGMENTS = 3u;
// Frames a surfel lives when it's not seen (it's seen when its radiance is
// used by a pixel)
const uint SURFEL_MAX_LIFE = 1200u;
// A surfel radius is this part of the radius of its cell at least
const float SURFEL_MIN_SIZE_RATIO = 0.15;
// Cells per axis of the inner cube of the grid, layers of each frustum, ratio
// of the size of the layers, half size of the inner cube (in cells: (2 *
// SURFEL_CELL_N + 6 * SURFEL_CELL_M) * SURFEL_CELL_N * SURFEL_CELL_N). The 32
// layers reach 96 + 3 * (1.3^32 - 1) / 0.3 ~= 44336 units from the camera, so
// the grid has every surface of the scene whatever its distance
const int SURFEL_CELL_N = 64;
const int SURFEL_CELL_M = 32;
const int SURFEL_CELL_HALF_N = SURFEL_CELL_N / 2;
const float SURFEL_CELL_P = 1.3;
const uint SURFEL_CELL_COUNT = 1048576u;

const uint SURFEL_NONE = 0xFFFFFFFFu;
const uint SURFEL_STATUS_SLEEPING = 1u;
// A pixel used the radiance of the surfel in this frame
const uint SURFEL_STATUS_SEEN = 2u;
// The surfel was used as the last hit of a ray in this frame
const uint SURFEL_STATUS_REFED = 4u;
// The surfel is in the pool (the free ones are in u_dead)
const uint SURFEL_STATUS_ALIVE = 8u;
// The surface turned under the surfel in this frame (a wheel): the mean of the
// surfel is the light of the old surface, so the integration merges the
// samples of the frame with a large short window blend (see
// surfel_gi_update.comp and surfel_gi_integrate.comp). The bit lasts one frame
const uint SURFEL_STATUS_TURNED = 16u;

// The same luminance as utils/rt_reflection.glsl (the weights of the samples of
// surfelMSME are the ones of Ray Tracing Gems)
const vec3 SURFEL_LUMINANCE = vec3(0.299, 0.587, 0.114);

// The relative noise of the mean of a surfel: the standard error of the
// luminance of its samples (the deviation over the square root of the
// effective count) over its level, what a pixel which its radiance lights sees
// as the noise of the GI (see SURFEL_NOISE_LOW)
float surfelMeanNoise(MSMEData data)
{
    const float level = dot(SURFEL_LUMINANCE, data.m_short_mean);
    const float deviation =
        dot(SURFEL_LUMINANCE, sqrt(max(data.m_variance, vec3(0.0))));
    return deviation / (max(level, 1.0e-3) * sqrt(max(data.m_count, 1.0)));
}

// The clamp of a sample of the integration (an irradiance over PI, the unit of
// surfel_gi_integrate.comp): the reference clamps a sample to 4 times the
// integral of the skybox (skydome.getIntegral() * 4), and the integral of a sky
// of the radiance L is 4 * PI * L over the sphere, so the threshold is
// 16 * PI * L in the unit of its samples (an irradiance). Our sample is this
// irradiance over PI (it's what the lighting pass expects, see
// surfel_gi_integrate.comp), so the threshold of our unit is 16 * L: a sample
// of a ray which hits the sky is L (cos / (PI * pdf) is 1), so a hit which is
// 16 times brighter than the sky is clamped. The sky of the engine is the flat
// skytop color, so the clamp is scene relative. A glossy hit of the sun is a
// sample several times the light of the sky and it would bias the mean
float surfelFireflyClamp(vec3 sky_color)
{
    return 16.0 * dot(sky_color, SURFEL_LUMINANCE);
}

// The map of the surfel is a 6x6 square of uv in [-1, 1]^2 which
// surfelSquareToDir maps to the hemisphere of its normal: the concentric map
// of Shirley and Chiu (surfelSquareToDisk) is area preserving and the disk is
// the cosine weighted hemisphere (z = sqrt(1 - r^2)), so a uv which is uniform
// in the square is a cosine weighted direction, and a texel is the same solid
// angle whichever its uv. The density of a cosine weighted sample in a texel is
// then 36 / PI * cosine (see surfelSampleDirection)

// Pixels of an image at half the size of the rect which cover it (x1 and y1
// are exclusive)
ivec4 surfelHalfRect(ivec4 rect)
{
    return ivec4(rect.xy >> 1, (rect.zw + 1) >> 1);
}

// The cell (x, y, z) of a world position in the region w (0 is the inner cube,
// 1 and 2 the X frustums, 3 and 4 the Y ones and 5 and 6 the Z ones). The cell
// can be outside of the grid, see surfelCellValid
ivec4 surfelCell(vec3 position)
{
    const vec3 relative = position - pc.m_origin.xyz;
    const float half_d = pc.m_half_d;
    if (abs(relative.x) <= half_d && abs(relative.y) <= half_d &&
        abs(relative.z) <= half_d)
    {
        // The cube is uniform
        const ivec3 index = clamp(ivec3((relative + half_d) / pc.m_delta),
            ivec3(0), ivec3(SURFEL_CELL_N - 1));
        return ivec4(index, 0);
    }

    // The main axis of the frustum, and the indices of the other axes
    int region;
    float main_position;
    float other_one;
    float other_two;
    const float abs_x = abs(relative.x);
    const float abs_y = abs(relative.y);
    const float abs_z = abs(relative.z);
    if (abs_x >= abs_y && abs_x >= abs_z)
    {
        region = relative.x > 0.0 ? 1 : 2;
        main_position = relative.x;
        other_one = relative.y;
        other_two = relative.z;
    }
    else if (abs_y >= abs_x && abs_y >= abs_z)
    {
        region = relative.y > 0.0 ? 3 : 4;
        main_position = relative.y;
        other_one = relative.x;
        other_two = relative.z;
    }
    else
    {
        region = relative.z > 0.0 ? 5 : 6;
        main_position = relative.z;
        other_one = relative.x;
        other_two = relative.y;
    }

    // Distance outside of the cube of the layer, the size of the layers grows
    // with SURFEL_CELL_P (the same as surfelMaxRadius)
    const float outside = abs(main_position) - half_d;
    const int layer = int(log(1.0 + float(SURFEL_CELL_N) * outside *
        (SURFEL_CELL_P - 1.0) / (2.0 * half_d)) / log(SURFEL_CELL_P));
    const float n = float(SURFEL_CELL_N);
    int u = int(ceil(other_one * n * 0.5 / main_position)) +
        SURFEL_CELL_HALF_N;
    int v = int(ceil(other_two * n * 0.5 / main_position)) +
        SURFEL_CELL_HALF_N;
    if (u > 0)
        u--;
    if (v > 0)
        v--;
    return ivec4(layer, u, v, region);
}

bool surfelCellValid(ivec4 cell)
{
    if (cell.w < 0 || cell.w > 6)
        return false;
    if (cell.x < 0 || cell.y < 0 || cell.z < 0)
        return false;
    if (cell.w == 0)
    {
        return cell.x < SURFEL_CELL_N && cell.y < SURFEL_CELL_N &&
            cell.z < SURFEL_CELL_N;
    }
    return cell.x < SURFEL_CELL_M && cell.y < SURFEL_CELL_N &&
        cell.z < SURFEL_CELL_N;
}

// Index of a cell in the buffers (the inner cube first, then the 6 frustums)
uint surfelCellIndex(ivec4 cell)
{
    if (cell.w == 0)
    {
        return uint(cell.x + SURFEL_CELL_N * (cell.y +
            SURFEL_CELL_N * cell.z));
    }
    const uint n = uint(SURFEL_CELL_N);
    const uint start = n * n * n;
    const uint layer = uint(cell.w - 1);
    return start + layer * uint(SURFEL_CELL_M) * n * n +
        uint(cell.x) * n * n + uint(cell.y * SURFEL_CELL_N + cell.z);
}

// Offsets of the 27 cells around a cell
const ivec3 SURFEL_NEIGHBORS[27] = ivec3[27](
    ivec3(-1, -1, -1), ivec3(-1, 0, -1), ivec3(-1, 1, -1),
    ivec3(0, -1, -1), ivec3(0, 0, -1), ivec3(0, 1, -1),
    ivec3(1, -1, -1), ivec3(1, 0, -1), ivec3(1, 1, -1),
    ivec3(-1, -1, 0), ivec3(-1, 0, 0), ivec3(-1, 1, 0),
    ivec3(0, -1, 0), ivec3(0, 0, 0), ivec3(0, 1, 0),
    ivec3(1, -1, 0), ivec3(1, 0, 0), ivec3(1, 1, 0),
    ivec3(-1, -1, 1), ivec3(-1, 0, 1), ivec3(-1, 1, 1),
    ivec3(0, -1, 1), ivec3(0, 0, 1), ivec3(0, 1, 1),
    ivec3(1, -1, 1), ivec3(1, 0, 1), ivec3(1, 1, 1));

// Whether the sphere of the surfel intersects the cell (the AABB of the cell
// in the world)
bool surfelIntersectsCell(Surfel surfel, ivec4 cell)
{
    if (!surfelCellValid(cell))
        return false;
    const float half_d = pc.m_half_d;
    const int half_n = SURFEL_CELL_HALF_N;
    vec3 min_position;
    vec3 max_position;
    if (cell.w == 0)
    {
        min_position = -half_d + vec3(cell.xyz) * pc.m_delta;
        max_position = min_position + pc.m_delta;
    }
    else
    {
        int u = cell.y;
        int v = cell.z;
        if (u < half_n)
            u -= half_n;
        else
            u = u - half_n + 1;
        if (v < half_n)
            v -= half_n;
        else
            v = v - half_n + 1;
        // Size of the layer from the cube, the near one first
        const float near_size = pc.m_delta * (1.0 - pow(SURFEL_CELL_P,
            float(cell.x))) / (1.0 - SURFEL_CELL_P);
        const float far_size = pc.m_delta * (1.0 - pow(SURFEL_CELL_P,
            float(cell.x + 1))) / (1.0 - SURFEL_CELL_P);
        float main_min;
        float main_max;
        if ((cell.w & 1) == 0)
        {
            main_min = -(half_d + far_size);
            main_max = -(half_d + near_size);
        }
        else
        {
            main_min = half_d + near_size;
            main_max = half_d + far_size;
        }
        const float n = float(SURFEL_CELL_N);
        // The frustum of a layer is a truncated pyramid, the cell is the quad
        // between the u and v indices at both ends of it. surfelCell divides
        // the other axis by the distance from the camera (not from the face of
        // the cube), so the lateral extent at main_min/main_max is the ratio
        // times that distance. Adding half_d here (as the reference does) moves
        // the near end of the quad outward and drops surfels inside the inner
        // part of their own cell, which leaves their cells empty
        float a_one;
        float b_one;
        float c_one;
        float d_one;
        if (u > 0)
        {
            a_one = 2.0 * float(u - 1) / n * main_min;
            b_one = 2.0 * float(u) / n * main_max;
            c_one = 2.0 * float(u - 1) / n * main_max;
            d_one = 2.0 * float(u) / n * main_min;
        }
        else
        {
            a_one = 2.0 * float(u + 1) / n * main_min;
            b_one = 2.0 * float(u) / n * main_max;
            c_one = 2.0 * float(u + 1) / n * main_max;
            d_one = 2.0 * float(u) / n * main_min;
        }
        const float min_one = min(min(a_one, b_one), min(c_one, d_one));
        const float max_one = max(max(a_one, b_one), max(c_one, d_one));
        float a_two;
        float b_two;
        float c_two;
        float d_two;
        if (v > 0)
        {
            a_two = 2.0 * float(v - 1) / n * main_min;
            b_two = 2.0 * float(v) / n * main_max;
            c_two = 2.0 * float(v - 1) / n * main_max;
            d_two = 2.0 * float(v) / n * main_min;
        }
        else
        {
            a_two = 2.0 * float(v + 1) / n * main_min;
            b_two = 2.0 * float(v) / n * main_max;
            c_two = 2.0 * float(v + 1) / n * main_max;
            d_two = 2.0 * float(v) / n * main_min;
        }
        const float min_two = min(min(a_two, b_two), min(c_two, d_two));
        const float max_two = max(max(a_two, b_two), max(c_two, d_two));
        if (cell.w == 1 || cell.w == 2)
        {
            min_position = vec3(main_min, min_one, min_two);
            max_position = vec3(main_max, max_one, max_two);
        }
        else if (cell.w == 3 || cell.w == 4)
        {
            min_position = vec3(min_one, main_min, min_two);
            max_position = vec3(max_one, main_max, max_two);
        }
        else
        {
            min_position = vec3(min_one, min_two, main_min);
            max_position = vec3(max_one, max_two, main_max);
        }
    }
    min_position += pc.m_origin.xyz;
    max_position += pc.m_origin.xyz;
    const vec3 closest = clamp(surfel.m_position, min_position, max_position);
    const vec3 bias = closest - surfel.m_position;
    return dot(bias, bias) <= surfel.m_radius * surfel.m_radius;
}

// Radius of a surfel which covers the same pixels as a surfel_size area of the
// viewport at this distance. The reference (calcRadiusApprox of
// shaderUtils_surfel_cell.glsl) passes SceneCamera::fov, which is the field of
// view in degrees (CameraManip.getFov() of the reference returns degrees), so
// the radius is much larger than the area of surfelSize pixels: it's clamped by
// surfelMaxRadius, i.e. it's about the size of the cell of the grid at its
// distance, and the pool covers the screen (with the radius of surfelSize
// pixels it would need thousands of times more surfels and every texel would be
// unfilled)
float surfelCoverRadius(float distance)
{
    const float resolution = max(float(pc.m_full_width),
        float(pc.m_full_height));
    const float fov = degrees(pc.m_fov);
    const float angle = sqrt(pc.m_surfel_size / 3.14159265) * fov * 2.0 /
        max(resolution, 1.0);
    return distance * tan(angle);
}

// Radius of the cell of a position at this distance from the camera (the size
// of the layer of the frustum it's in): a surfel is at most this large
float surfelMaxRadius(float distance)
{
    const float cell_radius = pc.m_half_d / float(SURFEL_CELL_N);
    if (distance <= pc.m_half_d)
        return cell_radius;
    const float outside = distance - pc.m_half_d;
    const int layer = int(log(1.0 + float(SURFEL_CELL_N) * outside *
        (SURFEL_CELL_P - 1.0) / (2.0 * pc.m_half_d)) /
        log(SURFEL_CELL_P));
    return cell_radius * pow(SURFEL_CELL_P, float(layer));
}

// The hash of Wellons ("Prospecting for Hash Functions")
uint surfelHash(uint x)
{
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

float surfelRand(uint index, uint frame)
{
    return float(surfelHash(index * 747796405u + frame * 2891336453u)) *
        (1.0 / 4294967296.0);
}

vec2 surfelRand2(uint index, uint frame)
{
    const uint h = surfelHash(index ^ (frame * 2891336453u));
    return vec2(float(h & 0xFFFFu), float(h >> 16u)) * (1.0 / 65536.0);
}

// A point of the R3 low discrepancy sequence (Roberts, "The unreasonable
// effectiveness of quasirandom sequences", 2018): the point of index i is
// {i * a1}, {i * a2}, {i * a3} in the three dimensions, and the sequence of the
// integers continues it (the multiplier of each axis is the high 32 bits of the
// powers of the plastic constant). The points of the sequence are much more
// uniform than random ones for a few samples (a surfel of 2 rays per frame has
// 24 of them after the warm up), so the directions of the rays of a surfel fill
// its hemisphere evenly instead of clustering. start decorrelates the surfels
// (the hash of the surfel index), and the index of the surfel continues over
// the frames (m_ray_index), so its directions stay a single sequence
vec3 surfelR3(uint start, uint index)
{
    const uvec3 base = uvec3(surfelHash(start), surfelHash(start + 1973u),
        surfelHash(start + 7919u));
    const uvec3 sequence = base + index * uvec3(0xD1B54A32u, 0xABC98388u,
        0x8CB92BA7u);
    return vec3(sequence >> 8u) * (1.0 / 16777216.0);
}

// Orthonormal basis of a normal (Duff et al. 2017, "Building an Orthonormal
// Basis, Revisited")
void surfelTangentFrame(vec3 normal, out vec3 tangent, out vec3 bitangent)
{
    const float s = normal.z >= 0.0 ? 1.0 : -1.0;
    const float a = -1.0 / (s + normal.z);
    const float b = normal.x * normal.y * a;
    tangent = vec3(1.0 + s * normal.x * normal.x * a, s * b, -s * normal.x);
    bitangent = vec3(b, s + normal.y * normal.y * a, -normal.y);
}

// Cosine weighted direction around a normal, u1 and u2 in [0, 1). cosine is
// the cosine of the direction and the normal, the pdf of the direction is
// cosine / PI
vec3 surfelCosineDirection(vec3 normal, float u1, float u2, out float cosine)
{
    vec3 tangent;
    vec3 bitangent;
    surfelTangentFrame(normal, tangent, bitangent);
    const float r = sqrt(u1);
    const float phi = 2.0 * 3.14159265 * u2;
    cosine = sqrt(max(0.0, 1.0 - u1));
    return normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) +
        normal * cosine);
}

// The concentric mapping of Shirley and Chiu ("Low distortion map between disk
// and square", 1997): the square [-1, 1]^2 to the unit disk. It's area
// preserving, so the area of a texel of the square is the same everywhere in
// the disk. radius is the signed radius of the square side which the point is
// in and phi its angle from the side
vec2 surfelSquareToDisk(vec2 uv)
{
    if (uv.x == 0.0 && uv.y == 0.0)
        return vec2(0.0);
    float radius;
    float phi;
    if (abs(uv.x) >= abs(uv.y))
    {
        radius = uv.x;
        phi = 0.7853981633974483 * (uv.y / uv.x);
    }
    else
    {
        radius = uv.y;
        phi = 1.5707963267948966 -
            0.7853981633974483 * (uv.x / uv.y);
    }
    return radius * vec2(cos(phi), sin(phi));
}

vec2 surfelDiskToSquare(vec2 disk)
{
    const float radius = length(disk);
    if (radius == 0.0)
        return vec2(0.0);
    const float fp = 1.2732395447351627; // 4 / PI
    float phi = atan(disk.y, disk.x);
    if (abs(disk.x) > abs(disk.y))
    {
        // The right (u >= 0) or left (u < 0) side of the square
        const float u = disk.x >= 0.0 ? radius : -radius;
        if (disk.x < 0.0)
            phi = phi > 0.0 ? phi - 3.1415926535897932 :
                phi + 3.1415926535897932;
        return vec2(u, fp * phi * u);
    }
    // The upper (v >= 0) or lower (v < 0) side of the square
    const float v = disk.y >= 0.0 ? radius : -radius;
    if (disk.y < 0.0)
        phi += 3.1415926535897932;
    return vec2(fp * (1.5707963267948966 - phi) * v, v);
}

// The direction of a uv of the 6x6 map (z is the normal). The disk of the
// concentric map is the cosine weighted hemisphere, so a uv which is uniform
// in the square is a cosine weighted direction
vec3 surfelSquareToDir(vec2 uv)
{
    const vec2 disk = surfelSquareToDisk(uv);
    return vec3(disk, sqrt(max(0.0, 1.0 - dot(disk, disk))));
}

// The uv of the 6x6 map of a direction (z >= 0, its plane is the disk)
vec2 surfelDirToSquare(vec3 direction)
{
    return surfelDiskToSquare(direction.xy);
}

// The cosine of the direction of a uv of the 6x6 map (its z)
float surfelSquareCosine(vec2 uv)
{
    const vec2 disk = surfelSquareToDisk(uv);
    return sqrt(max(0.0, 1.0 - dot(disk, disk)));
}

// The texel of the 6x6 map of a uv in [-1, 1]^2 (the texel 3, 3 is the one of
// the normal). A uv on the border belongs to the last texel, not to a 7th one
ivec2 surfelSquareTexel(vec2 uv)
{
    return clamp(ivec2(floor((uv + 1.0) * 3.0)), ivec2(0), ivec2(5));
}

// Top left pixel of the 6x6 tile of the irradiance map of the surfel
ivec2 surfelIrradianceBase(uint index)
{
    const ivec2 resolution = textureSize(u_surfel_irradiance, 0);
    const int columns = resolution.x / 6;
    return ivec2(int(index) % columns, int(index) / columns) * 6;
}

// The 6 sums of the rows of the 6x6 irradiance map of a surfel (see SurfelLight)
void surfelRowSums(SurfelLight light, out float rows[6])
{
    const vec2 row_zero = unpackHalf2x16(light.m_row_sums_01);
    const vec2 row_one = unpackHalf2x16(light.m_row_sums_23);
    const vec2 row_two = unpackHalf2x16(light.m_row_sums_45);
    rows[0] = row_zero.x;
    rows[1] = row_zero.y;
    rows[2] = row_one.x;
    rows[3] = row_one.y;
    rows[4] = row_two.x;
    rows[5] = row_two.y;
}

// The direction of a ray of a surfel. The directions of the texels of its 6x6
// irradiance map are sampled in proportion to their radiance (the ray guiding
// of SurfelPlus: the rays of a surfel go to its bright directions, so the noise
// of its mean falls for the same number of rays), mixed with the cosine
// weighted hemisphere (SURFEL_GUIDE_MIX of the directions): a map which is not
// full or whose sum is not positive guides nothing (a new surfel, or a recycled
// slot whose map was not integrated yet), and the mix keeps a part of the
// directions outside of the map's values even when it guides, so a direction
// which the map undervalues (a light which is just starting to shine on the
// surface) is still sampled. The walk is two levels (the sums of the rows of
// the map, then the 6 texels of the chosen row), so a ray reads 6 values
// instead of the 36 of a linear walk.
// weight is cosine / (PI * pdf), the factor which turns the radiance of the
// ray into its sample of the irradiance (see surfel_gi_trace.comp): the pdf is
// the density of the mixture, so the weight is at most
// 1 / SURFEL_GUIDE_MIX = 4 whatever the map guides the ray to, and it's the
// cosine weighted one (cosine / PI) when the map guides nothing.
// oct_uv is the uv of the texel of the map which the direction is in, in the
// space of the surfel (z is the normal, what surfelSquareTexel and
// surfelSquareCosine expect), so the integration stores the radiance of the ray
// in the texel of its direction
vec3 surfelSampleDirection(uint index, vec3 normal, uint irradiance,
                           float u1, float u2, float u3, out vec2 oct_uv,
                           out float weight)
{
    vec3 tangent;
    vec3 bitangent;
    surfelTangentFrame(normal, tangent, bitangent);
    const bool guided = (irradiance & 1u) != 0u;
    vec3 local;
    if (!guided)
    {
        // Cosine weighted hemisphere
        const float r = sqrt(u1);
        const float phi = 2.0 * 3.14159265 * u2;
        local = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        weight = 1.0;
        oct_uv = surfelDirToSquare(local);
    }
    else if (u3 < SURFEL_GUIDE_MIX)
    {
        // The cosine weighted direction, weighted by the mixture at its texel of
        // the map: the density of the mix is (1 - m) * cosine / PI +
        // m * p_texel * 36 * cosine / PI, so the ray is exact for both of its
        // branches whatever the map has at its direction
        const float r = sqrt(u1);
        const float phi = 2.0 * 3.14159265 * u2;
        local = vec3(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        oct_uv = surfelDirToSquare(local);
        const ivec2 texel = surfelSquareTexel(oct_uv);
        const SurfelLight light = u_surfel_lights[index];
        float rows[6];
        surfelRowSums(light, rows);
        const float rows_sum = rows[0] + rows[1] + rows[2] + rows[3] +
            rows[4] + rows[5];
        float probability = 1.0 / 36.0;
        if (rows_sum > 0.0)
        {
            // The probability of the texel in the guided branch (the row of the
            // walk, then its texel inside the row), so both branches of the
            // mixture use the same density
            const float value = texelFetch(u_surfel_irradiance,
                surfelIrradianceBase(index) + texel, 0).r;
            if (value > 0.0)
                probability = value / rows_sum;
        }
        weight = 1.0 / (SURFEL_GUIDE_MIX + (1.0 - SURFEL_GUIDE_MIX) *
            36.0 * probability);
    }
    else
    {
        // The first level: the row of the map, by the sums of its 6 texels (the
        // values of the rows of SurfelLight are half floats, and the walk
        // inside the row uses the exact texels of the map, so the row of the
        // threshold can be one whose texels are all 0)
        const SurfelLight light = u_surfel_lights[index];
        float rows[6];
        surfelRowSums(light, rows);
        const float rows_sum = rows[0] + rows[1] + rows[2] + rows[3] +
            rows[4] + rows[5];
        const float threshold = (u3 - SURFEL_GUIDE_MIX) /
            (1.0 - SURFEL_GUIDE_MIX) * max(rows_sum, 0.0);
        uint row = 5u;
        uint last_row = 0u;
        float cumulative = 0.0;
        float row_below = 0.0;
        bool found = false;
        for (uint y = 0u; y < 6u; y++)
        {
            if (rows[y] <= 0.0)
                continue;
            last_row = y;
            const float before = cumulative;
            cumulative += rows[y];
            if (cumulative > threshold)
            {
                row = y;
                row_below = before;
                found = true;
                break;
            }
        }
        // The threshold is above the sum of the rows: the last row which has a
        // value is the one
        if (!found)
        {
            row = last_row;
            row_below = cumulative - rows[last_row];
        }
        // The second level: the texel of the row, by its value. A row whose
        // texels are all 0 is sampled uniformly, and its probability of the mix
        // is the uniform one: the density of the direction is still the one of
        // its texel (less than 1/36), so the sample stays exact and its weight
        // is bounded
        const ivec2 base = surfelIrradianceBase(index) + ivec2(0, int(row));
        float texels[6];
        float row_sum = 0.0;
        for (uint x = 0u; x < 6u; x++)
        {
            texels[x] = texelFetch(u_surfel_irradiance, base + ivec2(x, 0),
                0).r;
            row_sum += texels[x];
        }
        uint column = min(uint(u1 * 6.0), 5u);
        float probability = 1.0 / 36.0;
        if (row_sum > 0.0)
        {
            float texel_cumulative = 0.0;
            for (uint x = 0u; x < 6u; x++)
            {
                if (texels[x] <= 0.0)
                    continue;
                column = x;
                texel_cumulative += texels[x];
                if (texel_cumulative > threshold - row_below)
                    break;
            }
            probability = max(texels[column], 0.0) / max(rows_sum, 1.0e-12);
        }
        // A point of the texel, uniform in it: the uv of the 6x6 square is
        // (column, row) + (u1, u2), so surfelSquareTexel finds the texel again
        oct_uv = (vec2(column, row) + vec2(u1, u2) - 3.0) / 3.0;
        local = surfelSquareToDir(oct_uv);
        weight = 1.0 / (SURFEL_GUIDE_MIX + (1.0 - SURFEL_GUIDE_MIX) *
            36.0 * probability);
    }
    return normalize(tangent * local.x + bitangent * local.y + normal * local.z);
}

// Multiscale mean estimator (Ray Tracing Gems chapter 25, "Multiscale Mean
// Estimator", the same as GIBS). The mean is a long running mean of the
// samples which doesn't flicker, the short mean is a short running mean which
// follows the lighting: when they are far apart (in standard deviations) the
// mean catches up quickly, so the surfel reacts to a change of its lighting as
// fast as it converges when it's stable. A sample which is a few standard
// deviations farther than the short mean is a firefly (a glossy hit or the
// sun) and it's clamped to it, so it doesn't bias the mean.
// The deviation has a relative floor (SURFEL_MSME_RELATIVE_FLOOR of the level
// of the surfel): the light of a shadowed surface is a few percent of the one
// of a lit one, and the plain deviation of its samples is a few 1e-5 (the
// engine works in hdr units of ~1), so without the floor any sample of another
// surface would be a firefly and the estimator would ignore the light of its
// own surface. mean_blend is the weight of the sample in the mean at least (0
// for a converged surfel, whose catch up is the one of the estimator): the warm
// up of a new surfel uses it to take the light of its first frames, see
// SURFEL_WARMUP_FRAMES. The mean is never more certain than n samples
// (SURFEL_MSME_N_MAX): the weight of a sample is 1/(n + 1) at least, so the
// mean follows a change of the lighting even when the catch up of the estimator
// doesn't (a surfel of a surface which the sun starts to light), and the noise
// of a converged mean stays the one of its rays instead of falling forever
vec3 surfelMSME(vec3 new_sample, inout MSMEData data, float short_window_blend,
                float mean_blend)
{
    const float level = dot(SURFEL_LUMINANCE, data.m_short_mean);
    const float floor_deviation = max(level * SURFEL_MSME_RELATIVE_FLOOR,
        SURFEL_MSME_ABSOLUTE_FLOOR);
    vec3 deviation = sqrt(max(data.m_variance, floor_deviation *
        floor_deviation));
    // Suppress the fireflies
    const vec3 sample_value = min(new_sample, data.m_short_mean + deviation *
        8.0);

    // The variance is estimated with a longer window than the short mean, to
    // avoid the bias of the variance getting smaller when the short mean does
    const vec3 delta = sample_value - data.m_short_mean;
    data.m_short_mean = mix(data.m_short_mean, sample_value,
        short_window_blend);
    const vec3 delta_two = sample_value - data.m_short_mean;
    data.m_variance = max(vec3(floor_deviation * floor_deviation), mix(
        data.m_variance, delta * delta_two, short_window_blend * 0.5));
    deviation = sqrt(data.m_variance);

    // Distance of the mean to the short mean in standard deviations, and how
    // inconsistent it was recently
    const float relative_diff = dot(SURFEL_LUMINANCE,
        abs(data.m_mean - data.m_short_mean) / deviation);
    data.m_inconsistency = mix(data.m_inconsistency, relative_diff, 0.08);

    // A converged surfel (its samples are small compared to its radiance) has a
    // small blend, so it doesn't flicker, but it's never frozen
    const float reduction = clamp(dot(SURFEL_LUMINANCE,
        0.5 * data.m_short_mean / deviation), 1.0 / 32.0, 1.0);
    float catch_up = clamp(smoothstep(0.0, 1.0,
        relative_diff * max(0.02, data.m_inconsistency - 0.2)), 1.0 / 256.0,
        1.0);
    catch_up *= data.m_vbbr;
    data.m_vbbr = mix(data.m_vbbr, reduction, 0.1);
    const float n = min(data.m_count, SURFEL_MSME_N_MAX);
    const float blend = clamp(max(max(catch_up, mean_blend), 1.0 / (n + 1.0)),
        0.0, 1.0);
    data.m_count = min(min(n + 1.0, 1.0 / blend), SURFEL_MSME_N_MAX);
    data.m_mean = mix(data.m_mean, sample_value, blend);
    return data.m_mean;
}

// Takes an id from the stack of the free surfels (SURFEL_NONE if the pool is
// full)
uint surfelAllocate()
{
    int free_count = u_counter.m_free;
    while (free_count > 0)
    {
        if (atomicCompSwap(u_counter.m_free, free_count, free_count - 1) ==
            free_count)
        {
            atomicAdd(u_counter.m_alive, 1);
            return u_dead[uint(free_count - 1)];
        }
        free_count = u_counter.m_free;
    }
    return SURFEL_NONE;
}

// Returns the id of the surfel to the stack of the free surfels
void surfelFree(uint index)
{
    const int free_count = atomicAdd(u_counter.m_free, 1);
    if (free_count < int(SURFEL_MAX_COUNT))
        u_dead[uint(free_count)] = index;
    atomicAdd(u_counter.m_alive, -1);
}

// The visibility of a surfel from a point of the world (Chebyshev's inequality
// on the depth moments of the texel of its direction, the same test as the
// probes of DDGI and W298 SurfelGI): the rays of the surfel which its map sent
// that way stopped at the mean distance m of the texel, with a variance s^2, so
// a point at the distance d from the surfel is at least (1 - s^2 / (s^2 + (d -
// m)^2)) visible. A point which is not farther than the mean is not hidden by
// the surface the texel sees (1), and one behind it is hidden more and more as
// it's farther (a wall at 2 units doesn't hide the light of the surfel from 3
// units as much as from 10). The deviation of the textures is clamped to a part
// of the range of the moments, so a texel which the rays didn't hit (a variance
// of 0) doesn't hide every point behind its first hit, and one which the rays
// hit at different distances doesn't hide everything (its variance is large).
// This is the visibility of the surfel alone, as the DDGI probe map: the
// shadows of the direct lights are traced.
float surfelVisibility(uint index, SurfelLight light, vec3 point)
{
    const float range = light.m_depth_range;
    if (range <= 0.0)
        return 1.0;
    const vec3 bias = point - light.m_position_radius.xyz;
    const float dist = length(bias);
    if (dist < max(light.m_position_radius.w * 0.05, 1.0e-3))
        return 1.0;

    // The direction in the space of the surfel, and the texel of the map of
    // that direction (a point behind the surface is the horizon of its normal)
    vec3 tangent;
    vec3 bitangent;
    surfelTangentFrame(light.m_normal_frame.xyz, tangent, bitangent);
    const vec3 local = vec3(dot(bias, tangent), dot(bias, bitangent),
        dot(bias, light.m_normal_frame.xyz));
    const vec3 planar = vec3(local.xy, 0.0);
    const float planar_length = length(planar);
    const vec3 direction = local.z >= 0.0 || planar_length < 1.0e-9 ?
        local : planar / planar_length;
    const ivec2 texel = surfelSquareTexel(surfelDirToSquare(
        normalize(direction)));
    const vec2 moments = texelFetch(u_surfel_depth,
        surfelIrradianceBase(index) + texel, 0).rg;
    const float mean = min(moments.x, range);
    const float variance = max(moments.y - mean * mean, 0.0);
    const float floor_deviation = range * SURFEL_DEPTH_VARIANCE;
    const float deviation = max(variance, floor_deviation * floor_deviation);
    const float point_distance = min(dist, range);
    if (point_distance <= mean)
        return 1.0;
    const float difference = point_distance - mean;
    return deviation / (deviation + difference * difference);
}

// The irradiance of a point of a surface from the surfels of its cell (the
// finalizePathWithSurfel of SurfelPlus, moved to every hit of the multi
// segment path of surfel_gi_trace.comp): the field of the surfels is the light
// which the continuation of the path would find again, so it replaces it.
// The list of the cell is walked from a random entry (the walk wraps around,
// SURFEL_GATHER_WALK entries), and every surfel which covers the point (its
// radius and the face of both of its normals) contributes its radiance
// weighted by the angle of the normals, the distance and its visibility from
// the point (the depth moments of the surfel, see surfelVisibility: a surfel
// which a wall hides from the point doesn't light it). The walk stops at
// SURFEL_GATHER_VALID surfels: the samples of the rays of a surfel and the
// estimator of its mean average the field, so a few of them are enough, and
// the pixels which the camera sees don't have to walk the whole list.
// sun_direction is the normalized direction of the sun, and sun is the
// visibility of the sun at the point, the weighted mean of the cached
// visibility of the surfels which face the sun (-1 if none of them was
// traced): a hit whose field covers it takes its shadow from the field instead
// of tracing a shadow ray (the direct light cache of the reference article).
// Returns false if no surfel of the cell covers the point
bool surfelGatherIrradiance(vec3 position, vec3 normal,
                            vec3 sun_direction, uint random_index,
                            out vec3 irradiance, out float sun)
{
    irradiance = vec3(0.0);
    sun = -1.0;
    const ivec4 cell = surfelCell(position);
    if (!surfelCellValid(cell))
        return false;
    const CellInfo cell_info = u_cells[surfelCellIndex(cell)];
    if (cell_info.m_offset >= SURFEL_CELL_SURFEL_COUNT)
        return false;
    const uint count = min(cell_info.m_count, SURFEL_CELL_LIST_MAX);
    if (count == 0u)
        return false;
    const uint start = uint(surfelRand(random_index, pc.m_frame) *
        float(count));
    vec4 sum = vec4(0.0);
    float sun_sum = 0.0;
    float sun_weight = 0.0;
    uint valid = 0u;
    for (uint i = 0u; i < min(count, SURFEL_GATHER_WALK); i++)
    {
        const uint other_index =
            u_cell_surfels[cell_info.m_offset + (start + i) % count];
        if (other_index >= SURFEL_MAX_COUNT)
            continue;
        // The position and the radius first: most of the surfels of the cell
        // don't cover the point, so the rest of the compact surfel is only read
        // by the ones which do (the list of a cell near the camera is long, the
        // memory of the field is the mass of the gather)
        const vec4 position_radius = u_surfel_lights[other_index]
            .m_position_radius;
        const vec3 bias = position_radius.xyz - position;
        const float dist_sq = dot(bias, bias);
        const float other_radius = position_radius.w;
        if (dist_sq >= other_radius * other_radius)
            continue;
        const float dist = sqrt(dist_sq);
        const SurfelLight other = u_surfel_lights[other_index];
        const vec3 other_normal = other.m_normal_frame.xyz;
        const float cosine_theta = dot(bias, normal) / max(dist, 1.0e-9);
        // The surfels on the backside of the surface are not shared
        if (cosine_theta < -0.2 ||
            dot(-bias, other_normal) / max(dist, 1.0e-9) < -0.2)
            continue;
        const float dot_n = dot(normal, other_normal);
        float contribution;
        if (dot_n > 0.0)
        {
            contribution = min(dot_n, 1.0) *
                clamp(1.0 - dist / other_radius, 0.0, 1.0);
            contribution = smoothstep(0.0, 1.0, contribution);
        }
        else
        {
            contribution = max(cosine_theta, 0.0) * pow(1.0 -
                dist / other_radius, 2.0);
        }
        contribution *= surfelVisibility(other_index, other, position);
        if (contribution > 0.0)
        {
            sum += vec4(other.m_radiance, 1.0) * contribution;
            // Its radiance is part of the GI even if the camera doesn't see it
            // (its life is reset, so the field of the surfaces the rays reach
            // doesn't slide away from the surfels of the rays)
            if ((u_surfels[other_index].m_status & SURFEL_STATUS_REFED) == 0u)
                atomicOr(u_surfels[other_index].m_status, SURFEL_STATUS_REFED);
            // The visibility of the sun is the one of the surfels which the sun
            // lights directly (a surfel which faces away from it is not a
            // shadow, the sun doesn't light the surface it's on)
            const float sun_cosine = dot(other_normal, sun_direction);
            if (other.m_sun >= 0.0 && sun_cosine > 0.0)
            {
                sun_sum += other.m_sun * contribution;
                sun_weight += contribution;
            }
            valid++;
        }
        if (valid >= SURFEL_GATHER_VALID)
            break;
    }
    if (sum.w <= 0.1)
        return false;
    irradiance = sum.xyz / sum.w;
    if (sun_weight > 0.0)
        sun = clamp(sun_sum / sun_weight, 0.0, 1.0);
    return true;
}
