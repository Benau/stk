// The surfels of the diffuse global illumination (GEVulkanSurfelGI), shared by
// its compute shaders:
//
// surfel_gi_clear.comp:  clears the cells of the grid (and the whole pool when
//                        the static scene is built again)
// surfel_gi_update.comp: recycles the surfels which were not seen for a while,
//                        sizes the others by their distance to the cameras and
//                        puts them in the cells of the grid they touch
// surfel_gi_trace.comp:  traces a few cosine weighted rays from every surfel
//                        in the static scene and adds them to its irradiance
// surfel_gi_apply.comp:  the irradiance of every pixel from the surfels of its
//                        cell, and a new surfel where the surfels don't cover
//                        the static scene enough
//
// A surfel is a disk on a surface of the static scene (the static BLAS of
// GEVulkanRTScene, RT_MASK_STATIC), it never moves. The grid is a hash of the
// cells of nested levels around the cameras: the cells of a level are twice
// the size of the previous one, and the level of a position is the one of its
// distance to the closest camera, like the radius of a surfel, so a far surface
// has fewer bigger surfels. The irradiance is the mean of the cosine weighted
// radiance of the rays, the same as the irradiance of the diffuse environment
// map which the lighting pass replaces with it.
//
// The shader which includes this file must define SURFEL_SET (the set of the
// buffers and the images of the pass) before it, the constants here are the
// same as the ones of GEVulkanSurfelGI.

// Same as SurfelPushConstants of GEVulkanSurfelGI, all the passes have them
layout(push_constant) uniform SurfelPushConstants
{
    // The world positions of the cameras of all the viewports
    vec4 m_cameras[4];
    // x0, y0, x1, y1 (exclusive) in pixels of the viewport of
    // surfel_gi_apply.comp
    ivec4 m_rect;
    // Incremented every frame
    uint m_frame;
    uint m_camera_count;
    // The pool is cleared (surfel_gi_clear.comp)
    uint m_reset;
    // The env set (surfel_gi_trace.comp) has the skybox, not the dummy one
    uint m_has_skybox;
} pc;

// Same as Surfel of GEVulkanSurfelGI (std430, 48 bytes), a free surfel has a
// radius of 0
struct Surfel
{
    vec3 m_position;
    float m_radius;
    vec3 m_normal;
    // m_frame of the push constants when a pixel last used it
    uint m_last_seen;
    vec3 m_irradiance;
    // Samples in the mean of the irradiance, up to SURFEL_MAX_SAMPLES
    float m_samples;
};

layout(std430, set = SURFEL_SET, binding = 3) buffer SurfelBuffer
{
    Surfel u_surfels[];
};

// The stack of the free surfels, m_free_count is signed so a pop which finds it
// empty can be undone
layout(std430, set = SURFEL_SET, binding = 4) buffer SurfelPool
{
    int m_free_count;
    uint m_free[];
} u_pool;

// Number of surfels in every cell of the grid of the frame (it can be more than
// SURFEL_CELL_CAPACITY, the others are not in the cell then)
layout(std430, set = SURFEL_SET, binding = 5) buffer SurfelCellCount
{
    uint u_cell_count[];
};

// SURFEL_CELL_CAPACITY surfels of every cell
layout(std430, set = SURFEL_SET, binding = 6) buffer SurfelCellSurfels
{
    uint u_cell_surfels[];
};

const uint SURFEL_MAX = 32768u;
const uint SURFEL_CELL_COUNT = 65536u;
const uint SURFEL_CELL_CAPACITY = 32u;
// Rays of a surfel in a frame and the samples of the mean of the irradiance:
// the irradiance follows a change of the lighting in about 64 frames
const uint SURFEL_RAYS = 4u;
const float SURFEL_MAX_SAMPLES = 64.0;
// Frames a surfel which no pixel uses is kept
const uint SURFEL_MAX_LIFE = 600u;
// A surfel farther than it from all the cameras is recycled
const float SURFEL_MAX_DISTANCE = 1000.0;
// The cells of the first level are SURFEL_CELL_SIZE wide, up to
// SURFEL_LEVEL_DISTANCE from the cameras, every next level doubles both
const float SURFEL_CELL_SIZE = 2.0;
const float SURFEL_LEVEL_DISTANCE = 16.0;
const uint SURFEL_LEVELS = 8u;

// The cameras of all the viewports are in the push constants
float getSurfelCameraDistance(vec3 position)
{
    float d = 1.0e10;
    for (uint i = 0u; i < pc.m_camera_count; i++)
        d = min(d, distance(position, pc.m_cameras[i].xyz));
    return d;
}

uint getSurfelLevel(float camera_distance)
{
    float level = floor(log2(max(camera_distance, SURFEL_LEVEL_DISTANCE) /
        SURFEL_LEVEL_DISTANCE));
    return uint(min(level, float(SURFEL_LEVELS - 1u)));
}

float getSurfelCellSize(uint level)
{
    return SURFEL_CELL_SIZE * exp2(float(level));
}

// At most half a cell of its level, so a surfel touches at most 2 cells per
// axis of its own level
float getSurfelRadius(float camera_distance)
{
    float radius = max(camera_distance, SURFEL_LEVEL_DISTANCE) *
        (SURFEL_CELL_SIZE / (4.0 * SURFEL_LEVEL_DISTANCE));
    return min(radius,
        getSurfelCellSize(getSurfelLevel(camera_distance)) * 0.5);
}

// The cells of the levels are in one hash table, a collision only puts surfels
// of another cell in a list (they are too far to be used)
uint getSurfelCell(ivec3 cell, uint level)
{
    uvec3 v = uvec3(cell) * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16u;
    return (v.x ^ v.y ^ v.z ^ (level * 0x9E3779B9u)) &
        (SURFEL_CELL_COUNT - 1u);
}

uint getSurfelCell(vec3 position, uint level)
{
    return getSurfelCell(ivec3(floor(position / getSurfelCellSize(level))),
        level);
}

// A random number in [0, 1) of a seed
float getSurfelRandom(inout uint seed)
{
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word >> 8u) * (1.0 / 16777216.0);
}
