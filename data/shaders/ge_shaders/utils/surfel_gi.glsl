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
    // Incremented every frame (random seeds)
    uint m_frame;
    uint m_camera_count;
    // The pool is cleared (surfel_gi_clear.comp)
    uint m_reset;
    // The env set (surfel_gi_trace.comp) has the skybox, not the dummy one
    uint m_has_skybox;
    // Milliseconds since the start (wraps), for the life of the surfels
    uint m_time;
    // Time of the frame in steps of 1/60 s (SURFEL_STEPS_PER_SECOND), so
    // the surfels fill in and follow the lighting in the same time at any
    // frame rate
    float m_steps;
    // The surfels of the pool in use: SURFEL_PER_CAMERA for every viewport
    // (the buffers have room for the most viewports), so the budget of a
    // camera doesn't depend on how many players share the screen
    uint m_surfel_count;
} pc;

// Same as Surfel of GEVulkanSurfelGI (std430, 48 bytes), a free surfel has a
// radius of 0
struct Surfel
{
    vec3 m_position;
    float m_radius;
    vec3 m_normal;
    // m_time of the push constants when a pixel last used it
    uint m_last_seen;
    vec3 m_irradiance;
    // Steps (m_steps of the push constants) of the samples in the mean of the
    // irradiance, up to SURFEL_MAX_SAMPLES
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

// Number of surfels in every cell of the grid of the frame, a surfel is in the
// cell of its center (it can be more than SURFEL_CELL_CAPACITY, the others are
// not in the cell then, and no surfel spawns in it)
layout(std430, set = SURFEL_SET, binding = 5) buffer SurfelCellCount
{
    uint u_cell_count[];
};

// SURFEL_CELL_CAPACITY surfels of every cell
layout(std430, set = SURFEL_SET, binding = 6) buffer SurfelCellSurfels
{
    uint u_cell_surfels[];
};

// Number of surfels spawned in every cell in this frame (cleared with the
// cells), see trySpawnSurfelInCell()
layout(std430, set = SURFEL_SET, binding = 7) buffer SurfelCellSpawns
{
    uint u_cell_spawns[];
};

// Surfels of the pool for every viewport, they are shared by the cameras (a
// surface seen by two of them has the same surfels), see pc.m_surfel_count
const uint SURFEL_PER_CAMERA = 32768u;
const uint SURFEL_CELL_COUNT = 65536u;
const uint SURFEL_CELL_CAPACITY = 64u;
// New surfels of a cell in a step (1/60 s), see trySpawnSurfelInCell()
const float SURFEL_CELL_SPAWNS_PER_STEP = 8.0;
// The constants of time are in steps of 1/60 s (pc.m_steps is the time of the
// frame in them), not in frames: a slower GPU has fewer frames in a second
const float SURFEL_STEPS_PER_SECOND = 60.0;
// Rays of a surfel in a frame and the steps of the samples of the mean of the
// irradiance: the irradiance follows a change of the lighting in about 64
// steps (a frame of a slower GPU is a sample which weighs more)
const uint SURFEL_RAYS = 4u;
const float SURFEL_MAX_SAMPLES = 64.0;
// Steps of samples a surfel needs to count fully, a new surfel has a noisy
// mean
const float SURFEL_CONFIDENT_SAMPLES = 8.0;
// Milliseconds a surfel which no pixel uses is kept, when the pool has enough
// free surfels (a quarter of the pool, SURFEL_LOW_POOL_PER_CAMERA per camera)
const uint SURFEL_MAX_LIFE = 5000u;
// With fewer free surfels than SURFEL_LOW_POOL_PER_CAMERA of every camera the
// life is shorter, down to SURFEL_MIN_LIFE when the pool is empty: the surfels
// which nothing used for the longest are recycled first, so a fast camera
// which keeps seeing new surfaces (a tunnel after a valley) doesn't wait for
// the old ones to expire
const uint SURFEL_MIN_LIFE = 500u;
const int SURFEL_LOW_POOL_PER_CAMERA = 8192;
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

// At most half a cell of its level, so the 2x2x2 cells nearest to a position
// (getSurfelNearestCells) have the centers of the surfels which touch it
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

// The first of the 2x2x2 cells of a level nearest to a position (it's at least
// half a cell from their sides)
ivec3 getSurfelNearestCells(vec3 position, float cell_size)
{
    return ivec3(floor(position / cell_size - 0.5));
}

// The new surfels of a frame are only in the grid in the next one, so every
// group of pixels which sees an empty cell would spawn in it in the same
// frame: a cell gets at most SURFEL_CELL_CAPACITY surfels with the ones
// spawned in this frame. And only SURFEL_CELL_SPAWNS_PER_STEP of them in a
// step: a cell comes into view as a thin strip at the side of the screen (the
// floor behind a camera which moves back), which is the least covered pixel
// of many groups, they would fill the whole cell on the strip and the rest of
// it would never get a surfel. The next frames spawn where the surfels of
// this one don't cover it. Returns false if it's full
bool trySpawnSurfelInCell(uint cell)
{
    if (u_cell_count[cell] >= SURFEL_CELL_CAPACITY)
        return false;
    uint spawned = atomicAdd(u_cell_spawns[cell], 1u);
    return spawned < uint(ceil(SURFEL_CELL_SPAWNS_PER_STEP * pc.m_steps)) &&
        u_cell_count[cell] + spawned < SURFEL_CELL_CAPACITY;
}

// A random number in [0, 1) of a seed
float getSurfelRandom(inout uint seed)
{
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word >> 8u) * (1.0 / 16777216.0);
}
