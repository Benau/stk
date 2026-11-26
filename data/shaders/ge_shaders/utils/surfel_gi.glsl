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
    // The env set (surfel_gi_trace.comp) has the skybox, not the dummy one
    uint m_has_skybox;
    // Milliseconds since the start (wraps), for the life of the surfels
    uint m_time;
    // Time of the frame in steps of 1/60 s (SURFEL_STEPS_PER_SECOND), so
    // the surfels fill in and follow the lighting in the same time at any
    // frame rate
    float m_steps;
    // The surfels of the pool in use: SURFELS_PER_CAMERA for every viewport
    // (the buffers have room for the most viewports), so the budget of a
    // camera doesn't depend on how many players share the screen
    uint m_surfel_count;
    // SURFEL_CACHE_* bits
    uint m_cache;
    // The list of u_trace_lists which surfel_gi_trace.comp traces with the
    // frame cache (SURFEL_TRACE_*)
    uint m_trace_list;
    uint m_geometry_count;
    float m_depth_step;
} pc;

const uint SURFEL_CACHE_FRAME = 1u;
const uint SURFEL_GI_RESET_CACHE = 16u;

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

// The same as u_surfels, 16 bytes at a time: the position and the radius of
// surfel i are at 3 * i, the normal and the last time seen (its bits) at
// 3 * i + 1, the irradiance and the samples at 3 * i + 2, so the loops which
// test many surfels can read the first of many together and the rest only for
// the ones which are close enough
layout(std430, set = SURFEL_SET, binding = 3) buffer SurfelBufferVec4
{
    vec4 u_surfels_vec4[];
};

// The stack of the free surfels, m_free_count is signed so a pop which finds it
// empty can be undone
layout(std430, set = SURFEL_SET, binding = 4) buffer SurfelPool
{
    int m_free_count;
    uint m_free[];
} u_pool;

layout(std430, set = SURFEL_SET, binding = 8) buffer SpawnQueue
{
    uint dispatch[3];
    uint count;
    uint ids[];
} new_surfels;

// Number of surfels in every cell, including its overflow list
layout(std430, set = SURFEL_SET, binding = 5) buffer SurfelCellCount
{
    uint u_cell_count[];
};

// SURFEL_CELL_CAPACITY surfels of every cell
layout(std430, set = SURFEL_SET, binding = 6) buffer SurfelCellSurfels
{
    uint u_cell_surfels[];
};

// The same, 4 surfels at a time (SURFEL_CELL_CAPACITY is a multiple of 4)
layout(std430, set = SURFEL_SET, binding = 6) buffer SurfelCellSurfels4
{
    uvec4 u_cell_surfels4[];
};

// Number of surfels spawned in every cell in this frame (cleared with the
// cells), see trySpawnSurfelInCell()
layout(std430, set = SURFEL_SET, binding = 7) buffer SurfelCellSpawns
{
    uint u_cell_spawns[];
};

// With the frame cache, the surfels which surfel_gi_trace.comp traces in the
// frame, put in 2 lists by surfel_gi_update.comp: the ones which didn't
// converge (they trace SURFEL_YOUNG_RAY_SCALE times the rays) and the
// converged ones whose turn it is. A group of threads takes as long as its
// slowest one, so mixed in one dispatch every group would trace the rays of a
// surfel which didn't converge, and the converged ones which don't trace in
// the frame would still take a thread. Every list has its indirect dispatch
// (x, y, z groups of 64) and its count, the ids of list i start at
// i * pc.m_surfel_count
layout(std430, set = SURFEL_SET, binding = 9) buffer SurfelTraceLists
{
    uint m_dispatch[2][3];
    uint m_count[2];
    uint m_selected[2];
    uint m_ids[];
} u_trace_lists;

const uint SURFEL_TRACE_YOUNG = 0u;
const uint SURFEL_TRACE_CONVERGED = 1u;

const uint SURFEL_CELL_COUNT = 65536u;
const uint SURFEL_CELL_CAPACITY = 64u;
const uint SURFEL_OVERFLOW_HEADS = SURFEL_CELL_COUNT * SURFEL_CELL_CAPACITY;
const uint SURFEL_OVERFLOW_LINKS = SURFEL_OVERFLOW_HEADS + SURFEL_CELL_COUNT;

uint nextCellSurfel(uint cell, uint index, inout uint link)
{
    if (index < SURFEL_CELL_CAPACITY)
        return u_cell_surfels[cell * SURFEL_CELL_CAPACITY + index];
    link = index == SURFEL_CELL_CAPACITY ? u_cell_surfels[SURFEL_OVERFLOW_HEADS + cell] :
        u_cell_surfels[SURFEL_OVERFLOW_LINKS + link];
    return link / 2u;
}
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
// With the frame cache, a surfel with SURFEL_MAX_SAMPLES converged: it only
// traces in one of SURFEL_CACHED_TRACE_PERIOD frames (they take turns), and its
// estimate uses the elapsed time since its last trace
const uint SURFEL_CACHED_TRACE_PERIOD = 4u;
// With the frame cache, a surfel with fewer than SURFEL_MAX_SAMPLES traces
// this many times SURFEL_RAYS in a frame, see surfel_gi_trace.comp
const uint SURFEL_YOUNG_RAY_SCALE = 4u;
// With the frame cache, the rays of a surfel in surfel_gi_trace.comp are
// traced by SURFEL_RAYS_PER_THREAD each of several threads (the threads of a
// surfel in the lists, see getSurfelTraceThreads()): the lists only have a
// few thousand surfels, one thread per surfel is so few waves that the gpu
// mostly waits for the rays one after another
const uint SURFEL_RAYS_PER_THREAD = 1u;

// The threads of a surfel of a list of u_trace_lists in surfel_gi_trace.comp
uint getSurfelTraceThreads(uint list)
{
    uint rays = list == SURFEL_TRACE_YOUNG && (pc.m_cache & SURFEL_CACHE_FRAME) != 0u ?
        SURFEL_RAYS * SURFEL_YOUNG_RAY_SCALE : SURFEL_RAYS;
    return rays / SURFEL_RAYS_PER_THREAD;
}

// Milliseconds a surfel which no pixel uses is kept, when the pool has enough
// free surfels (a quarter of the active pool)
const uint SURFEL_MAX_LIFE = 5000u;
// With fewer than a quarter of the active pool free, the
// life is shorter, down to SURFEL_MIN_LIFE when the pool is empty: the surfels
// which nothing used for the longest are recycled first, so a fast camera
// which keeps seeing new surfaces (a tunnel after a valley) doesn't wait for
// the old ones to expire
const uint SURFEL_MIN_LIFE = 500u;
// The cells of the first level are SURFEL_CELL_SIZE wide, up to
// SURFEL_LEVEL_DISTANCE from the cameras, every next level doubles both
#include "surfel_gi_grid.glsl"
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

// At most half a cell of its level, so the 2x2x2 cells nearest to a position
// (getSurfelNearestCells) have the centers of the surfels which touch it
float getSurfelRadius(float camera_distance)
{
    float radius = max(camera_distance, SURFEL_LEVEL_DISTANCE) *
        (SURFEL_CELL_SIZE / (4.0 * SURFEL_LEVEL_DISTANCE));
    return min(radius,
        getSurfelCellSize(getSurfelLevel(camera_distance)) * 0.5);
}

// The largest radius of a surfel which can touch a position at a distance from
// the closest camera: the distance of the surfel to it is at most its radius
// more or less (the distance to the closest camera can't change faster than
// the position), and its radius grows with that distance like
// getSurfelRadius(), so a cell farther than it from the position has no
// surfel which touches it (with a little margin for the rounding)
float getSurfelMaxRadius(float camera_distance)
{
    const float scale = SURFEL_CELL_SIZE / (4.0 * SURFEL_LEVEL_DISTANCE);
    return max(camera_distance * (scale / (1.0 - scale)),
        SURFEL_LEVEL_DISTANCE * scale) * 1.01 + 0.001;
}

// The squared distance of a position to a cell of the grid (0 inside)
float getSurfelCellDistance2(vec3 position, ivec3 cell, float cell_size)
{
    vec3 cell_min = vec3(cell) * cell_size;
    vec3 gap = max(max(cell_min - position,
        position - (cell_min + cell_size)), vec3(0.0));
    return dot(gap, gap);
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
    return getSurfelCell(surfelGICellAt(position, getSurfelCellSize(level)), level);
}

const float SURFEL_SHARING_RADIUS_SCALE = 2.0;

void binSurfel(uint id, vec3 position, float camera_distance, float radius)
{
    float reach = SURFEL_SHARING_RADIUS_SCALE * radius;
    uint first_level = getSurfelLevel(max(camera_distance - reach, 0.0));
    uint last_level = getSurfelLevel(camera_distance + reach);
    uint previous_cell = ~0u;
    for (uint level = first_level; level <= last_level; ++level)
    {
        uint cell = getSurfelCell(position, level);
        if (cell != previous_cell)
        {
            uint slot = atomicAdd(u_cell_count[cell], 1u);
            if (slot < SURFEL_CELL_CAPACITY)
                u_cell_surfels[cell * SURFEL_CELL_CAPACITY + slot] = id;
            else
            {
                uint link = 2u * id + (level & 1u);
                u_cell_surfels[SURFEL_OVERFLOW_LINKS + link] =
                    atomicExchange(u_cell_surfels[SURFEL_OVERFLOW_HEADS + cell], link);
            }
        }
        previous_cell = cell;
    }
}

// The first of the 2x2x2 cells of a level nearest to a position (it's at least
// half a cell from their sides)
ivec3 getSurfelNearestCells(vec3 position, float cell_size)
{
    return ivec3(floor(surfelGICenterGrid(position, cell_size)));
}

bool trySpawnSurfelInCell(uint cell)
{
    uint spawned = atomicAdd(u_cell_spawns[cell], 1u);
    return spawned < uint(ceil(SURFEL_CELL_SPAWNS_PER_STEP * pc.m_steps));
}

// A random number in [0, 1) of a seed
float getSurfelRandom(inout uint seed)
{
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word >> 8u) * (1.0 / 16777216.0);
}
