// GPU culling of the static scene (see GEVulkanStaticDrawSource), the part
// which is the same for every view. Each thread tests one source instance and
// appends it to the object data of its draw command, the command was copied
// from the template with a zero instance count just before this dispatch.
//
// A shader includes this file, then one file which defines isInView (the test
// of the view, static_draws_view.glsl or static_draws_sphere.glsl), then one
// which defines the statistics (static_draws_stats.glsl or
// static_draws_no_stats.glsl)
layout(local_size_x = 64) in;

// The uniform buffer of the view that is culled (set 0, binding 0)
#define CAMERA_SET 0
#include "camera.glsl"

layout(push_constant) uniform LODConstants
{
    // The camera position used to select the level of detail (xyz)
    vec4 m_lod_camera;
} u_lod;

struct Source
{
    uvec4 object_data[4];
    vec3 origin;
    int min_distance;
    vec3 min_edge;
    int max_distance;
    vec3 max_edge;
    uint command;
};

struct Draw
{
    uint index_count;
    uint instance_count;
    uint first_index;
    int vertex_offset;
    uint first_instance;
};

layout(std430, set = 0, binding = 1) readonly buffer Sources
{
    Source sources[];
};

layout(std430, set = 0, binding = 2) buffer Commands
{
    Draw draws[];
};

layout(std430, set = 0, binding = 3) writeonly buffer Objects
{
    uvec4 objects[];
};

// Defined by the test of the view
bool isInView(const Source source);

// Defined by the statistics, all the threads of the workgroup must call them
// (they may have barriers)
void statsBegin();
void statsAddVisible(uint polygons);
void statsEnd();

// ----------------------------------------------------------------------------
bool isInLevelOfDetail(const Source source)
{
    // Squared distance interval of the level of detail, the maximum is
    // negative if there is none
    vec3 delta = u_lod.m_lod_camera.xyz - source.origin;
    precise float squared = delta.x * delta.x + delta.y * delta.y +
        delta.z * delta.z;
    int distance_sq = int(squared);
    return distance_sq >= source.min_distance &&
        (source.max_distance < 0 || distance_sq < source.max_distance);
}   // isInLevelOfDetail

// ----------------------------------------------------------------------------
void main()
{
    // All the threads must reach the barriers, so no early return
    statsBegin();

    uint index = gl_GlobalInvocationID.x;
    if (index < uint(sources.length()))
    {
        Source source = sources[index];
        if (isInLevelOfDetail(source) && isInView(source))
        {
            uint slot = atomicAdd(draws[source.command].instance_count, 1u);
            uint dest = (draws[source.command].first_instance + slot) * 4u;
            for (uint word = 0u; word < 4u; word++)
                objects[dest + word] = source.object_data[word];
            statsAddVisible(draws[source.command].index_count / 3u);
        }
    }

    statsEnd();
}   // main
