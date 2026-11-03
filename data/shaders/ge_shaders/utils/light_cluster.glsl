// Same values as GEVulkanLightHandler (ge_vulkan_light_handler.cpp)
#define LIGHT_CLUSTER_TILE_SIZE 16
#define LIGHT_CLUSTER_DEPTH_DIVISIONS 64

// The depth slices of the lights, generated every frame by
// GEVulkanLightHandler::generateClusters and bound after the light block of
// the camera buffer. Every slice has the compact list of the lights which
// reach it, and a bitmap per x strip and y strip of the screen, with a bit per
// light of the slice list (not every light of the scene).
layout(std430, set = 1, binding = 5) readonly buffer LightClusterBuffer
{
    uint  m_depth_divisions;
    uint  m_x_tiles;
    uint  m_y_tiles;
    uint  m_light_index_count;
    float m_kernel_near;
    float m_kernel_scale;
    float m_kernel_far;
    uint  m_reserved;
    uint  m_slice_light_offset[LIGHT_CLUSTER_DEPTH_DIVISIONS];
    uint  m_slice_bitmap_offset[LIGHT_CLUSTER_DEPTH_DIVISIONS];
    uint  m_slice_word_count[LIGHT_CLUSTER_DEPTH_DIVISIONS];
    uint  m_data[];
} u_light_cluster;

// Same parameters and output as accumulateLights of utils/pbr_light.glsl, the
// list of the lights is the cluster of the pixel
vec3 accumulateClusterLights(vec3 diffuse_color, vec3 normal, vec3 xpos,
                             vec3 eyedir, float perceptual_roughness,
                             float metallic, vec3 world_position,
                             vec3 world_normal)
{
    vec3 accumulated_color = vec3(0.0);
    if (u_light_cluster.m_depth_divisions == 0)
        return accumulated_color;

    // The slices are uniform in sqrt(distance), and the generator uses the
    // radial distance of the light from the camera
    int slice = clamp(int((sqrt(length(xpos)) - u_light_cluster.m_kernel_near) *
        u_light_cluster.m_kernel_scale), 0,
        int(u_light_cluster.m_depth_divisions) - 1);
    const uint word_count = u_light_cluster.m_slice_word_count[slice];
    const uint x_tiles = u_light_cluster.m_x_tiles;
    const uint y_tiles = u_light_cluster.m_y_tiles;
    if (word_count == 0 || x_tiles == 0 || y_tiles == 0)
        return accumulated_color;

    // The x strips go from the left to the right of the screen in the view
    // space, the y strips from the bottom to the top, and gl_FragCoord is top
    // down in the render target
    const uvec2 pixel = uvec2(max(gl_FragCoord.xy - u_camera.m_viewport.xy,
        vec2(0.0)));
    const uint xt = min(pixel.x / LIGHT_CLUSTER_TILE_SIZE, x_tiles - 1);
    const uint yt = y_tiles - 1 -
        min(pixel.y / LIGHT_CLUSTER_TILE_SIZE, y_tiles - 1);
    const uint bitmap_offset = u_light_cluster.m_slice_bitmap_offset[slice];
    const uint x_offset = bitmap_offset + xt * word_count;
    const uint y_offset = bitmap_offset + x_tiles * word_count +
        yt * word_count;
    const uint light_offset = u_light_cluster.m_slice_light_offset[slice];
    for (uint w = 0; w < word_count; w++)
    {
        uint bits = u_light_cluster.m_data[x_offset + w] &
            u_light_cluster.m_data[y_offset + w];
        while (bits != 0)
        {
            const uint local_index = w * 32 + findLSB(bits);
            bits &= bits - 1;
            const int light = int(u_light_cluster.m_data[light_offset +
                local_index]);
            accumulated_color += calculateLight(light, diffuse_color, normal,
                xpos, eyedir, perceptual_roughness, metallic, world_position,
                world_normal);
        }
    }
    return accumulated_color;
}
