#ifndef HEADER_GE_VULKAN_SURFEL_GI_HPP
#define HEADER_GE_VULKAN_SURFEL_GI_HPP

#include "ge_vma.hpp"
#include "ge_vulkan_post_processing.hpp"


#include <array>
#include <cstdint>

namespace GE
{
class GEVulkanAttachmentTexture;

// Surfel based diffuse global illumination (SurfelPlus, Wang et al. 2025),
// owned by GEVulkanDeferredFBOSplit (only one for all the viewports of it,
// like GEVulkanSSAO). It's all compute shaders between the g-buffer and the
// lighting render pass:
//
// 1. surfel_gi_prepare.comp: the cells of the grid and the counters of the
//    frame are cleared, on the first frame the stack of the free surfels is
//    filled
// 2. surfel_gi_update.comp: every surfel follows the triangle of the scene it
//    was created on (the node transformation of GEVulkanRTScene, see
//    utils/surfel_gi_rt.glsl), updates its radius, registers itself in the
//    cells it intersects and asks for the rays of the frame. The surfels
//    which are not seen, are far or are old are recycled
// 3. surfel_gi_cell_info.comp and surfel_gi_cell_to_surfel.comp: the lists of
//    the surfels of every cell of the grid
// 4. surfel_gi_trace.comp: a few cosine weighted rays per surfel, traced in
//    the scene of GEVulkanRTScene and shaded like the reflection of a solid
//    material (utils/rt_reflection.glsl), a ray which hits nothing sees the
//    skybox. The first hit of a ray spawns a sleeping surfel there, the ones
//    the field covers too, so the rays generate surfels along with the pixels
//    of the generate pass
// 5. surfel_gi_integrate.comp: one cooperative group per active surfel merges
//    the samples with the multiscale mean estimator of Ray Tracing Gems
//    chapter 25 (surfelMSME) and updates its irradiance and depth moments
// 6. surfel_gi_share.comp: shares irradiance using immutable SurfelLight and
//    updated depth moments, writing only the MSME mean
// 7. surfel_gi_generate.comp: the irradiance of every 2x2 block of the
//    viewport (the weighted mean of the surfels of its cell), and the spawn of
//    the surfels of the holes of the field (the blocks which no surfel covers,
//    at once) and of the pixels with the least coverage of their group
//
// There is no screen space temporal filter of the irradiance: the mean of a
// surfel is temporally stable (the multiscale mean estimator) and it only
// changes when its lighting changes. The image is half the size of the
// g-buffer, and its alpha is the coverage of the surfels (the confidence of
// the lighting pass).
//
// The pool, the cells and the rays are created once (the surfels are zeroed by
// the first frame, the stack of the free surfels and the counters are filled
// by surfel_gi_prepare.comp), the pipelines and the images are created again
// with the FBO
class GEVulkanSurfelGI : public GEVulkanPostProcessing
{
public:
    enum : uint32_t
    {
        // Surfels of the pool
        MAX_SURFELS = 131072,
        // Rays a surfel traces in a frame (the same as SURFEL_MAX_RAYS of
        // utils/surfel_gi.glsl): the count of the reference, 64
        MAX_RAYS = 64,
        // Rays a surfel which is warmed up traces in a frame (the same as
        // SURFEL_WARMUP_RAYS of utils/surfel_gi.glsl): the same count
        WARMUP_RAYS = 64,
        // Rays of all the surfels of a frame at most (the surfels which don't
        // get any keep their radiance, they're traced in the next frame) and
        // the size of the buffers of the rays
        RAY_BUDGET = MAX_SURFELS * WARMUP_RAYS,
        // (surfel, cell) pairs of a frame at most (a surfel is in the 27 cells
        // around it at most)
        CELL_SURFELS = MAX_SURFELS * 32,
        // Pixels of the tile of a surfel in the map of its irradiance, tiles
        // per row of the map and rows of the map
        IRRADIANCE_TILE = 6,
        IRRADIANCE_COLUMNS = 512,
        IRRADIANCE_ROWS = MAX_SURFELS / IRRADIANCE_COLUMNS,
        // Cells per axis of the inner cube of the grid, layers of each frustum
        // and the cells of all of them (the inner cube first, then 6 frustums).
        // The 32 layers reach ~44336 units from the camera, so the grid has
        // every surface of the scene whatever its distance
        CELL_N = 64,
        CELL_M = 32,
        CELL_COUNT = CELL_N * CELL_N * CELL_N +
            6 * CELL_M * CELL_N * CELL_N,
        // Frames a surfel lives when it's not seen
        MAX_LIFE = 1200
    };
private:
    // Size of a cell of the inner cube, half its size and the area of a surfel
    // in pixels (the same as the constants of utils/surfel_gi.glsl)
    static const float CELL_DELTA;
    static const float CELL_HALF_DELTA;
    static const float SURFEL_SIZE;

    // Same layout as MSMEData of utils/surfel_gi.glsl (std430, 48 bytes)
    struct MSMEData
    {
        float m_mean[3];
        float m_vbbr;
        float m_short_mean[3];
        float m_inconsistency;
        float m_variance[3];
        // Effective number of samples of the mean, bounded by
        // SURFEL_MSME_N_MAX
        float m_count;
    };
    static_assert(sizeof(MSMEData) == 48, "MSMEData is 48 bytes in the shaders");

    // Same layout as Surfel of utils/surfel_gi.glsl (std430, 144 bytes)
    struct Surfel
    {
        float m_position[3];
        float m_radius;
        float m_normal[3];
        // Key of the node of GEVulkanRTScene of the triangle (0 is the static
        // geometry, which is already in world space)
        uint32_t m_node_key;
        // Geometry in the instance of the node, triangle in it and the 2
        // barycentrics of the attachment (2 16 bit unorms)
        uint32_t m_geometry;
        uint32_t m_primitive;
        uint32_t m_bary;
        uint32_t m_ray_offset;
        uint32_t m_ray_count;
        uint32_t m_life;
        uint32_t m_frame;
        uint32_t m_status;
        MSMEData m_msme;
        // The mean of the samples at the end of the previous frame (the
        // "radiance" of SurfelPlus)
        float m_radiance[3];
        uint32_t m_last_seen;
        // floatBitsToUint of the sum of the 6x6 irradiance map, and 1 in the
        // lowest bit if it's full (it guides the rays then)
        uint32_t m_irradiance;
        // Visibility of the sun at the surfel, traced by its first ray (-1
        // while it was never traced), and the index of the R3 sequence of its
        // next ray
        float m_sun;
        uint32_t m_ray_index;
        // The shaders round the struct to 16 bytes
        uint32_t m_pad;
    };
    static_assert(sizeof(Surfel) == 144, "Surfel is 144 bytes in the shaders");

    // Same layout as SurfelLight of utils/surfel_gi.glsl (std430, 64 bytes):
    // the fields of a surfel which the passes which read every surfel of a cell
    // need, so they don't read the whole Surfel (144 bytes) for each of them
    struct SurfelLight
    {
        float m_position_radius[4];
        // Normal and number of frames since the surfel was created
        float m_normal_frame[4];
        float m_radiance[3];
        // Sums of the rows of the 6x6 irradiance map, 2 half floats each
        uint32_t m_row_sums_01;
        uint32_t m_row_sums_23;
        uint32_t m_row_sums_45;
        // Range of the depth moments of the surfel (0 while it has none), the
        // visibility of the field, and the visibility of the sun at the surfel
        float m_depth_range;
        float m_sun;
    };
    static_assert(sizeof(SurfelLight) == 64,
        "SurfelLight is 64 bytes in the shaders");

    // Same layout as SurfelCounter of utils/surfel_gi.glsl (std430, 36 bytes)
    struct SurfelCounter
    {
        int32_t m_alive;
        int32_t m_free;
        uint32_t m_cell_surfels;
        uint32_t m_rays;
        uint32_t m_active;
        uint32_t m_ray_requests;
        uint32_t m_last_ray_requests;
        // Sleeping surfels the tracing of the frame spawned
        uint32_t m_sleeping_spawns;
        // Spawns of the holes of the screen which the generate pass of the
        // frame tried
        uint32_t m_hole_spawns;
    };
    static_assert(sizeof(SurfelCounter) == 36,
        "SurfelCounter is 36 bytes in the shaders");

    struct Buffer
    {
        VkBuffer m_buffer;
        VmaAllocation m_memory;
        VkDeviceSize m_size;
    };

    enum BufferIndex : unsigned
    {
        BI_COUNTER = 0,
        BI_SURFEL,
        BI_DEAD,
        BI_CELL,
        BI_CELL_SURFEL,
        BI_RAY_ID,
        BI_RAY_RESULT,
        BI_SURFEL_LIGHT,
        BI_RAY_DATA,
        // VkDispatchIndirectCommand for tracing, then cooperative integration
        BI_DISPATCH,
        BI_ACTIVE,
        BI_COUNT
    };

    std::array<Buffer, BI_COUNT> m_buffers;

    // Half the size of the render target, irradiance in general layout all the
    // time (storage image and sampled)
    GEVulkanAttachmentTexture* m_gi;

    // The 6x6 octahedral irradiance map of every surfel (a tile per surfel,
    // r16f, storage image and sampled, general layout): the rays of a surfel
    // are guided by its bright directions
    GEVulkanAttachmentTexture* m_surfel_irradiance;

    // Depth moments in matching 6x6 tiles (rg32f, sampled and storage, general)
    GEVulkanAttachmentTexture* m_surfel_depth;

    // Binding 0 is the normal and 1 the depth of the g-buffer (combined image
    // samplers), 2 the storage image of the result, 3 to 9 the buffers of the
    // pool, the cells and the rays, 10 and 11 the irradiance map of the
    // surfels (a sampler and a storage image), 12 the compact copy of the
    // surfels, 13 ray PDF/distance, 14 trace/integration groups, 15 and 16
    // depth moments (sampler/storage), and 17 active surfel IDs
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    VkDescriptorSet m_set;

    // Same definition as the data descriptor set layout of draw calls (camera
    // and the global light), see GEVulkanSSAO
    VkDescriptorSetLayout m_data_layout;

    // The sets are the one of GEVulkanRTScene (0), m_data_layout (1), m_layout
    // (2), the mesh textures of the driver (3) and the environment (4, the
    // skybox of a miss and the diffuse of a hit which the field doesn't cover,
    // see surfel_gi_trace.comp). The layout of the last one is owned by
    // GEVulkanSkyBoxRenderer, which is created after the FBO (and so after this
    // pass), so the pipelines are created by the first render()
    VkPipelineLayout m_pipeline_layout;

    // The pipelines were already created (or failed to be created)
    bool m_pipelines_created;

    enum PipelineIndex : unsigned
    {
        PI_PREPARE = 0,
        PI_UPDATE,
        PI_CELL_INFO,
        PI_CELL_TO_SURFEL,
        PI_TRACE,
        PI_INTEGRATE,
        PI_SHARE,
        PI_GENERATE,
        PI_COUNT
    };

    std::array<VkPipeline, PI_COUNT> m_pipelines;

    // Incremented every frame with the passes
    uint32_t m_frame;

    // GEVulkanRTScene::getBuildCounter() the pool was built with: the surfels
    // are attached to the triangles of the scene, so when it's built again
    // they would follow triangles which don't exist anymore
    uint32_t m_rt_build_counter;

    // The pool was never initialized (the buffers are not zeroed and the stack
    // of the free surfels is not filled), see surfel_gi_prepare.comp
    bool m_needs_clear;

    // The image doesn't hold content computed by this pass (it was never
    // computed, or the ray traced scene was not ready since then), it's
    // cleared if the pass can't compute before the lighting pass reads it
    bool m_needs_image_clear;
    // ------------------------------------------------------------------------
    bool createBuffers();
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth,
                           GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    void createPipelines();
    // ------------------------------------------------------------------------
    void createPipeline(const char* shader, VkPipeline* pipeline,
                        const VkSpecializationInfo* specialization = NULL)
                                                                          const;
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    // The depth and the normal are the ones of the g-buffer, which are in read
    // only layouts after it. It needs GEVulkanRTScene to exist (not to be
    // ready)
    GEVulkanSurfelGI(GEVulkanDriver* vk,
                     const irr::core::dimension2d<irr::u32>& size,
                     GEVulkanAttachmentTexture* depth,
                     GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    ~GEVulkanSurfelGI();
    // ------------------------------------------------------------------------
    // Renders the global illumination of all the viewports between the
    // g-buffer and the lighting render pass (must be outside of render pass).
    // Returns false if there is nothing to render, the image has no valid
    // content then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the lighting pass in general layout (half the size)
    VkImageView getImageView() const;
};   // GEVulkanSurfelGI

}

#endif
