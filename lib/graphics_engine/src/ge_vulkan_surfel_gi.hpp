#ifndef HEADER_GE_VULKAN_SURFEL_GI_HPP
#define HEADER_GE_VULKAN_SURFEL_GI_HPP

#include "ge_vma.hpp"
#include "ge_vulkan_post_processing.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace GE
{
class GEVulkanAttachmentTexture;

// Diffuse global illumination of the static scene with surfels (see
// isSurfelGI()), owned by GEVulkanDeferredFBOSplit (only one for all the
// viewports of it, like GEVulkanRTShadow). It's compute shaders between the
// g-buffer and the lighting pass, see utils/surfel_gi.glsl:
//
// 1. surfel_gi_clear.comp: the cells of the grid are cleared (and the pool on
//    the first frame or when the static scene is built again)
// 2. surfel_gi_update.comp: the surfels which are not used anymore are free,
//    the others are sized by their distance to the cameras and put in the
//    cells of the grid
// 3. surfel_gi_trace.comp: a few rays per surfel in the static scene, which
//    update the irradiance of the surfel
// 4. surfel_gi_apply.comp: the irradiance of every pixel from the surfels of
//    its cell (the output image), and new surfels where the static scene is
//    not covered. With the frame cache (GERayTracingConfig::m_gi_frame_cache)
//    a pixel reuses the irradiance of the last frame if its surface is still
//    on the screen (the history), and the pixels which can't are done by
//    surfel_gi_gather.comp and surfel_gi_spawn.comp after it
//
// Only the static BLAS of GEVulkanRTScene (GRTIM_STATIC) is traced: the
// surfels are created on it and never move, so they don't follow the nodes
// which move, and the karts don't block the rays or the shadows of the hits.
// The pixels of the dynamic nodes still get the irradiance of the surfels
// around them.
//
// The output has the size of the FBO: the irradiance (the same as the diffuse
// environment map, which the lighting pass replaces with it) and the coverage
// of the surfels in alpha. It's black without a ray traced scene (menus).
class GEVulkanSurfelGI : public GEVulkanPostProcessing
{
public:
    // Same as the constants of utils/surfel_gi.glsl
    enum : uint32_t
    {
        // Surfels of every viewport (SURFEL_PER_CAMERA), the buffers have
        // room for MAX_CAMERAS of them and a frame only uses the ones of its
        // viewports
        SURFELS_PER_CAMERA = 32768,
        MAX_CAMERAS = 4,
        MAX_SURFELS = SURFELS_PER_CAMERA * MAX_CAMERAS,
        CELL_COUNT = 65536,
        CELL_CAPACITY = 64
    };
private:
    // Same as Surfel of utils/surfel_gi.glsl (std430)
    struct Surfel
    {
        float m_position[3];
        float m_radius;
        float m_normal[3];
        uint32_t m_last_seen;
        float m_irradiance[3];
        float m_samples;
    };
    static_assert(sizeof(Surfel) == 48, "Surfel is 48 bytes in the shaders");

    // Same as SurfelPushConstants of utils/surfel_gi.glsl
    struct SurfelPushConstants
    {
        float m_cameras[4][4];
        int32_t m_rect[4];
        uint32_t m_frame;
        uint32_t m_camera_count;
        uint32_t m_reset;
        // The env set of the first viewport has the skybox (binding 2), not
        // the dummy one (no skybox, or it's loading)
        uint32_t m_has_skybox;
        // Milliseconds since the start (wraps, the life of the surfels)
        uint32_t m_time;
        // Time of the frame in steps of 1/60 s (the constants of time of the
        // shaders are in them, so they don't depend on the frame rate)
        float m_steps;
        // SURFELS_PER_CAMERA of every camera, the pool in use
        uint32_t m_surfel_count;
        // CACHE_* bits, see isFrameCache()
        uint32_t m_cache;
        // The viewport of surfel_gi_apply.comp (its PrevCamera)
        uint32_t m_viewport_index;
        // The list of the surfels which surfel_gi_trace.comp traces with the
        // frame cache (TRACE_*)
        uint32_t m_trace_list;
    };
    static_assert(sizeof(SurfelPushConstants) <= 128, "The push constants "
        "which every device has");

    // The lists of BI_TRACE_LIST
    enum : uint32_t
    {
        // The surfels which didn't converge
        TRACE_YOUNG = 0,
        // The converged surfels whose turn it is to trace
        TRACE_CONVERGED,
        TRACE_LIST_COUNT
    };

    // Bits of m_cache of the push constants
    enum : uint32_t
    {
        // The frame cache: the irradiance of the pixels of the last frame is
        // reused, and the converged surfels trace fewer rays (see
        // surfel_gi_apply.comp and surfel_gi_trace.comp)
        CACHE_FRAME = 1,
        // The history of the last frame can be read, it was written by the
        // same viewports with the cache on
        CACHE_HISTORY = 2,
        // The pixels which can't be reused are in a list, which
        // surfel_gi_gather.comp looks for the surfels of (and
        // surfel_gi_spawn.comp spawns in the tiles)
        CACHE_DEFER = 4
    };

    // Same as SurfelPrevCamera of utils/surfel_gi.glsl (std430), the camera of
    // a viewport in the last frame, which the history of the pixels was
    // rendered with
    struct PrevCamera
    {
        float m_projection_view_matrix[16];
        float m_viewport[4];
        float m_position[4];
    };
    static_assert(sizeof(PrevCamera) == 96, "PrevCamera is 96 bytes in the "
        "shaders");

    enum BufferIndex : unsigned
    {
        // The surfels
        BI_SURFEL = 0,
        // The count and the stack of the free surfels
        BI_POOL,
        // The number of surfels of every cell, and their indices
        BI_CELL_COUNT,
        BI_CELL_SURFEL,
        // The number of surfels spawned in every cell in the frame
        BI_CELL_SPAWN,
        // PrevCamera of every viewport
        BI_PREV_CAMERA,
        // The pixels which can't be reused (the count, the indirect dispatch
        // of surfel_gi_gather.comp and the pixels), and the spawn of every
        // 8x8 tile (see utils/surfel_gi_apply.glsl), for CACHE_DEFER
        BI_MISS,
        BI_TILE_SPAWN,
        // With the frame cache, the surfels which surfel_gi_trace.comp traces
        // in the frame in TRACE_LIST_COUNT lists, every one with its indirect
        // dispatch and its count (see u_trace_lists of utils/surfel_gi.glsl)
        BI_TRACE_LIST,
        BI_COUNT
    };

    struct Buffer
    {
        VkBuffer m_buffer;
        VmaAllocation m_memory;
    };

    std::array<Buffer, BI_COUNT> m_buffers;

    // RGBA16F, general layout all the time (storage image and sampled)
    GEVulkanAttachmentTexture* m_gi;

    // The irradiance of every pixel which can be reused in the next frame
    // (RG32UI, general layout, see surfel_gi_apply.comp), one is read and the
    // other one written in a frame, they swap every frame
    std::array<GEVulkanAttachmentTexture*, 2> m_history;

    // Binding 0 is the depth and 1 is the normal of the g-buffer (combined
    // image samplers), 2 is the output, 3 to 11 the buffers, 12 the history
    // which is read and 13 the one which is written
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    // Set i reads m_history[i] and writes the other one
    std::array<VkDescriptorSet, 2> m_sets;

    // Same definition as the data descriptor set layout of draw calls (camera
    // and the global light), see GEVulkanRTShadow
    VkDescriptorSetLayout m_data_layout;

    // The sets are the one of GEVulkanRTScene (0), m_data_layout (1), m_layout
    // (2), the mesh textures of the driver (3) and the environment (4), the
    // same for all the passes. The environment layout belongs to
    // GEVulkanSkyBoxRenderer, which is created after the FBO, so the pipelines
    // are created by the first render()
    VkPipelineLayout m_pipeline_layout;

    // The layout of the mesh textures is replaced when its pool grows, the
    // pipelines are created again then
    std::weak_ptr<VkDescriptorSetLayout> m_mesh_texture_layout;

    // u_ibl of the trace (isIBL()) when the pipelines were created, they are
    // created again if it changes (GEVulkanDriver::updateDriver doesn't create
    // this again when only image based lighting is toggled)
    bool m_ibl;

    enum PipelineIndex : unsigned
    {
        PI_CLEAR = 0,
        PI_UPDATE,
        PI_TRACE,
        PI_APPLY,
        PI_GATHER,
        PI_SPAWN,
        PI_COUNT
    };

    std::array<VkPipeline, PI_COUNT> m_pipelines;

    // Incremented every frame which is rendered
    uint32_t m_frame;

    // m_time of the push constants, and getMonoTimeMs() of the last frame
    uint32_t m_time;
    uint64_t m_last_render;

    // GEVulkanRTScene::getBuildCount() of the surfels, they are on the
    // triangles of that scene
    uint32_t m_rt_build_count;

    // The pool was never cleared
    bool m_needs_reset;
    // The pool in use, the pool is cleared when it changes (the cameras of
    // the viewports)
    uint32_t m_surfel_count;

    // The output was cleared to black (nothing is traced), it's not cleared
    // again until something is traced
    bool m_cleared;

    // The camera of every viewport in the last frame, and how many viewports
    // it had (0 if it can't be reused: the cache was off, or the pool was
    // cleared)
    std::array<PrevCamera, MAX_CAMERAS> m_prev_cameras;
    unsigned m_prev_camera_count;

    // The history which the next frame reads (the index of m_sets)
    unsigned m_history_index;
    // ------------------------------------------------------------------------
    void createBuffers();
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth,
                           GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    void createPipelines();
    // ------------------------------------------------------------------------
    // Same as u_ibl of GEVulkanDrawCall::createPipeline, the environment maps
    // only exist then
    static bool isIBL();
    // ------------------------------------------------------------------------
    // GERayTracingConfig::m_gi_frame_cache
    static bool isFrameCache();
    // ------------------------------------------------------------------------
    void destroyPipelines();
    // ------------------------------------------------------------------------
    void createPipeline(const char* shader, VkPipeline* pipeline,
                        const VkSpecializationInfo* specialization = NULL)
                                                                         const;
    // ------------------------------------------------------------------------
    // Empties the list of the misses and the spawns of the tiles before
    // surfel_gi_apply.comp of a viewport (CACHE_DEFER)
    void clearMissList(VkCommandBuffer cmd, bool first);
    // ------------------------------------------------------------------------
    // Black, and sampled by the lighting pass after it
    void clearOutput(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    // The depth and the normal are the ones of the g-buffer, which are in read
    // only layouts after it
    GEVulkanSurfelGI(GEVulkanDriver* vk,
                     const irr::core::dimension2d<irr::u32>& size,
                     GEVulkanAttachmentTexture* depth,
                     GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    ~GEVulkanSurfelGI();
    // ------------------------------------------------------------------------
    // Renders the irradiance of all the viewports between the g-buffer and
    // the lighting render pass (must be outside of render pass). Returns false
    // if nothing is traced, the output is black then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the lighting pass in general layout
    VkImageView getImageView() const;
};   // GEVulkanSurfelGI

}

#endif
