#ifndef HEADER_GE_VULKAN_SURFEL_GI_HPP
#define HEADER_GE_VULKAN_SURFEL_GI_HPP

#include "ge_vma.hpp"
#include "ge_vulkan_pipeline_constants.hpp"
#include "ge_vulkan_post_processing.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace GE
{
class GEVulkanAttachmentTexture;

// One shared surfel grid and one scrolling probe field per camera. Each frame
// updates persistent anchors, bins new surfels, traces the shared ray budget,
// then resolves quarter-rate diffuse irradiance and coverage per viewport.
class GEVulkanSurfelGI : public GEVulkanPostProcessing
{
public:
    // Same as the constants of utils/surfel_gi.glsl
    enum : uint32_t
    {
        // Surfels of every viewport, the buffers have
        // room for MAX_CAMERAS of them and a frame only uses the ones of its
        // viewports
        SURFELS_PER_CAMERA = 65536,
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
        // The list of the surfels which surfel_gi_trace.comp traces with the
        // frame cache (TRACE_*)
        uint32_t m_trace_list;
        uint32_t m_geometry_count;
        float m_depth_step;
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
        CACHE_FRAME = 1,
        RESET_CACHE = 16
    };

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
        BI_SPAWN_QUEUE,
        // With the frame cache, the surfels which surfel_gi_trace.comp traces
        // in the frame in TRACE_LIST_COUNT lists, every one with its indirect
        // dispatch and its count (see u_trace_lists of utils/surfel_gi.glsl)
        BI_TRACE_LIST,
        BI_PROBES,
        BI_RAY_WORK,
        BI_LIGHT_CACHE,
        BI_COUNT
    };

    struct Buffer
    {
        VkBuffer m_buffer;
        VmaAllocation m_memory;
    };

    std::array<Buffer, BI_COUNT> m_buffers;

    enum : uint32_t
    {
        PROBE_LEVEL_SIZE = 16 * 8 * 16,
        PROBE_BATCH = 256,
        PROBE_RAYS = 16,
        PRIMARY_RAY_BUDGET = 100000,
        PROBE_BANDS = 8
    };

    struct Probe
    {
        int32_t m_cell[3];
        uint32_t m_frame;
        float m_statistics[4];
        float m_offset[3];
        float m_proximity;
        uint32_t m_light[6 * 6];
        uint32_t m_depth[14 * 14];
        float m_sky_visibility[4];
    };
    static_assert(sizeof(Probe) == 992, "Probe must match the shader stride");

    unsigned m_probe_layers = MAX_CAMERAS;
    unsigned m_grid_probe_limit = 0;
    std::array<unsigned, MAX_CAMERAS> m_probe_levels = {};

    // RGBA16F, general layout all the time (storage image and sampled)
    GEVulkanAttachmentTexture* m_gi;
    GEVulkanAttachmentTexture* m_contact;

    GEVulkanAttachmentTexture* m_representatives;

    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    VkDescriptorSet m_set;

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

    // GEVulkanPipelineConstants::Options of the trace when the pipelines were
    // created, they are created again if it changes
    // (GEVulkanDriver::updateDriver doesn't create this again when only image
    // based lighting is toggled)
    GEVulkanPipelineConstants::Options m_options;

    enum PipelineIndex : unsigned
    {
        PI_CLEAR = 0,
        PI_MATERIALS,
        PI_UPDATE,
        PI_TRACE,
        PI_SHADOW,
        PI_TRACE_RESOLVE,
        PI_APPLY,
        PI_SPAWN,
        PI_SPAWN_BIN,
        PI_TRACE_SCHEDULE,
        PI_PROBE_TRACE,
        PI_PROBE_FILTER,
        PI_SHARE,
        PI_UPSCALE,
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
    float m_depth_step;
    float m_grid_cameras[MAX_CAMERAS][4] = {};

    // The output was cleared to black (nothing is traced), it's not cleared
    // again until something is traced
    bool m_cleared;

    // ------------------------------------------------------------------------
    void createBuffers();
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth,
                           GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    void createPipelines();
    // ------------------------------------------------------------------------
    // GEConfig::m_frame_cache
    static bool isFrameCache();
    // ------------------------------------------------------------------------
    void destroyPipelines();
    // ------------------------------------------------------------------------
    void createPipeline(const char* shader, VkPipeline* pipeline,
                        const VkSpecializationInfo* specialization = NULL)
                                                                         const;
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
    VkImageView getContactView() const;
    VkBuffer getProbeField() const { return m_buffers[BI_PROBES].m_buffer; }
    void prepareProbeField(const std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCamera*> >& views);
};   // GEVulkanSurfelGI

}

#endif
