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
// viewports of it, like GEVulkanRTShadow). It's 4 compute shaders between the
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
//    not covered
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
        MAX_SURFELS = 32768,
        CELL_COUNT = 65536,
        CELL_CAPACITY = 32
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

    // Binding 0 is the depth and 1 is the normal of the g-buffer (combined
    // image samplers), 2 is the output, 3 to 6 the buffers
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
        PI_COUNT
    };

    std::array<VkPipeline, PI_COUNT> m_pipelines;

    // Incremented every frame which is rendered
    uint32_t m_frame;

    // GEVulkanRTScene::getBuildCount() of the surfels, they are on the
    // triangles of that scene
    uint32_t m_rt_build_count;

    // The pool was never cleared
    bool m_needs_reset;

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
    // Same as u_ibl of GEVulkanDrawCall::createPipeline, the environment maps
    // only exist then
    static bool isIBL();
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
};   // GEVulkanSurfelGI

}

#endif
