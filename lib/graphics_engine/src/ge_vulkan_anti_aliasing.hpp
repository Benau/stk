#ifndef HEADER_GE_VULKAN_ANTI_ALIASING_HPP
#define HEADER_GE_VULKAN_ANTI_ALIASING_HPP

#include "ge_final_pass.hpp"
#include "ge_vulkan_post_processing.hpp"

#include "ge_vma.hpp"

namespace GE
{

// Anti-aliasing of the final (tonemapped, display referred) image of a FBO,
// the stage of GEVulkanFinalChain for GAA_CMAA2: CMAA2 (Conservative
// Morphological Anti-Aliasing 2.3 of Intel, see utils/cmaa2.glsl), which is
// all compute shaders and needs nothing else of the renderer (no velocity, no
// history, no multisampled image), so it works for any FBO. For every
// viewport (camera) of the FBO, which never reads or writes outside of its
// rect (splitscreen):
//
// 0. cmaa2_copy.comp (once for the whole image): output = input
// 1. cmaa2_edges.comp: the edges, and the list of pixels which can be the
//    start of a shape
// 2. cmaa2_args.comp, then cmaa2_candidates.comp (indirect): the colors of the
//    pixels of the shapes, in lists of the 2x2 quads
// 3. cmaa2_args.comp, then cmaa2_apply.comp (indirect): the average of the
//    list of every pixel, written over the copy
//
// The working buffers (storage buffers only) are shared by all the FBOs (and
// the viewports), they are as big as the biggest FBO (capacity)
class GEVulkanAntiAliasing : public GEVulkanPostProcessing,
                             public GEVulkanFinalStage
{
private:
    enum Kernel : unsigned
    {
        K_COPY = 0,
        K_EDGES,
        K_ARGS,
        K_CANDIDATES,
        K_APPLY,
        K_COUNT
    };

    enum Buffer : unsigned
    {
        B_EDGES = 0,
        B_HEADS,
        B_CANDIDATES,
        B_LOCATIONS,
        B_ITEMS,
        B_CONTROL,
        B_INDIRECT,
        B_COUNT
    };

    // Push constants of utils/cmaa2.glsl
    struct PushConstants
    {
        int32_t m_rect[4];
        uint32_t m_buf[4];
        uint32_t m_max_items;
    };

    std::array<VkBuffer, B_COUNT> m_buffers;

    std::array<VmaAllocation, B_COUNT> m_allocations;

    // Set 1: the buffers
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    VkDescriptorSet m_set;

    // Set 0 is the one of GEVulkanFinalChain::getIOLayout(), set 1 is m_layout
    VkPipelineLayout m_pipeline_layout;

    std::array<VkPipeline, K_COUNT> m_pipelines;

    // m_buf and m_max_items of the push constants, the same for all
    PushConstants m_constants;

    // True after the buffers have their start values (the first apply)
    bool m_initialized;
    // ------------------------------------------------------------------------
    void createBuffers(const irr::core::dimension2d<irr::u32>& capacity);
    // ------------------------------------------------------------------------
    void createDescriptors();
    // ------------------------------------------------------------------------
    void createPipelines(VkDescriptorSetLayout io_layout);
    // ------------------------------------------------------------------------
    void initBuffers(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    static void barrier(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    void push(VkCommandBuffer cmd, const std::array<int32_t, 4>& rect);
    // ------------------------------------------------------------------------
    void bind(VkCommandBuffer cmd, Kernel kernel);
public:
    // ------------------------------------------------------------------------
    GEVulkanAntiAliasing(GEVulkanDriver* vk, VkDescriptorSetLayout io_layout,
                         const irr::core::dimension2d<irr::u32>& capacity);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanAntiAliasing();
    // ------------------------------------------------------------------------
    // Not used, it's the stage of GEVulkanFinalChain (see apply)
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p)                { return false; }
    // ------------------------------------------------------------------------
    virtual void apply(VkCommandBuffer cmd, VkDescriptorSet io_set,
                       const irr::core::dimension2d<irr::u32>& size,
                       const std::vector<std::pair<GEVulkanDrawCall*,
                       GEVulkanCamera*> >& p);
};   // GEVulkanAntiAliasing

}

#endif
