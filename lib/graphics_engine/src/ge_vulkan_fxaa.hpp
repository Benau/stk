#ifndef HEADER_GE_VULKAN_FXAA_HPP
#define HEADER_GE_VULKAN_FXAA_HPP

#include "ge_vulkan_final_chain.hpp"
#include "ge_vulkan_post_processing.hpp"

namespace GE
{

// The stage of GEVulkanFinalChain for GAA_FXAA: the one compute kernel of
// fxaa.comp, which needs no working buffer and no history, so it's the
// cheapest of the anti-aliasing (and the lowest quality). For every viewport
// (camera) of the FBO, which never reads or writes outside of its rect
// (splitscreen):
//
// 0. The whole image is copied, if the viewports are not all of it (the
//    stages have to write all pixels)
// 1. FXAA over the copy, only the pixels of the viewport
class GEVulkanFXAA : public GEVulkanPostProcessing, public GEVulkanFinalStage
{
private:
    // Push constants of fxaa.comp
    struct PushConstants
    {
        int32_t m_rect[4];
        int32_t m_copy;
    };

    // Set 0 is the one of GEVulkanFinalChain::getIOLayout()
    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    void dispatch(VkCommandBuffer cmd, const std::array<int32_t, 4>& rect,
                  bool copy);
public:
    // ------------------------------------------------------------------------
    GEVulkanFXAA(GEVulkanDriver* vk, VkDescriptorSetLayout io_layout,
                 const irr::core::dimension2d<irr::u32>& capacity);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanFXAA();
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
};   // GEVulkanFXAA

}

#endif
