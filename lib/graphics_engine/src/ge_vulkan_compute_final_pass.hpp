#ifndef HEADER_GE_VULKAN_COMPUTE_FINAL_PASS_HPP
#define HEADER_GE_VULKAN_COMPUTE_FINAL_PASS_HPP

#include "ge_final_pass.hpp"
#include "ge_vulkan_post_processing.hpp"

namespace GE
{

// The stage of GEVulkanFinalChain for a pass added with addComputeFinalPass():
// one compute shader of the game (GEComputePassDesc). For every viewport
// (camera) of the FBO which the pass is active for, which never reads or
// writes outside of its rect (splitscreen):
//
// 0. final_pass_copy.comp (once for the whole image, if the viewports are not
//    all of it): output = input
// 1. The shader of the pass, over the rect of the viewport, with the header
//    (see GEFinalPassView, utils/final_pass.glsl) and the push constants the
//    game fills
class GEVulkanComputeFinalPass : public GEVulkanPostProcessing,
                                 public GEVulkanFinalStage
{
private:
    // The first bytes of the push constants, GEFinalPassHeader of the shader
    struct Header
    {
        int32_t m_rect[4];
        float m_params[4];
    };

    struct Item
    {
        Viewport m_viewport;

        GEFinalPassView m_view;
    };

    GEComputePassDesc m_desc;

    bool m_needs_depth;

    // Set 1: the camera of the viewport (utils/camera.glsl), only if
    // m_desc.m_camera_set
    VkDescriptorSetLayout m_data_layout;

    // Set 0 is the one of GEVulkanFinalChain::getIOLayout()
    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;

    VkPipeline m_copy_pipeline;
    // ------------------------------------------------------------------------
    // The size of the push constants, a multiple of 4
    uint32_t getPushSize() const
    {
        return sizeof(Header) + ((m_desc.m_push_size + 3) & ~3u);
    }
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    // The viewports of p which the pass is active for (m_size is what they
    // are clamped to)
    void collect(const std::vector<std::pair<GEVulkanDrawCall*,
                 GEVulkanCamera*> >& p, std::vector<Item>* items) const;
    // ------------------------------------------------------------------------
    VkPipeline createPipeline(const char* shader) const;
public:
    // ------------------------------------------------------------------------
    GEVulkanComputeFinalPass(GEVulkanDriver* vk,
                             VkDescriptorSetLayout io_layout,
                             const irr::core::dimension2d<irr::u32>& capacity,
                             bool needs_depth, const GEComputePassDesc& desc);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanComputeFinalPass();
    // ------------------------------------------------------------------------
    virtual bool needsDepth() const                   { return m_needs_depth; }
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
};   // GEVulkanComputeFinalPass

}

#endif
