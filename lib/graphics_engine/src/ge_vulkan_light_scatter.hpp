#ifndef HEADER_GE_VULKAN_LIGHT_SCATTER_HPP
#define HEADER_GE_VULKAN_LIGHT_SCATTER_HPP

#include "ge_vulkan_post_processing.hpp"

namespace GE
{

// Light scattering of the point lights in the fog (the same look as the
// OpenGL renderer, see light_scatter.comp), owned by GEVulkanDeferredFBOSplit
// (there is only one for all the viewports of it, so splitscreen shares the
// images and the pipelines). It's all compute shaders between the g-buffer
// and the lighting render pass, so it doesn't need any graphics pipeline or
// render pass on its own:
//
// 1. For each viewport (draw call) at half the size of it, light_scatter.comp
//    marches the ray of every pixel in the fog lit by the lights, using the
//    camera and light data of the draw call and the depth of the g-buffer.
// 2. light_scatter_blur.comp blurs it horizontally and then vertically, never
//    reading outside of the rect of the viewport.
// 3. The lighting pass adds it to hdr together with the glow outline (see
//    lighting_composite.frag and GEVulkanDrawCall::renderLightingComposite),
//    so it's above the fog of the lit geometry and the sky
class GEVulkanLightScatter : public GEVulkanPostProcessing
{
private:
    // Half size, in general layout all the time (storage image and sampled).
    // 0 is the result, 1 is the horizontal blur
    std::array<GEVulkanAttachmentTexture*, 2> m_scatter;

    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    // 0: depth -> [0], 1: [0] -> [1] (horizontal), 2: [1] -> [0] (vertical)
    std::array<VkDescriptorSet, 3> m_sets;

    VkPipelineLayout m_blur_pipeline_layout;

    VkPipeline m_blur_pipeline;

    // Same definition as the data descriptor set layout of draw calls (camera
    // and lights), so their sets can be bound with m_pipeline_layout. It's
    // owned here and not borrowed from a draw call, because draw calls (and
    // their layouts) are destroyed and created again (for example when
    // settings is updated) while this lives as long as the FBO, and a pipeline
    // layout can't be used after the set layout it was created with is gone
    VkDescriptorSetLayout m_data_layout;

    // Set 0 is m_layout and set 1 is m_data_layout
    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth);
    // ------------------------------------------------------------------------
    void createBlurPipeline();
    // ------------------------------------------------------------------------
    void createPipeline();
public:
    // ------------------------------------------------------------------------
    // The depth is the one of the g-buffer, which is in read only depth layout
    // after it
    GEVulkanLightScatter(GEVulkanDriver* vk,
                         const irr::core::dimension2d<irr::u32>& size,
                         GEVulkanAttachmentTexture* depth);
    // ------------------------------------------------------------------------
    ~GEVulkanLightScatter();
    // ------------------------------------------------------------------------
    // Renders and blurs the scatter of draw calls which have any, between the
    // g-buffer and the lighting render pass (must be outside of render pass).
    // Returns false if none of them has so nothing is done, the lighting pass
    // must skip it then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCameraSceneNode*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the lighting pass in general layout
    VkImageView getImageView() const;
};   // GEVulkanLightScatter

}

#endif
