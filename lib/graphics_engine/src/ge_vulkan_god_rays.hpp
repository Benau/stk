#ifndef HEADER_GE_VULKAN_GOD_RAYS_HPP
#define HEADER_GE_VULKAN_GOD_RAYS_HPP

#include "ge_vulkan_post_processing.hpp"

namespace GE
{

// Light shafts (god rays) of the sun of the track (GEVulkanDriver::setGodRays),
// the same look as the OpenGL renderer, owned by GEVulkanDeferredFBOSplit
// (there is only one for all the viewports of it, so splitscreen shares the
// images and the pipelines). It's all compute shaders between the g-buffer
// and the lighting render pass, so it doesn't need any graphics pipeline,
// render pass or mesh (the OpenGL renderer draws a sphere for the sun) of its
// own:
//
// 1. For each viewport (draw call) at a quarter of the size of it,
//    god_rays.comp marches from every pixel to the sun, which is a disc of the
//    screen space (using the camera data of the draw call) hidden by what is
//    closer in the depth of the g-buffer, and writes the sum of the sun mask
//    on the way (so the sun is in the image with its rays).
// 2. blur_hdr.comp (the gaussian mode of utils/blur.glsl) blurs it
//    horizontally and then vertically, never reading outside of the rect of
//    the viewport.
// 3. The lighting pass adds it to hdr together with the glow outline and the
//    light scattering (see lighting_composite.frag and
//    GEVulkanDrawCall::renderLightingComposite)
class GEVulkanGodRays : public GEVulkanPostProcessing
{
private:
    // A quarter of the size, in general layout all the time (storage image
    // and sampled). 0 is the result, 1 is the horizontal blur
    std::array<GEVulkanAttachmentTexture*, 2> m_rays;

    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    // 0: depth -> [0], 1: [0] -> [1] (horizontal), 2: [1] -> [0] (vertical)
    std::array<VkDescriptorSet, 3> m_sets;

    // Only BM_GAUSSIAN
    BlurPipelines m_blur_pipelines;

    // Same as GEVulkanLightScatter::m_data_layout, owned here for the same
    // reason
    VkDescriptorSetLayout m_data_layout;

    // Set 0 is m_layout and set 1 is m_data_layout
    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;
    // ------------------------------------------------------------------------
    static irr::core::dimension2d<irr::u32> quarterSize(
                                 const irr::core::dimension2d<irr::u32>& size);
    // ------------------------------------------------------------------------
    // Covers all pixels of a rect: floor of the start, ceil of the end
    static std::array<int32_t, 4> quarterRect(
                                           const std::array<int32_t, 4>& rect);
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth);
    // ------------------------------------------------------------------------
    void createBlurPipeline();
    // ------------------------------------------------------------------------
    void createPipeline();
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    // The depth is the one of the g-buffer, which is in read only depth layout
    // after it
    GEVulkanGodRays(GEVulkanDriver* vk,
                    const irr::core::dimension2d<irr::u32>& size,
                    GEVulkanAttachmentTexture* depth);
    // ------------------------------------------------------------------------
    ~GEVulkanGodRays();
    // ------------------------------------------------------------------------
    // Renders and blurs the rays of all the viewports, between the g-buffer
    // and the lighting render pass (must be outside of render pass). Returns
    // false if the track has no sun for it so nothing is done, the lighting
    // pass must skip it then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the lighting pass in general layout
    VkImageView getImageView() const;
};   // GEVulkanGodRays

}

#endif
