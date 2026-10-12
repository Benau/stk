#ifndef HEADER_GE_VULKAN_SSAO_HPP
#define HEADER_GE_VULKAN_SSAO_HPP

#include "ge_vulkan_post_processing.hpp"

namespace GE
{
class GEVulkanArrayTexture;

// Screen space ambient occlusion (the same look as ssao.frag of the OpenGL
// renderer), owned by GEVulkanDeferredFBOSplit (only one for all the viewports
// of it, so splitscreen shares the images and the pipelines). It's all compute
// shaders between the g-buffer and the lighting render pass, in the same
// place as GEVulkanLightScatter:
//
// 1. ssao_depth.comp: the linear view space depth of the g-buffer (level 0 of
//    the depth chain) and the filtered 2x2 blocks of it (level 1), then
//    ssao_depth_down.comp for the next levels (up to level 3). The chain has
//    its own filter and values, it can't be shared with the HiZ depth (see
//    GEVulkanHiZDepth) which has the raw closest depth for tracing rays
// 2. ssao.comp: the visibility at half the size, using the depth chain and
//    the normal of the g-buffer (the same as solid_ssr.frag uses). The
//    estimator is a single function of the shader, a different one (GTAO)
//    can replace it without changing anything else
// 3. ssao_blur.comp: horizontal and then vertical blur of it which doesn't
//    blur over the edges of the depth, its width is relative to the height of
//    the viewport (REFERENCE_HEIGHT like the other blurs) so it's resolution
//    independent
// 4. ssao_upsample.comp: the full size image (bilinear, but weighted by the
//    depth), which is sampled by the lighting pass (deferred_pbr.frag) for
//    the ambient light only, like the OpenGL renderer does
//
// Everything is done for each viewport, with its own camera and rect, and
// never reads outside of it
class GEVulkanSSAO : public GEVulkanPostProcessing
{
private:
    // Levels of the depth chain which are used (ssao.comp can fetch them all)
    static const unsigned DEPTH_LEVELS = 4;

    // Linear depth, level 0 has the size of the g-buffer. In general layout
    // all the time (storage image and sampled)
    GEVulkanArrayTexture* m_depth;

    // Storage image view of every level of m_depth
    std::array<VkImageView, DEPTH_LEVELS> m_depth_views;

    // Half size, in general layout all the time. 0 is the result of ssao.comp
    // and then the blur, 1 is the horizontal blur
    std::array<GEVulkanAttachmentTexture*, 2> m_ao_half;

    // Full size, the result which the lighting pass samples (general layout
    // all the time)
    GEVulkanAttachmentTexture* m_ao;

    // Binding 0 and 1 are combined image samplers and 2 and 3 are storage
    // images (the unused ones are written with the same image as another one
    // so a descriptor is never left unwritten)
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    enum DescriptorSet : unsigned
    {
        // depth of the g-buffer -> level 0 and 1
        DS_DEPTH = 0,
        // level 1 -> 2, level 2 -> 3
        DS_DOWN_1,
        DS_DOWN_2,
        // depth chain and normal -> m_ao_half[0]
        DS_AO,
        // m_ao_half[0] -> [1] (horizontal), [1] -> [0] (vertical)
        DS_BLUR_H,
        DS_BLUR_V,
        // m_ao_half[0] -> m_ao
        DS_UPSAMPLE,
        DS_COUNT
    };
    std::array<VkDescriptorSet, DS_COUNT> m_sets;

    // Same definition as the data descriptor set layout of draw calls (camera),
    // so their sets can be bound with m_pipeline_layout. It's owned here for
    // the same reason as in GEVulkanLightScatter
    VkDescriptorSetLayout m_data_layout;

    // Set 0 is m_layout and set 1 is m_data_layout
    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_depth_pipeline;

    VkPipeline m_down_pipeline;

    VkPipeline m_ao_pipeline;

    VkPipeline m_blur_pipeline;

    VkPipeline m_upsample_pipeline;
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth,
                           GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    void createPipelines();
    // ------------------------------------------------------------------------
    void createPipeline(const char* shader, VkPipeline* pipeline) const;
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    // The depth and the normal are the ones of the g-buffer, which are in read
    // only layouts after it
    GEVulkanSSAO(GEVulkanDriver* vk,
                 const irr::core::dimension2d<irr::u32>& size,
                 GEVulkanAttachmentTexture* depth,
                 GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    ~GEVulkanSSAO();
    // ------------------------------------------------------------------------
    // Renders the ambient occlusion of all the viewports between the g-buffer
    // and the lighting render pass (must be outside of render pass). Returns
    // false if there is nothing to render, the image has no valid content then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCameraSceneNode*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the lighting pass in general layout
    VkImageView getImageView() const;
};   // GEVulkanSSAO

}

#endif
