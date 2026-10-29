#ifndef HEADER_GE_VULKAN_DEPTH_OF_FIELD_HPP
#define HEADER_GE_VULKAN_DEPTH_OF_FIELD_HPP

#include "ge_final_pass.hpp"
#include "ge_vulkan_post_processing.hpp"

namespace GE
{

// Depth of field of the final (tonemapped, display referred) image of the main
// FBO, the stage of GEVulkanFinalChain for GEConfig::m_dof. The image is
// blurred at half size, guided by the circle of confusion (coc) of every
// pixel, which is computed from the distance of the point to the camera (the
// settings m_dof_* of GEConfig, see utils/dof_common.glsl). For every viewport
// (camera) of the FBO, which never reads or writes outside of its rect
// (splitscreen), and the width of the blur is scaled with the height of it
// (REFERENCE_HEIGHT, the render scale is in it), so it's the same part of the
// screen at any resolution:
//
// 0. dof_copy.comp (once for the whole image, if the viewports are not all of
//    it): output = input
// 1. dof_downsample.comp: half size color weighted by the coc + the coc
// 2. dof_blur.comp, horizontal and then vertical: gaussian blur with the
//    radius from the coc, samples which don't reach the pixel are rejected
// 3. dof_composite.comp: mix of the sharp and the blurred image by the coc of
//    the full size pixel, with an upsample which rejects the samples with
//    another coc
//
// It samples the depth of the FBO (GEVulkanFBOTexture::hasStoredDepth()), so
// it's the first stage: the anti-aliasing runs after it, on the edges of the
// blur too. It works with any of them, or without. The two half size images
// are shared by the FBOs, they are as big as the biggest one (capacity)
class GEVulkanDepthOfField : public GEVulkanPostProcessing,
                             public GEVulkanFinalStage
{
private:
    enum Kernel : unsigned
    {
        K_COPY = 0,
        K_DOWNSAMPLE,
        K_BLUR,
        K_COMPOSITE,
        K_COUNT
    };

    // The sets of the half size images (set 2), the input is sampled and the
    // output is the storage image
    enum DescriptorSet : unsigned
    {
        DS_DOWNSAMPLE = 0,
        DS_BLUR_H,
        DS_BLUR_V,
        DS_COMPOSITE,
        DS_COUNT
    };

    // Push constants of utils/dof_common.glsl
    struct PushConstants
    {
        int32_t m_rect[4];
        int32_t m_half_rect[4];
        float m_focus[4];
        float m_params[4];
    };

    // rgba16f: color in rgb, signed coc in alpha. Always in general layout
    std::array<GEVulkanAttachmentTexture*, 2> m_half;

    // Set 2: binding 0 = combined image sampler, 1 = storage image (compute)
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    std::array<VkDescriptorSet, DS_COUNT> m_sets;

    // Set 1: the camera of the viewport, same as the other passes
    VkDescriptorSetLayout m_data_layout;

    // Set 0 is the one of GEVulkanFinalChain::getIOLayout()
    VkPipelineLayout m_pipeline_layout;

    std::array<VkPipeline, K_COUNT> m_pipelines;
    // ------------------------------------------------------------------------
    void createDescriptors();
    // ------------------------------------------------------------------------
    void createPipelines(VkDescriptorSetLayout io_layout);
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    GEVulkanDepthOfField(GEVulkanDriver* vk, VkDescriptorSetLayout io_layout,
                         const irr::core::dimension2d<irr::u32>& capacity);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanDepthOfField();
    // ------------------------------------------------------------------------
    virtual bool needsDepth() const                              { return true; }
    // ------------------------------------------------------------------------
    // Nothing to blur: the stage is skipped, not an image copied
    virtual bool isActive(const std::vector<std::pair<GEVulkanDrawCall*,
                          GEVulkanCamera*> >& p) const;
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
};   // GEVulkanDepthOfField

}

#endif
