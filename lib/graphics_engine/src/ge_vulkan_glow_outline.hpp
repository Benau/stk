#ifndef HEADER_GE_VULKAN_GLOW_OUTLINE_HPP
#define HEADER_GE_VULKAN_GLOW_OUTLINE_HPP

#include "vulkan_wrapper.h"

#include "dimension2d.h"

#include <array>
#include <utility>
#include <vector>

namespace GE
{
class GEVulkanAttachmentTexture;
class GEVulkanCameraSceneNode;
class GEVulkanDrawCall;
class GEVulkanDriver;

// Glow outline of the meshes with GERenderInfo::getGlowOutlineColor() != 0,
// owned by GEVulkanDeferredFBOSplit (there is only one for all the viewports
// of it, so splitscreen shares the images and the render pass):
//
// 1. Each draw call has its own list of glow draws (see
//    GEVulkanDrawCall::renderGlowOutline), they are rendered in one render
//    pass (one viewport each) with the depth of the g-buffer as a read only
//    depth attachment and depth test equal, so only the visible part of the
//    meshes is drawn, as the color of the glow.
// 2. The glow color is downscaled twice and upscaled once by a compute shader
//    (dual filter blur) for each viewport, which never reads outside of the
//    rect of the viewport so the glow doesn't leak into another viewport.
// 3. The lighting pass (see GEVulkanDrawCall::renderGlowOutlineComposite)
//    adds the blur with the area of the meshes themselves masked out, so the
//    meshes are rendered above their glow.
class GEVulkanGlowOutline
{
private:
    GEVulkanDriver* m_vk;

    irr::core::dimension2d<irr::u32> m_size;

    // Color of the glow, same size as the FBO, rgb is the color and alpha is
    // the coverage of the meshes
    GEVulkanAttachmentTexture* m_color;

    // Half size (from m_color), quarter size (from [0]) and half size again
    // (from [1], this is the one sampled by the lighting pass), and an eighth
    // size one which is only used by a viewport larger than the reference
    // (see blur()). They are in general layout all the time (storage image
    // and sampled)
    std::array<GEVulkanAttachmentTexture*, 4> m_blur;

    VkRenderPass m_render_pass;

    VkFramebuffer m_framebuffer;

    // Sampled by the lighting pass: 0 = m_color, 1 = m_blur[2]
    VkDescriptorSetLayout m_lighting_layout;

    VkDescriptorPool m_lighting_pool;

    VkDescriptorSet m_lighting_set;

    VkDescriptorSetLayout m_blur_layout;

    VkDescriptorPool m_blur_pool;

    // 0: m_color -> [0], 1: [0] -> [1], 2: [1] -> [2] (upscale),
    // 3: [1] -> [3], 4: [3] -> [1] (upscale)
    std::array<VkDescriptorSet, 5> m_blur_sets;

    VkPipelineLayout m_blur_pipeline_layout;

    VkPipeline m_blur_pipeline;
    // ------------------------------------------------------------------------
    void createRenderPass(GEVulkanAttachmentTexture* depth);
    // ------------------------------------------------------------------------
    void createLightingDescriptor();
    // ------------------------------------------------------------------------
    void createBlur();
    // ------------------------------------------------------------------------
    struct BlurViewport
    {
        // x0, y0, x1, y1 in pixels of the FBO
        std::array<int32_t, 4> m_rect;
        // Height of the viewport in pixels of the FBO, without rotation
        float m_height;
    };
    // ------------------------------------------------------------------------
    // The width of the glow is a fixed fraction of the height of each
    // viewport (the look at 1080 pixels high is the reference), so it doesn't
    // depend on the resolution, render scale or splitscreen
    void blur(VkCommandBuffer cmd,
              const std::vector<BlurViewport>& viewports);
public:
    // ------------------------------------------------------------------------
    // The depth is the one of the g-buffer, which is in read only depth layout
    // after it
    GEVulkanGlowOutline(GEVulkanDriver* vk,
                        const irr::core::dimension2d<irr::u32>& size,
                        GEVulkanAttachmentTexture* depth);
    // ------------------------------------------------------------------------
    ~GEVulkanGlowOutline();
    // ------------------------------------------------------------------------
    // Renders and blurs the glow of draw calls which have any, between the
    // g-buffer and the lighting render pass (must be outside of render pass).
    // Returns false if none of them has so nothing is done, the lighting pass
    // must skip GEVulkanDrawCall::renderGlowOutlineComposite then
    bool render(VkCommandBuffer cmd,
                const std::vector<std::pair<GEVulkanDrawCall*,
                GEVulkanCameraSceneNode*> >& p);
    // ------------------------------------------------------------------------
    VkRenderPass getRenderPass() const                { return m_render_pass; }
    // ------------------------------------------------------------------------
    VkDescriptorSetLayout getDescriptorSetLayout() const
                                                  { return m_lighting_layout; }
    // ------------------------------------------------------------------------
    const VkDescriptorSet* getDescriptorSet() const
                                                    { return &m_lighting_set; }
};   // GEVulkanGlowOutline

}

#endif
