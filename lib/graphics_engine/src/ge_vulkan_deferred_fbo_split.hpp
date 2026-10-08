#ifndef HEADER_GE_VULKAN_DEFERRED_FBO_SPLIT_HPP
#define HEADER_GE_VULKAN_DEFERRED_FBO_SPLIT_HPP

#include "ge_vulkan_deferred_fbo.hpp"

#include <memory>

namespace GE
{
class GEVulkanGlowOutline;
class GEVulkanLightScatter;

// Same output as GEVulkanDeferredFBO, but every stage is its own traditional
// render pass (single subpass) instead of a subpass of one, so there is no
// input attachment at all: the g-buffer (color, normal, depth) and the hdr
// image are written by one render pass and sampled as regular textures by the
// next one (shaders are compiled with SPLIT defined for it, see
// refreshDeferredSplit()). The pass structure is:
//
//   g-buffer   : color + normal + depth
//   [glow      : meshes with glow outline color, see GEVulkanGlowOutline, it's
//                its own render pass (and compute dispatches for the blur)
//                only if glow outline is enabled and any mesh needs it]
//   [scatter   : compute shaders only, light scattering of the lights in the
//                fog, see GEVulkanLightScatter, if it's enabled and any
//                viewport has fog and lights]
//   lighting   : (g-buffer as textures) -> hdr, skybox, [blurred glow and
//                light scattering, added together]
//   tonemap    : (hdr as texture) -> output, then ghost / transparent
//   [displace mask -> displace color, only if the FBO has displace support]
//
// The output of the tonemap pass is the swapchain image when there is no
// displace support, so it has the same format as the swapchain, otherwise it
// is the displace color texture, and the last displace color pass writes to
// the swapchain image. Like GEVulkanDeferredFBO the last render pass is left
// open by render().
class GEVulkanDeferredFBOSplit : public GEVulkanDeferredFBO
{
private:
    // ------------------------------------------------------------------------
    // Replace the input attachment descriptors of GEVulkanDeferredFBO (they
    // are not created if GEConfig::m_deferred_split != 0) with combined image
    // samplers of the same bindings, GVDFP_HDR is color + normal + depth for
    // the lighting pass, GVDFP_CONVERT_COLOR is hdr for the tonemap pass
    void initSplitGBufferDescriptor(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void initSplitConvertColorDescriptor(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void beginPass(VkCommandBuffer cmd, unsigned pass, unsigned framebuffer,
                   uint32_t clear_count, const VkClearValue* clears) const;

    // ------------------------------------------------------------------------
    // GVDFP_LIGHTING_COMPOSITE: glow color, glow blur and light scattering
    // (with a transparent texture for the ones which are not enabled)
    void initLightingCompositeDescriptor(GEVulkanDriver* vk);

    // Only one for all the viewports (draw calls) of this FBO, NULL if glow
    // outline is not enabled
    std::unique_ptr<GEVulkanGlowOutline> m_glow_outline;

    // Same, NULL if light scattering is not enabled
    std::unique_ptr<GEVulkanLightScatter> m_light_scatter;
public:
    // ------------------------------------------------------------------------
    GEVulkanDeferredFBOSplit(GEVulkanDriver* vk,
                             const core::dimension2d<u32>& size,
                             bool swapchain_output);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanDeferredFBOSplit();
    // ------------------------------------------------------------------------
    virtual void createRTT();
    // ------------------------------------------------------------------------
    virtual void render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCameraSceneNode*> >& p);
    // ------------------------------------------------------------------------
    virtual bool isSplit() const                               { return true; }
    // ------------------------------------------------------------------------
    // Decided when the FBO is created, draw calls only create the glow
    // pipelines (GVPT_GLOW_OUTLINE) if it has
    bool hasGlowOutline() const              { return m_glow_outline != NULL; }
    // ------------------------------------------------------------------------
    // Same for GVPT_LIGHTING_COMPOSITE (GVDFP_LIGHTING_COMPOSITE is the
    // descriptor of it), which draws them both
    bool hasLightScatter() const            { return m_light_scatter != NULL; }
    // ------------------------------------------------------------------------
    // pipeline_type is a GEVulkanPipelineType, the render pass which a
    // pipeline of this type is used in
    VkRenderPass getRenderPassForPipeline(unsigned pipeline_type) const;
};   // GEVulkanDeferredFBOSplit

}

#endif
