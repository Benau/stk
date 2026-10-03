#ifndef HEADER_GE_VULKAN_DEFERRED_FBO_SPLIT_HPP
#define HEADER_GE_VULKAN_DEFERRED_FBO_SPLIT_HPP

#include "ge_vulkan_deferred_fbo.hpp"

namespace GE
{
// Order of the render passes (and of their framebuffers, except that the last
// pass has one framebuffer per swapchain image if it renders to the swapchain
// directly, which are put from its index) in m_rtt_render_pass
enum GEVulkanDeferredSplitPass : unsigned
{
    GVDSP_GBUFFER = 0,
    GVDSP_LIGHTING,
    GVDSP_TONEMAP,
    GVDSP_DISPLACE_MASK,
    GVDSP_DISPLACE_COLOR,
    GVDSP_COUNT,
};

// Same output as GEVulkanDeferredFBO, but every stage is its own traditional
// render pass (single subpass) instead of a subpass of one, so there is no
// input attachment at all: the g-buffer (color, normal, depth) and the hdr
// image are written by one render pass and sampled as regular textures by the
// next one (shaders are compiled with SPLIT defined for it, see
// refreshDeferredSplit()). The pass structure is:
//
//   g-buffer   : color + normal + depth
//   lighting   : (g-buffer as textures) -> hdr, skybox
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
    // are not created if GEConfig::m_deferred_split is true) with combined
    // image samplers of the same bindings, GVDFP_HDR is color + normal + depth
    // for the lighting pass, GVDFP_CONVERT_COLOR is hdr for the tonemap pass
    void initSplitGBufferDescriptor(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void initSplitConvertColorDescriptor(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void beginPass(VkCommandBuffer cmd, unsigned pass, unsigned framebuffer,
                   uint32_t clear_count, const VkClearValue* clears) const;
public:
    // ------------------------------------------------------------------------
    GEVulkanDeferredFBOSplit(GEVulkanDriver* vk,
                             const core::dimension2d<u32>& size,
                             bool swapchain_output);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanDeferredFBOSplit() {}
    // ------------------------------------------------------------------------
    virtual void createRTT();
    // ------------------------------------------------------------------------
    virtual void render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCameraSceneNode*> >& p);
    // ------------------------------------------------------------------------
    virtual bool isSplit() const                               { return true; }
    // ------------------------------------------------------------------------
    // pipeline_type is a GEVulkanPipelineType, the render pass which a
    // pipeline of this type is used in
    VkRenderPass getRenderPassForPipeline(unsigned pipeline_type) const;
};   // GEVulkanDeferredFBOSplit

}

#endif
