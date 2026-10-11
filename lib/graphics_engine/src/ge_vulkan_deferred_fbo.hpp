#ifndef HEADER_GE_VULKAN_DEFERRED_FBO_HPP
#define HEADER_GE_VULKAN_DEFERRED_FBO_HPP

#include "ge_vulkan_fbo_texture.hpp"

#include <array>
#include <utility>
#include <vector>

namespace GE
{
class GEVulkanDrawCall;
class GEVulkanCameraSceneNode;

enum GEVulkanDeferredFBOType : unsigned
{
    GVDFT_COLOR = 0,
    GVDFT_NORMAL,
    GVDFT_HDR,
    GVDFT_DISPLACE_MASK,
    GVDFT_DISPLACE_SSR,
    GVDFT_DISPLACE_COLOR,
    GVDFT_COUNT,
};

enum GEVulkanDeferredFBOPass : unsigned
{
    GVDFP_HDR = 0,
    GVDFP_CONVERT_COLOR,
    GVDFP_DISPLACE_MASK,
    GVDFP_DISPLACE_COLOR,
    // Only used by GEVulkanDeferredFBOSplit if glow outline or light
    // scattering is enabled: the glow color, its blurred version and the
    // blurred light scattering, sampled by the lighting pass
    GVDFP_LIGHTING_COMPOSITE,
    // Only used by GEVulkanDeferredFBOSplit if solid screen space reflection
    // is enabled (GEVulkanDrawCall::renderSolidSSR), it has no descriptor set
    // layout of its own: set 0 uses the layout of GVDFP_DISPLACE_COLOR (the
    // output of the tonemap pass of the previous frame, the depth and the
    // HiZ depth, see GEVulkanHiZDepth::getRenderingDescriptorSet, or
    // GVDFP_DISPLACE_MASK if there is no HiZ) and set 2 is GVDFP_HDR
    GVDFP_SOLID_SSR,
    GVDFP_COUNT,
};

class GEVulkanDeferredFBO : public GEVulkanFBOTexture
{
protected:
    std::array<GEVulkanAttachmentTexture*, GVDFT_COUNT> m_attachments;

    std::array<VkDescriptorSetLayout, GVDFP_COUNT> m_descriptor_layout;

    std::array<VkDescriptorPool, GVDFP_COUNT> m_descriptor_pool;

    std::array<VkDescriptorSet, GVDFP_COUNT> m_descriptor_set;

    const bool m_swapchain_output;
    // ------------------------------------------------------------------------
    // True while a separate offscreen RTT (not the swapchain FBO) is being
    // created, see setCreatingOffscreenRTT
    static bool s_creating_offscreen_rtt;
    // ------------------------------------------------------------------------
    // Input attachment descriptors (single render pass with subpasses), not
    // created when GEConfig::m_deferred_split != 0
    void initGBufferDescriptor(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void initConvertColorDescriptor(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void initDisplaceDescriptor(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    // Render passes and framebuffers of displace are put at the given indices
    // (mask_pass < color_pass, and the passes before them already created),
    // if the output is the swapchain there is one framebuffer per swapchain
    // image starting from color_pass for the last one
    void createDisplacePasses(unsigned mask_pass = GVDFP_DISPLACE_MASK,
                              unsigned color_pass = GVDFP_DISPLACE_COLOR);
public:
    // ------------------------------------------------------------------------
    // Needs to be true from before the constructor until createRTT() returns
    // for an offscreen RTT, because it shares pipelines with the swapchain
    // FBO and it needs alpha:
    // 1. The HDR attachment needs an alpha channel (B10G11R11 has none), so
    //    the area without any mesh keeps the (transparent) clear color and
    //    deferred_convert_color.frag can output it.
    // 2. Render passes need to be compatible with the ones of the swapchain
    //    FBO (including subpass dependencies), see
    //    GEVulkanDeferredFBOSplit::createRTT
    static void setCreatingOffscreenRTT(bool b)
                                            { s_creating_offscreen_rtt = b; }
    // ------------------------------------------------------------------------
    // output_attachment: the tonemap pass never renders to the swapchain
    // image directly, it renders to GVDFT_DISPLACE_COLOR (like if there is
    // displace) even if there is no displace support, so the output of the
    // previous frame can be sampled by the next one (it is decided by
    // GEVulkanDeferredFBOSplit if solid screen space reflection is enabled)
    GEVulkanDeferredFBO(GEVulkanDriver* vk, const core::dimension2d<u32>& size,
                        bool swapchain_output, bool output_attachment = false);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanDeferredFBO();
    // ------------------------------------------------------------------------
    virtual void createRTT();
    // ------------------------------------------------------------------------
    // Renders all deferred passes (g-buffer, lighting, tonemap and displace if
    // there is any) of one frame for the draw calls. Here the only render pass
    // created by createRTT() is already begun by the caller, and the last
    // render pass is left open (also when displace needs more of them)
    virtual void render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCameraSceneNode*> >& p);
    // ------------------------------------------------------------------------
    virtual bool isDeferredFBO() const                         { return true; }
    // ------------------------------------------------------------------------
    virtual bool useSwapChainOutput() const      { return m_swapchain_output; }
    // ------------------------------------------------------------------------
    virtual unsigned getZeroClearCountForPass(unsigned pass) const
    {
        switch (pass)
        {
        case GVDFP_HDR:
        {
            unsigned count = 0;
            for (unsigned i = 0; i < m_attachments.size(); i++)
            {
                if (i == GVDFT_HDR)
                    break;
                GEVulkanAttachmentTexture* t = m_attachments[i];
                if (t)
                    count++;
            }
            return count;
        }
        case GVDFP_DISPLACE_MASK:
            return getAttachment<GVDFT_DISPLACE_SSR>() ? 2 : 1;
        case GVDFP_DISPLACE_COLOR:
            return 1;
        default:
            return GEVulkanFBOTexture::getZeroClearCountForPass(pass);
        }
    }
    // ------------------------------------------------------------------------
    virtual VkDescriptorSetLayout getDescriptorSetLayout(unsigned id) const
                                         { return m_descriptor_layout.at(id); }
    // ------------------------------------------------------------------------
    virtual const VkDescriptorSet* getDescriptorSet(unsigned id) const
                                           { return &m_descriptor_set.at(id); }
    // ------------------------------------------------------------------------
    // True if displace materials can be rendered (the mask exists), the
    // GVDFT_DISPLACE_COLOR attachment can exist without it (see the
    // constructor), then the last render pass only copies it
    bool hasDisplace() const
    {
        return getAttachment<GVDFT_DISPLACE_MASK>() != NULL;
    }
    // ------------------------------------------------------------------------
    template<unsigned AttachmentType>
    GEVulkanAttachmentTexture* getAttachment() const
    {
        return std::get<AttachmentType>(m_attachments);
    }
};   // GEVulkanDeferredFBO

}

#endif
