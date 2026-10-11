#ifndef HEADER_GE_VULKAN_DEFERRED_FBO_SPLIT_HPP
#define HEADER_GE_VULKAN_DEFERRED_FBO_SPLIT_HPP

#include "ge_vulkan_deferred_fbo.hpp"

#include <memory>

namespace GE
{
class GEVulkanBloom;
class GEVulkanGlowOutline;
class GEVulkanHiZDepth;
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
//   [bloom     : compute shaders only, bloom of the finished hdr, see
//                GEVulkanBloom, if it's enabled and any viewport has anything
//                to draw]
//   tonemap    : (hdr as texture, and the bloom is added to it in the same
//                pass) -> output, then ghost / transparent
//   [displace mask -> displace color, only if the FBO has displace support]
//
// If screen space reflection is enabled (GEConfig::m_screen_space_reflection_
// type, needsSolidSSR()), there is also:
//
//   [hiz       : compute shaders only, the max depth mip chain of the depth of
//                every viewport (if it's the type of the reflection), see
//                GEVulkanHiZDepth, generated right after the g-buffer pass,
//                used by the lighting and displace pass]
//   lighting   : the solid materials reflect the output of the tonemap pass
//                of the previous frame (converted back to hdr), see
//                GEVulkanDrawCall::renderSolidSSR and solid_ssr.frag, which
//                needs the output to be a texture
//
// The output of the tonemap pass is the swapchain image when there is no
// displace support (and no screen space reflection), so it has the same
// format as the swapchain, otherwise it is the displace color texture, and the
// last displace color pass writes to the swapchain image (it only copies it if
// there is no displace). Like GEVulkanDeferredFBO the last render pass is left
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

    // Same, NULL if bloom is not enabled
    std::unique_ptr<GEVulkanBloom> m_bloom;

    // GEConfig::m_bloom_blend_texture, loaded by this FBO (NULL if there is
    // none or it failed to load)
    irr::video::ITexture* m_bloom_blend_texture;

    // One for each viewport (draw call) of the last frame, created when they
    // are needed (GEVulkanHiZDepth::isEnabled())
    std::vector<std::unique_ptr<GEVulkanHiZDepth> > m_hiz_depth;

    // Decided when the FBO is created, see needsSolidSSR()
    bool m_solid_ssr;

    // The output of the tonemap pass (the reflection of the next frame) has
    // anything to sample from the second frame
    bool m_solid_ssr_ready;
    // ------------------------------------------------------------------------
    void generateHiZ(VkCommandBuffer cmd,
                     const std::vector<std::pair<GEVulkanDrawCall*,
                     GEVulkanCameraSceneNode*> >& p);
public:
    // ------------------------------------------------------------------------
    // True if the solid materials reflect the previous frame, the tonemap pass
    // doesn't render to the swapchain image directly then (see the
    // constructor of GEVulkanDeferredFBO, not decided by GEVulkanDriver)
    static bool needsSolidSSR();
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
    // Same for the bloom which deferred_convert_color.frag adds to hdr
    // (the second binding of GVDFP_CONVERT_COLOR, the widest level of it and
    // the blend texture are the third and fourth)
    bool hasBloom() const                           { return m_bloom != NULL; }
    // ------------------------------------------------------------------------
    // Same for the blend texture (lens dust) of the bloom
    bool hasBloomBlend() const        { return m_bloom_blend_texture != NULL; }
    // ------------------------------------------------------------------------
    // Same for GVPT_SOLID_SSR (GVDFP_SOLID_SSR is the layout of it)
    bool hasSolidSSR() const                           { return m_solid_ssr; }
    // ------------------------------------------------------------------------
    // GVDFP_SOLID_SSR uses the layout and the set (the one without HiZ depth)
    // of the displace pass, see GVDFP_SOLID_SSR
    virtual VkDescriptorSetLayout getDescriptorSetLayout(unsigned id) const
    {
        if (id == GVDFP_SOLID_SSR)
        {
            return m_solid_ssr ? m_descriptor_layout[GVDFP_DISPLACE_COLOR] :
                VK_NULL_HANDLE;
        }
        return GEVulkanDeferredFBO::getDescriptorSetLayout(id);
    }
    // ------------------------------------------------------------------------
    virtual const VkDescriptorSet* getDescriptorSet(unsigned id) const
    {
        if (id == GVDFP_SOLID_SSR)
            return &m_descriptor_set[GVDFP_DISPLACE_MASK];
        return GEVulkanDeferredFBO::getDescriptorSet(id);
    }
    // ------------------------------------------------------------------------
    // pipeline_type is a GEVulkanPipelineType, the render pass which a
    // pipeline of this type is used in
    VkRenderPass getRenderPassForPipeline(unsigned pipeline_type) const;
};   // GEVulkanDeferredFBOSplit

}

#endif
