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
class GEVulkanRTShadow;
class GEVulkanSSAO;

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
// When direct swapchain output is requested, tonemap writes to the swapchain
// only when neither displace nor solid SSR is active; otherwise it uses the
// displace-color attachment when present, or this FBO's offscreen image
// for solid-SSR history. If using the swapchain, the final displace-color
// pass writes to it (copying only when there is no displace). Like the base
// FBO, the last pass is left open by render().
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

    // Same, NULL if the ambient occlusion is not enabled
    std::unique_ptr<GEVulkanSSAO> m_ssao;

    // The visibility of the ray traced shadows which the lighting pass samples,
    // it is always created (inactive without ray tracing) so the descriptor of
    // the g-buffer is valid
    std::unique_ptr<GEVulkanRTShadow> m_rt_shadow;

    // Same, NULL if bloom is not enabled
    std::unique_ptr<GEVulkanBloom> m_bloom;

    // GEConfig::m_bloom_blend_texture, loaded by this FBO (NULL if there is
    // none or it failed to load)
    irr::video::ITexture* m_bloom_blend_texture;

    // One per viewport (draw call), created when HiZ SSR is enabled by
    // GEVulkanHiZDepth::isEnabled()
    std::vector<std::unique_ptr<GEVulkanHiZDepth> > m_hiz_depth;

    // Decided when the FBO is created, see needsSolidSSR()
    const bool m_solid_ssr;

    // Decided when the FBO is created, see needsSolidRT()
    const bool m_solid_rt;

    // History is available from the second frame onward; the first frame has
    // no previous tonemap output to sample
    bool m_solid_ssr_ready;
    // ------------------------------------------------------------------------
    void generateHiZ(VkCommandBuffer cmd, bool has_displace_fbo,
                     const std::vector<std::pair<GEVulkanDrawCall*,
                     GEVulkanCamera*> >& p);
public:
    // ------------------------------------------------------------------------
    static bool needsSolidSSR();
    // ------------------------------------------------------------------------
    // True when ray tracing is on (the FBO has the passes of it, which of them
    // is rendered is decided every frame by isSolidRayTracing() and
    // getRTShadowType(), so the FBO doesn't change with them)
    static bool needsSolidRT();
    // ------------------------------------------------------------------------
    static bool resolveSwapchainOutput(GEVulkanDriver* vk,
                                       const core::dimension2d<u32>& size,
                                       bool swapchain_output);
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
                        GEVulkanCamera*> >& p);
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
    // Same for the ambient occlusion which the lighting pass samples (the
    // fourth binding of GVDFP_HDR)
    bool hasSSAO() const                             { return m_ssao != NULL; }
    // Size of the ray traced shadow relative to the FBO, 1.0 if there is none
    float getRTShadowScale() const;
    // ------------------------------------------------------------------------
    // Same for the blend texture (lens dust) of the bloom
    bool hasBloomBlend() const        { return m_bloom_blend_texture != NULL; }
    // ------------------------------------------------------------------------
    // Same for GVPT_SOLID_SSR (GVDFP_SOLID_SSR is the layout of it)
    bool hasSolidSSR() const                           { return m_solid_ssr; }
    // ------------------------------------------------------------------------
    // Same for GVPT_SOLID_RT (it has its own pipeline layout in
    // GEVulkanDrawCall)
    bool hasSolidRT() const                              { return m_solid_rt; }
    // ------------------------------------------------------------------------
    // GVDFP_SOLID_SSR uses the layout and the set (the one without HiZ depth)
    // of the displace pass, see GVDFP_SOLID_SSR
    virtual VkDescriptorSetLayout getDescriptorSetLayout(unsigned id) const
    {
        if (id == GVDFP_SOLID_SSR)
        {
            return m_solid_ssr ? getOutputDescriptorSetLayout() :
                VK_NULL_HANDLE;
        }
        // The layout exists without the displace color attachment (it samples
        // the image of this FBO for the solid ssr), but there is no copy pass
        if (id == GVDFP_DISPLACE_COLOR &&
            !getAttachment<GVDFT_DISPLACE_COLOR>())
            return VK_NULL_HANDLE;
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
    virtual std::vector<VkDescriptorSetLayout>
           getAllDescriptorSetLayouts(VkDescriptorSetLayout data_layout,
                                                            unsigned id) const;
    // ------------------------------------------------------------------------
    // pipeline_type is a GEVulkanPipelineType, the render pass which a
    // pipeline of this type is used in
    VkRenderPass getRenderPassForPipeline(unsigned pipeline_type) const;
};   // GEVulkanDeferredFBOSplit

}

#endif
