#ifndef HEADER_GE_VULKAN_SOLID_REFLECTION_HPP
#define HEADER_GE_VULKAN_SOLID_REFLECTION_HPP

#include "ge_vulkan_pipeline_constants.hpp"
#include "ge_vulkan_post_processing.hpp"

#include "vk_mem_alloc.h"

#include <array>
#include <memory>

namespace GE
{
class GEVulkanAttachmentTexture;

// The reflection of the glossy solid materials, which the lighting pass adds
// to the lit color (deferred_pbr.frag, u_solid_reflection_image), see
// utils/solid_reflection.glsl. It's owned by GEVulkanDeferredFBOSplit and
// rendered between the g-buffer and the lighting pass (compute shaders only),
// for every viewport:
//
//   classify : solid_reflection_classify.comp, the glossy pixels are added
//              to a list
//   trace    : only the pixels of the list (an indirect dispatch), traced in
//              the ray traced scene (solid_reflection_rt.comp) or over the
//              depth / HiZ depth with the output of the previous frame
//              (solid_reflection_ssr.comp), with the camera, the environment
//              and the HiZ depth of the draw call of the viewport
//
// A fullscreen draw took as long as its slowest pixels: a group of threads
// with one glossy pixel which traces a ray waited for it, and the pixels of
// the fragment shader are written in order, so the ones which were discarded
// waited for the ray traced ones too. The list packs the glossy pixels in full
// groups.
//
// The output is the specular radiance which the lighting pass uses instead of
// the specular IBL, of the viewports which were traced in this frame
// (isTraced, the others keep the specular IBL: the screen space trace has no
// output of a previous frame in the first frame, the ray traced scene may not
// be ready). Rough reflections are a single mirror ray mixed with the
// specular IBL by the gloss.
//
// With GEConfig::m_stochastic_reflection (decided when it's created, see
// isStochastic()) it's the stochastic reflection of FidelityFX SSSR instead,
// with the same traces (see utils/stochastic_reflection.glsl): the classify
// and the trace are at half size, a few samples of the lobe of the specular
// BRDF per texel (stochastic_reflection_classify.comp,
// stochastic_reflection_ssr.comp and stochastic_reflection_rt.comp), then
//
//   mipmap  : stochastic_reflection_mipmap.comp, the mip chain of the trace
//             (the average of the traced texels)
//   resolve : stochastic_reflection_resolve.comp, the spatial filter of the
//             trace (the prefilter of FidelityFX DNSR) into the output
//
// and the output is the radiance premultiplied by its coverage (RGBA16F, half
// size), which the lighting pass upsamples and mixes with the specular IBL.
// Nothing else changes without it: the same shaders, output and commands.
class GEVulkanSolidReflection : public GEVulkanPostProcessing
{
private:
    // The output (R11G11B10F, RGBA16F if it can't be a storage image), always
    // in the general layout. The resolved stochastic reflection (RGBA16F, half
    // size) if m_stochastic
    GEVulkanAttachmentTexture* m_reflection;

    // The trace of the stochastic reflection (RGBA16F, half size), with its
    // mip chain (m_trace_levels levels), always in the general layout
    VkImage m_trace_image;

    VmaAllocation m_trace_memory;

    // All the levels (sampled by the resolve) and each level (written by the
    // trace and the mipmap, the unused ones are the last level)
    VkImageView m_trace_view;

    std::array<VkImageView, 4> m_trace_level_views;

    unsigned m_trace_levels;

    // The glossy pixels of the viewport, with their count and the indirect
    // dispatch of the trace, see utils/solid_reflection.glsl
    VkBuffer m_list;

    VmaAllocation m_list_memory;

    // The output, the list, the normal and the depth of the g-buffer (only for
    // the classify, the traces have the g-buffer in GVDFP_HDR). If
    // m_stochastic: the level 0 of the trace instead of the output, the list,
    // the normal, the depth, the levels of the trace, the trace (sampled) and
    // the output, see utils/stochastic_reflection.glsl
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    VkDescriptorSet m_set;

    VkPipelineLayout m_classify_layout;

    VkPipeline m_classify_pipeline;

    // The mipmap and the resolve of the stochastic reflection
    VkPipelineLayout m_mipmap_layout;

    VkPipeline m_mipmap_pipeline;

    VkPipelineLayout m_resolve_layout;

    VkPipeline m_resolve_pipeline;

    // The draw calls whose viewport was traced in this frame
    std::vector<GEVulkanDrawCall*> m_traced;

    // The FBO has the screen space or the ray traced trace (decided when it's
    // created)
    const bool m_has_ssr, m_has_rt;

    // The reflection is the stochastic one (GEConfig::m_stochastic_reflection
    // when it's created)
    const bool m_stochastic;

    // The pipelines of the trace, the screen space one and the one of the ray
    // traced scene (NULL if the FBO has none of them)
    enum TracePipeline : unsigned
    {
        TP_SSR = 0,
        TP_RT,
        TP_COUNT
    };
    std::array<VkPipelineLayout, TP_COUNT> m_trace_layouts;

    std::array<VkPipeline, TP_COUNT> m_trace_pipelines;

    // The layout of the camera and the lights, it's the same as the one of
    // every draw call (GEVulkanDrawCall::createDataLayout), whose set is bound
    VkDescriptorSetLayout m_data_layout;

    // The layout of the mesh textures of the ray traced trace, it's created
    // again when its pool grows, so are the pipelines
    std::weak_ptr<VkDescriptorSetLayout> m_mesh_texture_layout;

    // The part of the specialization constants of the draw call which the
    // pipelines were created with (the first viewport, whose environment
    // the trace of every viewport has, the same as GEVulkanSurfelGI), they
    // are created again if it changes (a track with or without a skybox)
    GEVulkanPipelineConstants::Options m_options;

    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    void createDescriptor(GEVulkanAttachmentTexture* depth,
                          GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    void createClassifyPipeline();
    // ------------------------------------------------------------------------
    // The set of the stochastic reflection (m_layout and m_set, see
    // StochasticBinding)
    void createStochasticDescriptor(GEVulkanAttachmentTexture* depth,
                                    GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    // The trace target of the stochastic reflection, half of the size of the
    // FBO, with its views
    void createTraceImage(const irr::core::dimension2d<irr::u32>& size);
    // ------------------------------------------------------------------------
    void createStochasticPipelines();
    // ------------------------------------------------------------------------
    // Clears the levels of an image of the stochastic reflection to 0 (it's
    // in the general layout)
    void clearStochasticImages(VkCommandBuffer cmd, VkImage image,
                               unsigned levels);
    // ------------------------------------------------------------------------
    // The mipmap and the resolve of the stochastic reflection of the viewport
    // (after its trace)
    void dispatchStochasticResolve(VkCommandBuffer cmd, GEVulkanDrawCall* dc,
                                   const int* rect);
    // ------------------------------------------------------------------------
    void destroyTracePipelines();
    // ------------------------------------------------------------------------
    // Creates the pipelines of the trace if there are none, or if they were
    // created with other options, false if they can't be created
    bool prepareTracePipelines(GEVulkanDrawCall* dc);
    // ------------------------------------------------------------------------
    void createTracePipelines(GEVulkanDrawCall* dc);
    // ------------------------------------------------------------------------
    // Whether the viewport of the draw call can be traced (it has visible
    // nodes, and the ray traced scene is ready or it has the sets of the
    // screen space trace), its glossy pixels get an output of 0 otherwise
    bool canTrace(GEVulkanDrawCall* dc, bool rt) const;
    // ------------------------------------------------------------------------
    // The first set of the screen space trace: the HiZ depth of the viewport
    // of the draw call and the output of the previous frame, or the set of
    // the FBO with the depth only (the pipeline has no iterations of HiZ then)
    const VkDescriptorSet* getSSRDescriptorSet(GEVulkanDrawCall* dc) const;
    // ------------------------------------------------------------------------
    // The indirect dispatch of the trace of the list of the viewport
    void dispatchTrace(VkCommandBuffer cmd, GEVulkanDrawCall* dc, bool rt);
public:
    // ------------------------------------------------------------------------
    GEVulkanSolidReflection(GEVulkanDriver* vk,
                            const irr::core::dimension2d<irr::u32>& size,
                            GEVulkanAttachmentTexture* depth,
                            GEVulkanAttachmentTexture* normal, bool ssr,
                            bool rt, bool stochastic);
    // ------------------------------------------------------------------------
    ~GEVulkanSolidReflection();
    // ------------------------------------------------------------------------
    // The reflection of all the viewports, the ones of the ray traced scene
    // if rt or the screen space ones if ssr (the output of the previous frame
    // is there), nothing if neither. Only the glossy pixels of the viewports
    // which can be traced (canTrace, see isTraced) are written. The depth
    // and the normal of the g-buffer have to be written. The output is read by
    // the fragment shader of the lighting pass after it
    bool render(VkCommandBuffer cmd, bool rt, bool ssr,
                const std::vector<std::pair<GEVulkanDrawCall*,
                GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p)
                                       { return render(cmd, false, false, p); }
    // ------------------------------------------------------------------------
    // The output has the reflection of the viewport of the draw call in this
    // frame, so the lighting pass uses it (only after render())
    bool isTraced(GEVulkanDrawCall* dc) const;
    // ------------------------------------------------------------------------
    VkImageView getImageView() const;
    // ------------------------------------------------------------------------
    bool isStochastic() const                          { return m_stochastic; }
};   // GEVulkanSolidReflection

}

#endif
