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
// be ready). A pass which resolves rough reflections (a stochastic trace with
// several rays, or a mip chain of the output which is sampled by the
// roughness) would go between the trace and the lighting pass.
class GEVulkanSolidReflection : public GEVulkanPostProcessing
{
private:
    // The output (R11G11B10F, RGBA16F if it can't be a storage image), always
    // in the general layout
    GEVulkanAttachmentTexture* m_reflection;

    // The glossy pixels of the viewport, with their count and the indirect
    // dispatch of the trace, see utils/solid_reflection.glsl
    VkBuffer m_list;

    VmaAllocation m_list_memory;

    // The output, the list, the normal and the depth of the g-buffer (only for
    // the classify, the traces have the g-buffer in GVDFP_HDR)
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    VkDescriptorSet m_set;

    VkPipelineLayout m_classify_layout;

    VkPipeline m_classify_pipeline;

    // The draw calls whose viewport was traced in this frame
    std::vector<GEVulkanDrawCall*> m_traced;

    // The FBO has the screen space or the ray traced trace (decided when it's
    // created)
    const bool m_has_ssr, m_has_rt;

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
                            bool rt);
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
};   // GEVulkanSolidReflection

}

#endif
