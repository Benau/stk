#ifndef HEADER_GE_VULKAN_SOLID_REFLECTION_HPP
#define HEADER_GE_VULKAN_SOLID_REFLECTION_HPP

#include "ge_vulkan_pipeline_constants.hpp"
#include "ge_vulkan_post_processing.hpp"

#include "vk_mem_alloc.h"

#include <array>
#include <memory>
#include <utility>
#include <vector>

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
// GEVulkanStochasticReflection is the stochastic reflection, with the same
// list and traces.
class GEVulkanSolidReflection : public GEVulkanPostProcessing
{
private:
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

    // The layout of the mesh textures of the ray traced trace, it's created
    // again when its pool grows, so are the pipelines
    std::weak_ptr<VkDescriptorSetLayout> m_mesh_texture_layout;

    // The part of the specialization constants of the draw call which the
    // pipelines were created with (the first viewport, whose environment
    // the trace of every viewport has, the same as GEVulkanSurfelGI), they
    // are created again if it changes (a track with or without a skybox)
    GEVulkanPipelineConstants::Options m_options;

    // The draw calls whose viewport was traced in this frame
    std::vector<GEVulkanDrawCall*> m_traced;

    // The FBO has the screen space or the ray traced trace (decided when it's
    // created)
    const bool m_has_ssr, m_has_rt;

    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    void createDescriptor(GEVulkanAttachmentTexture* depth,
                          GEVulkanAttachmentTexture* normal);
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
protected:
    // The output (R11G11B10F, RGBA16F if it can't be a storage image), always
    // in the general layout
    GEVulkanAttachmentTexture* m_reflection;

    // The glossy pixels of the viewport, with their count and the indirect
    // dispatch of the trace, see utils/solid_reflection.glsl
    VkBuffer m_list;

    VmaAllocation m_list_memory;

    // The output, the list, the normal and the depth of the g-buffer (only for
    // the classify, the traces have the g-buffer in GVDFP_HDR), the last set
    // of the traces
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    VkDescriptorSet m_set;

    VkPipelineLayout m_classify_layout;

    VkPipeline m_classify_pipeline;

    // The layout of the camera and the lights, it's the same as the one of
    // every draw call (GEVulkanDrawCall::createDataLayout), whose set is bound
    VkDescriptorSetLayout m_data_layout;

    // ------------------------------------------------------------------------
    // Only sets the members, for a derived class which creates everything
    GEVulkanSolidReflection(GEVulkanDriver* vk,
                            const irr::core::dimension2d<irr::u32>& size,
                            bool ssr, bool rt);
    // ------------------------------------------------------------------------
    // Groups of 8x8 threads which cover a size
    static uint32_t groups8(int32_t size);
    // ------------------------------------------------------------------------
    // The list of the glossy pixels (or texels): the header (uints) and the
    // pixels of a size, one viewport at a time uses it
    void createList(const irr::core::dimension2d<irr::u32>& size,
                    unsigned header);
    // ------------------------------------------------------------------------
    void createClassifyPipeline(const char* shader);
    // ------------------------------------------------------------------------
    // The shader of the trace of the ray traced scene or of the screen space
    // one
    virtual const char* getTraceShader(bool rt) const
    {
        return rt ? "solid_reflection_rt.comp" : "solid_reflection_ssr.comp";
    }
    // ------------------------------------------------------------------------
    // Empties the list before the classify of a viewport (first is the first
    // traced viewport of the frame), with the camera of the viewport. Every
    // command before which read the list is before a barrier of transfer
    // writes, and the transfers are before the compute shaders after it
    virtual void resetList(VkCommandBuffer cmd, bool first,
                           const Viewport& vp, GEVulkanCamera* cam);
    // ------------------------------------------------------------------------
    // The groups of the classify of the pixels of the rect of a viewport
    virtual std::array<uint32_t, 2>
                    getClassifyGroups(const std::array<int32_t, 4>& rect) const
    {
        // Groups of 128 threads, two tiles of 8x8 pixels side by side (see
        // solid_reflection_classify.comp)
        return {{ (groups8(rect[2] - rect[0]) + 1) / 2,
            groups8(rect[3] - rect[1]) }};
    }
    // ------------------------------------------------------------------------
    // After the trace of the viewport (its draw call and its rect in pixels of
    // the FBO) of the camera
    virtual void finishTrace(VkCommandBuffer cmd, const Viewport& vp,
                             GEVulkanCamera* cam)                            {}
public:
    // ------------------------------------------------------------------------
    GEVulkanSolidReflection(GEVulkanDriver* vk,
                            const irr::core::dimension2d<irr::u32>& size,
                            GEVulkanAttachmentTexture* depth,
                            GEVulkanAttachmentTexture* normal, bool ssr,
                            bool rt);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanSolidReflection();
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
    // The set whose binding 0 is the output of the trace (the trace target of
    // GEVulkanStochasticReflection) and 1 the list, GEVulkanDeferredFBOSplit
    // copies them to GVDFP_HDR for the screen space trace
    const VkDescriptorSet* getTraceOutputSet() const         { return &m_set; }
    // ------------------------------------------------------------------------
    // The bindings of getTraceOutputSet() which the screen space trace uses,
    // with their type, GVDFP_HDR has them in this order after its samplers
    virtual std::vector<std::pair<uint32_t, VkDescriptorType> >
        getTraceOutputBindings() const
    {
        return
        {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER }
        };
    }
    // ------------------------------------------------------------------------
    // The output is the one of GEVulkanStochasticReflection
    virtual bool isStochastic() const                         { return false; }
};   // GEVulkanSolidReflection

}

#endif
