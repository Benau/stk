#ifndef HEADER_GE_VULKAN_STOCHASTIC_REFLECTION_HPP
#define HEADER_GE_VULKAN_STOCHASTIC_REFLECTION_HPP

#include "ge_vulkan_solid_reflection.hpp"

namespace GE
{
// The stochastic reflection of FidelityFX SSSR, used instead of
// GEVulkanSolidReflection with GEConfig::m_stochastic_reflection (decided when
// it's created), with the same traces (see utils/stochastic_reflection.glsl):
// the classify and the trace are at half size, a few samples of the lobe of
// the specular BRDF per texel (stochastic_reflection_classify.comp,
// stochastic_reflection_ssr.comp and stochastic_reflection_rt.comp), then
//
//   mipmap  : stochastic_reflection_mipmap.comp, the mip chain of the trace
//             (the average of the traced texels)
//   resolve : stochastic_reflection_resolve.comp, the spatial filter of the
//             trace (the prefilter of FidelityFX DNSR) into the output
//
// and the output is the radiance premultiplied by its coverage (RGBA16F, half
// size), which the lighting pass upsamples and mixes with the specular IBL.
class GEVulkanStochasticReflection : public GEVulkanSolidReflection
{
private:
    // The trace (RGBA16F, half size), with its mip chain (m_trace_levels
    // levels), always in the general layout
    VkImage m_trace_image;

    VmaAllocation m_trace_memory;

    // All the levels (sampled by the resolve) and each level (written by the
    // trace and the mipmap, the unused ones are the last level)
    VkImageView m_trace_view;

    std::array<VkImageView, 4> m_trace_level_views;

    unsigned m_trace_levels;

    // The mipmap and the resolve
    VkPipelineLayout m_mipmap_layout;

    VkPipeline m_mipmap_pipeline;

    VkPipelineLayout m_resolve_layout;

    VkPipeline m_resolve_pipeline;

    // ------------------------------------------------------------------------
    // Destroys what isn't in GEVulkanSolidReflection
    void destroy();
    // ------------------------------------------------------------------------
    // The set (m_layout and m_set): the level 0 of the trace instead of the
    // output, the list, the normal, the depth, the levels of the trace, the
    // trace (sampled) and the output, see StochasticBinding
    void createDescriptor(GEVulkanAttachmentTexture* depth,
                          GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    // The trace target, half of the size of the FBO, with its views
    void createTraceImage(const irr::core::dimension2d<irr::u32>& size);
    // ------------------------------------------------------------------------
    void createPipelines();
    // ------------------------------------------------------------------------
    // Clears the levels of an image to 0 (it's in the general layout)
    void clearImage(VkCommandBuffer cmd, VkImage image, unsigned levels);
protected:
    // ------------------------------------------------------------------------
    virtual const char* getTraceShader(bool rt) const
    {
        return rt ? "stochastic_reflection_rt.comp" :
            "stochastic_reflection_ssr.comp";
    }
    // ------------------------------------------------------------------------
    // Also clears the trace once per frame
    virtual void resetList(VkCommandBuffer cmd, bool first);
    // ------------------------------------------------------------------------
    // The texels of the half size target whose pixel can be in the rect (the
    // texel t is the pixel 2t)
    virtual std::array<uint32_t, 2>
                    getClassifyGroups(const std::array<int32_t, 4>& rect) const
    {
        return {{ groups8((rect[2] + 1) / 2 - rect[0] / 2),
            groups8((rect[3] + 1) / 2 - rect[1] / 2) }};
    }
    // ------------------------------------------------------------------------
    // The mipmap and the resolve of the viewport
    virtual void finishTrace(VkCommandBuffer cmd, GEVulkanDrawCall* dc,
                             const std::array<int32_t, 4>& rect);
public:
    // ------------------------------------------------------------------------
    GEVulkanStochasticReflection(GEVulkanDriver* vk,
                                 const irr::core::dimension2d<irr::u32>& size,
                                 GEVulkanAttachmentTexture* depth,
                                 GEVulkanAttachmentTexture* normal, bool ssr,
                                 bool rt);
    // ------------------------------------------------------------------------
    ~GEVulkanStochasticReflection();
    // ------------------------------------------------------------------------
    virtual bool isStochastic() const                          { return true; }
};   // GEVulkanStochasticReflection

}

#endif
