#ifndef HEADER_GE_VULKAN_SPECULAR_GI_HPP
#define HEADER_GE_VULKAN_SPECULAR_GI_HPP

#include "ge_vulkan_prev_camera.hpp"
#include "ge_vulkan_solid_reflection.hpp"

#include <vector>

namespace GE
{
// The specular GI of FidelityFX SSSR, used instead of
// GEVulkanSolidReflection with GEConfig::m_specular_gi (decided when
// it's created), with the same traces (see utils/specular_gi.glsl):
// the classify and the trace are at half size, a few samples of the lobe of
// the specular BRDF per texel (specular_gi_classify.comp,
// specular_gi_ssr.comp and specular_gi_rt.comp), then
//
//   mipmap  : specular_gi_mipmap.comp, the mip chain of the trace
//             (the average of the traced texels)
//   resolve  : specular_gi_resolve.comp, the spatial filter of the
//              trace (the prefilter of FidelityFX DNSR) into the output
//   temporal : with the frame cache (GEConfig::m_frame_cache, it can be
//              changed at any time), specular_gi_temporal.comp:
//              the resolve writes another target, which is mixed with the
//              history of the last frames into the output (the
//              reprojection and the temporal resolve of FidelityFX DNSR),
//              and the rays of a texel change with the frame
//
// and the output is the radiance premultiplied by its coverage (RGBA16F, half
// size), which the lighting pass upsamples and mixes with the specular IBL.
class GEVulkanSpecularGI : public GEVulkanSolidReflection
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

    // The length of the rays of every texel (R32F, half size), written by the
    // trace for the temporal filter
    GEVulkanAttachmentTexture* m_ray_length;

    // The output of the resolve with the temporal filter (RGBA16F, half size)
    GEVulkanAttachmentTexture* m_current;

    // The history of the temporal filter: the radiance (RGBA16F) and the
    // surface of every texel (RG32UI, see specular_gi_temporal.comp,
    // 0 where it can't be reused). One is read and the other one written in
    // a frame, they swap every frame which writes one
    std::array<GEVulkanAttachmentTexture*, 2> m_history;

    std::array<GEVulkanAttachmentTexture*, 2> m_history_surface;

    // GEVulkanPrevCamera of every viewport of the frame (up to
    // g_specular_gi_max_viewports), updated before its temporal filter
    VkBuffer m_prev_camera_buffer;

    VmaAllocation m_prev_camera_memory;

    // m_set with the output of the resolve in m_current (for the temporal
    // filter)
    VkDescriptorSet m_resolve_set;

    // The set of the temporal filter, set i reads the history i and writes the
    // other one
    VkDescriptorSetLayout m_temporal_layout;

    VkDescriptorPool m_temporal_pool;

    std::array<VkDescriptorSet, 2> m_temporal_sets;

    // The mipmap, the resolve and the temporal filter
    VkPipelineLayout m_mipmap_layout;

    VkPipeline m_mipmap_pipeline;

    VkPipelineLayout m_resolve_layout;

    VkPipeline m_resolve_pipeline;

    VkPipelineLayout m_temporal_pipeline_layout;

    VkPipeline m_temporal_pipeline;

    // A viewport which was filtered, the next frame reads its history if it
    // has the same camera with the same rect
    struct ViewportHistory
    {
        GEVulkanCamera* m_camera;
        std::array<int32_t, 4> m_rect;
        GEVulkanPrevCamera m_prev_camera;
    };

    // The viewports of the last frame which wrote the history, and the ones
    // of this frame (empty if the frame has no temporal filter)
    std::vector<ViewportHistory> m_prev_viewports;

    std::vector<ViewportHistory> m_viewports;

    // The frame of the temporal filter (m_frame of the list, never 0)
    uint32_t m_frame;

    // The history which this frame reads (the index of m_temporal_sets)
    unsigned m_history_index;

    // This frame has the temporal filter (decided by its first viewport)
    bool m_temporal;

    // The viewport of the frame which is traced (its GEVulkanPrevCamera in the
    // buffer), and its history can be read
    unsigned m_viewport_index;

    bool m_viewport_history;
    // ------------------------------------------------------------------------
    // Destroys what isn't in GEVulkanSolidReflection
    void destroy();
    // ------------------------------------------------------------------------
    // The set (m_layout and m_set): the level 0 of the trace instead of the
    // output, the list, the normal, the depth, the levels of the trace, the
    // trace (sampled) and the output, see SpecularGIBinding
    void createDescriptor(GEVulkanAttachmentTexture* depth,
                          GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    // The trace target, half of the size of the FBO, with its views
    void createTraceImage(const irr::core::dimension2d<irr::u32>& size);
    // ------------------------------------------------------------------------
    // The targets of the temporal filter, the history and m_temporal_sets
    void createTemporalDescriptors(
                                 const irr::core::dimension2d<irr::u32>& size);
    // ------------------------------------------------------------------------
    void createPipelines(bool surfel_gi);
    VkDeviceSize getTraceDispatchOffset(bool rt) const override;
    // ------------------------------------------------------------------------
    // Clears the levels of an image to 0 (it's in the general layout)
    void clearImage(VkCommandBuffer cmd, VkImage image, unsigned levels);
protected:
    // ------------------------------------------------------------------------
    virtual const char* getTraceShader(bool rt) const
    {
        return rt ? "specular_gi_rt.comp" :
            "specular_gi_ssr.comp";
    }
    // ------------------------------------------------------------------------
    // Also clears the trace (and the history which is written) once per
    // frame, and updates the camera of the last frame of the viewport
    virtual void resetList(VkCommandBuffer cmd, bool first,
                           const Viewport& vp, GEVulkanCamera* cam);
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
    // The mipmap, the resolve and the temporal filter of the viewport
    virtual void finishTrace(VkCommandBuffer cmd, const Viewport& vp,
                             GEVulkanCamera* cam);
public:
    // ------------------------------------------------------------------------
    GEVulkanSpecularGI(GEVulkanDriver* vk,
                                 const irr::core::dimension2d<irr::u32>& size,
                                 GEVulkanAttachmentTexture* depth,
                                 GEVulkanAttachmentTexture* normal, bool ssr,
                                 bool rt, bool surfel_gi);
    // ------------------------------------------------------------------------
    ~GEVulkanSpecularGI();
    // ------------------------------------------------------------------------
    virtual bool isSpecularGI() const                          { return true; }
    // ------------------------------------------------------------------------
    // The screen space trace also writes the length of the rays for the
    // temporal filter
    virtual std::vector<std::pair<uint32_t, VkDescriptorType> >
        getTraceOutputBindings() const;
};   // GEVulkanSpecularGI

}

#endif
