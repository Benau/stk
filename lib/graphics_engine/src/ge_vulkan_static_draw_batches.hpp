#ifndef HEADER_GE_VULKAN_STATIC_DRAW_BATCHES_HPP
#define HEADER_GE_VULKAN_STATIC_DRAW_BATCHES_HPP

#include "ge_vma.hpp"
#include "ge_vulkan_static_draw_source.hpp"
#include "vulkan_wrapper.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace GE
{
class GEVulkanCamera;
class GEVulkanDrawCall;
enum GEVulkanPipelineType : unsigned;

// ============================================================================
// The static draws of one draw call (view): the culled commands and objects
// which the GPU generates from the shared GEVulkanStaticDrawSource each frame
// before the render pass.
//
// The destination is only written by the compute pass of this view, and the
// render thread always submits to the same queue, so the barriers of
// dispatch cover the previous frame which may still read it.
//
// This is the batches of a camera, the shadow views derive from it (see
// GEVulkanStaticShadowBatches): they use another culling shader and the shader
// names of the shadow draw calls, and they don't count the polygons
class GEVulkanStaticDrawBatches
{
public:
    typedef GEVulkanStaticDrawSource::Range Range;
    typedef GEVulkanStaticDrawSource::Variant Variant;

private:
    std::shared_ptr<GEVulkanStaticDrawSource> m_source;

    // The commands, then the objects
    VkBuffer m_buffer;

    VmaAllocation m_memory;

    VkDeviceSize m_objects_offset;

    // The polygons of the visible objects, one value for each workgroup of the
    // culling, in host cached memory with a slot for each buffer of the camera
    // uniform buffer. Every value is overwritten by each dispatch and the CPU
    // only reads it (after the fence of the frame which used the slot), so no
    // copy, clear or barrier is needed
    VkBuffer m_stats_buffer;

    VmaAllocation m_stats_memory;

    const uint32_t* m_stats_mapped;

    VkDeviceSize m_stats_stride;

    unsigned m_group_count;

    unsigned m_poly_count;

    VkDescriptorPool m_pool;

    // For each buffer of the camera uniform buffer (see getCurrentBufferIdx)
    std::vector<VkDescriptorSet> m_compute_sets, m_draw_sets;

    std::vector<std::weak_ptr<bool> > m_camera_observers;

    // ------------------------------------------------------------------------
    bool createBuffer();
    // ------------------------------------------------------------------------
    bool createStats();
    // ------------------------------------------------------------------------
    void readPolyCount(unsigned idx);
    // ------------------------------------------------------------------------
    bool createDescriptors(VkDescriptorSetLayout draw_layout);
    // ------------------------------------------------------------------------
    void updateDescriptors(GEVulkanCamera* cam, bool lighting);

protected:
    // ------------------------------------------------------------------------
    // The statistics (with_stats) are created here, so a derived class which
    // doesn't need them can't have it as a virtual function
    GEVulkanStaticDrawBatches(
        const std::shared_ptr<GEVulkanStaticDrawSource>& source,
        VkDescriptorSetLayout draw_layout, bool with_stats);
    // ------------------------------------------------------------------------
    // The kind of view, which selects the culling shader and the shader names
    virtual Variant getVariant() const
    {
        return GEVulkanStaticDrawSource::SV_VIEW;
    }
    // ------------------------------------------------------------------------
    // Not zero if the view skips the small objects, see isSmallForFarShadow
    virtual float getSkipSmall() const                         { return 0.0f; }
    // ------------------------------------------------------------------------
    // The ranges of the commands which use the shader of the draw call
    bool findRanges(const std::string& shader, const Range*& ranges,
                    unsigned& count) const;

public:
    // ------------------------------------------------------------------------
    GEVulkanStaticDrawBatches(
        const std::shared_ptr<GEVulkanStaticDrawSource>& source,
        VkDescriptorSetLayout draw_layout);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanStaticDrawBatches();
    // ------------------------------------------------------------------------
    bool isUsable() const;
    // ------------------------------------------------------------------------
    bool matches(const std::shared_ptr<GEVulkanStaticDrawSource>& source) const
                                                 { return m_source == source; }
    // ------------------------------------------------------------------------
    const GEVulkanStaticDrawSource& getSource() const     { return *m_source; }
    // ------------------------------------------------------------------------
    // Whether the entry (of the static scene) is registered to the draw call
    bool onCPU(unsigned entry) const;
    // ------------------------------------------------------------------------
    // The polygons of the static draws of this view, from the last frame whose
    // slot was finished by the GPU when this frame was recorded (a few frames
    // old)
    unsigned getPolyCount() const                      { return m_poly_count; }
    // ------------------------------------------------------------------------
    // Calls the function with each shader name of the draw call that has
    // commands
    template<typename F>
    void forEachShader(F function) const
    {
        if (getVariant() == GEVulkanStaticDrawSource::SV_VIEW)
        {
            for (const auto& range : m_source->getRanges())
                function(range.first);
        }
        else
        {
            for (const auto& ranges : m_source->getShadowRanges())
                function(ranges.first);
        }
    }
    // ------------------------------------------------------------------------
    void dispatch(VkCommandBuffer cmd, GEVulkanDrawCall& dc,
                  GEVulkanCamera* cam);
    // ------------------------------------------------------------------------
    void render(VkCommandBuffer cmd, GEVulkanDrawCall& dc,
                const std::string& shader, GEVulkanPipelineType pt,
                VkPipelineLayout layout, std::vector<uint32_t> offsets);
};   // GEVulkanStaticDrawBatches

// ============================================================================
// The static draws of a shadow view of a directional light (a cascade). They
// are culled by the frustum of the light (static_shadow_draws.comp), the far
// cascade skips the same small objects as GEVulkanShadowDrawCall::skip
class GEVulkanStaticShadowBatches : public GEVulkanStaticDrawBatches
{
private:
    const bool m_far_cascade;

protected:
    // ------------------------------------------------------------------------
    virtual Variant getVariant() const
    {
        return GEVulkanStaticDrawSource::SV_SHADOW;
    }
    // ------------------------------------------------------------------------
    virtual float getSkipSmall() const
                                        { return m_far_cascade ? 1.0f : 0.0f; }

public:
    // ------------------------------------------------------------------------
    GEVulkanStaticShadowBatches(
        const std::shared_ptr<GEVulkanStaticDrawSource>& source,
        VkDescriptorSetLayout draw_layout, bool far_cascade)
        : GEVulkanStaticDrawBatches(source, draw_layout, false),
          m_far_cascade(far_cascade)
    {
    }
};   // GEVulkanStaticShadowBatches

// ============================================================================
// The static draws of the omni shadow draw call of a bucket of lights, which
// all the layers of the bucket share (static_omni_draws.comp)
class GEVulkanStaticOmniShadowBatches : public GEVulkanStaticShadowBatches
{
protected:
    // ------------------------------------------------------------------------
    virtual Variant getVariant() const
    {
        return GEVulkanStaticDrawSource::SV_OMNI;
    }
    // ------------------------------------------------------------------------
    virtual float getSkipSmall() const                         { return 0.0f; }

public:
    // ------------------------------------------------------------------------
    GEVulkanStaticOmniShadowBatches(
        const std::shared_ptr<GEVulkanStaticDrawSource>& source,
        VkDescriptorSetLayout draw_layout)
        : GEVulkanStaticShadowBatches(source, draw_layout, false)
    {
    }
};   // GEVulkanStaticOmniShadowBatches

}

#endif
