#ifndef HEADER_GE_VULKAN_STATIC_DRAW_BATCHES_HPP
#define HEADER_GE_VULKAN_STATIC_DRAW_BATCHES_HPP

#include "ge_vma.hpp"
#include "vulkan_wrapper.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace GE
{
class GEVulkanCamera;
class GEVulkanDrawCall;
class GEVulkanStaticDrawSource;
enum GEVulkanPipelineType : unsigned;

// ============================================================================
// The static draws of one draw call (view): the culled commands and objects
// which the GPU generates from the shared GEVulkanStaticDrawSource each frame
// before the render pass.
//
// The destination is only written by the compute pass of this view, and the
// render thread always submits to the same queue, so the barriers of
// dispatch cover the previous frame which may still read it.
class GEVulkanStaticDrawBatches
{
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
public:
    // ------------------------------------------------------------------------
    GEVulkanStaticDrawBatches(
        const std::shared_ptr<GEVulkanStaticDrawSource>& source,
        VkDescriptorSetLayout draw_layout);
    // ------------------------------------------------------------------------
    ~GEVulkanStaticDrawBatches();
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
    void dispatch(VkCommandBuffer cmd, GEVulkanDrawCall& dc,
                  GEVulkanCamera* cam);
    // ------------------------------------------------------------------------
    void render(VkCommandBuffer cmd, GEVulkanDrawCall& dc,
                const std::string& shader, GEVulkanPipelineType pt,
                VkPipelineLayout layout, std::vector<uint32_t> offsets);
};   // GEVulkanStaticDrawBatches

}

#endif
