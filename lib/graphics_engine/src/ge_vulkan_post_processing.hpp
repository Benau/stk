#ifndef HEADER_GE_VULKAN_POST_PROCESS_HPP
#define HEADER_GE_VULKAN_POST_PROCESS_HPP

#include "vulkan_wrapper.h"

#include "dimension2d.h"

#include <array>
#include <utility>
#include <vector>

namespace GE
{
class GEVulkanAttachmentTexture;
class GEVulkanCameraSceneNode;
class GEVulkanDrawCall;
class GEVulkanDriver;

// Common Vulkan helpers for effects rendered between the g-buffer and the
// lighting pass. The derived classes own their actual images/pipelines
class GEVulkanPostProcessing
{
protected:
    GEVulkanDriver* m_vk;

    // Size of the g-buffer/FBO
    irr::core::dimension2d<irr::u32> m_size;

    // Reference height used by effects whose blur width scales with the
    // viewport height
    static const float REFERENCE_HEIGHT;

    // ------------------------------------------------------------------------
    static irr::core::dimension2d<irr::u32> halfSize(
                                 const irr::core::dimension2d<irr::u32>& size);
    // ------------------------------------------------------------------------
    // Covers all pixels of a rect: floor of the start, ceil of the end.
    // x0, y0, x1, y1 where x1/y1 are exclusive
    static std::array<int32_t, 4> halfRect(const std::array<int32_t, 4>& rect);
    // ------------------------------------------------------------------------
    // Transition all images from UNDEFINED to GENERAL. These images are
    // intended to remain in GENERAL for their entire lifetime
    void initializeGeneralImages(GEVulkanAttachmentTexture** images,
                                 unsigned count) const;
    // ------------------------------------------------------------------------
    // Standard layout used by the compute blur passes:
    //
    // binding 0 = combined image sampler
    // binding 1 = storage image
    //
    // Both are compute-only
    void createImageDescriptorSetLayout(VkDescriptorSetLayout* layout) const;
    // ------------------------------------------------------------------------
    // Create a pool containing exactly set_count sets of the standard
    // two-descriptor layout
    void createImageDescriptorPool(unsigned set_count,
                                   VkDescriptorPool* pool) const;
    // ------------------------------------------------------------------------
    // Allocate set_count copies of layout into sets
    void allocateImageDescriptorSets(VkDescriptorPool pool,
                                     VkDescriptorSetLayout layout,
                                     unsigned set_count,
                                     VkDescriptorSet* sets) const;
    // ------------------------------------------------------------------------
    // Fill one standard descriptor set
    //
    // input_view/input_sampler/input_layout describe binding 0
    // output_view/output_layout describe binding 1
    void writeImageDescriptorSet(VkDescriptorSet set,
                                 VkImageView input_view,
                                 VkSampler input_sampler,
                                 VkImageLayout input_layout,
                                 VkImageView output_view,
                                 VkImageLayout output_layout) const;

public:
    // ------------------------------------------------------------------------
    GEVulkanPostProcessing(GEVulkanDriver* vk,
      const irr::core::dimension2d<irr::u32>& size) : m_vk(vk), m_size(size) {}
    // ------------------------------------------------------------------------
    virtual ~GEVulkanPostProcessing()                                        {}
    // ------------------------------------------------------------------------
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCameraSceneNode*> >& p) = 0;

};   // GEVulkanPostProcessing

}

#endif
