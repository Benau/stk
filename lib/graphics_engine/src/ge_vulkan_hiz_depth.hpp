#ifndef HEADER_GE_VULKAN_HIZ_DEPTH_HPP
#define HEADER_GE_VULKAN_HIZ_DEPTH_HPP

#include "vulkan_wrapper.h"

#include "rect.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GE
{
class GEVulkanCameraSceneNode;
class GEVulkanDeferredFBO;
class GEVulkanDriver;
class GEVulkanTexture;

class GEVulkanHiZDepth
{
private:
    GEVulkanDriver* m_vk;

    GEVulkanTexture* m_hiz_depth;

    VkDescriptorSetLayout m_descriptor_layout;

    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;

    VkDescriptorPool m_descriptor_pool, m_rendering_descriptor_pool;

    VkDescriptorSet m_rendering_descriptor_set;

    std::vector<VkDescriptorSet> m_descriptor_sets;

    std::vector<VkImageView> m_hiz_views;

    irr::core::recti m_hiz_size;

    std::weak_ptr<bool> m_dfbo_observer;

    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    void init();
    // ------------------------------------------------------------------------
    void loadRenderingDescriptor();
public:
    // ------------------------------------------------------------------------
    GEVulkanHiZDepth(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    ~GEVulkanHiZDepth();
    // ------------------------------------------------------------------------
    void prepare(GEVulkanCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void generate(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    const VkDescriptorSet* getRenderingDescriptorSet() const
     { return m_dfbo_observer.expired() ? NULL : &m_rendering_descriptor_set; }
};

}

#endif
