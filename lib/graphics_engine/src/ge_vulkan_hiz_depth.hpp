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

// Max depth mip chain (reverse-Z) of the depth buffer of one viewport (camera),
// used by the screen space reflection of the displace pass and the solid
// materials (solid_ssr.frag). It is owned by GEVulkanDeferredFBOSplit (one per
// viewport), generated right after the g-buffer pass, so the lifetime of it is
// the one of the FBO
class GEVulkanHiZDepth
{
private:
    GEVulkanDriver* m_vk;

    GEVulkanDeferredFBO* m_dfbo;

    GEVulkanTexture* m_hiz_depth;

    VkDescriptorSetLayout m_descriptor_layout;

    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;

    VkDescriptorPool m_descriptor_pool, m_rendering_descriptor_pool;

    VkDescriptorSet m_rendering_descriptor_set;

    std::vector<VkDescriptorSet> m_descriptor_sets;

    std::vector<VkImageView> m_hiz_views;

    irr::core::recti m_hiz_size;

    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    void init();
    // ------------------------------------------------------------------------
    void loadRenderingDescriptor();
public:
    // ------------------------------------------------------------------------
    // True if the HiZ screen space reflection is selected (the types of
    // GEScreenSpaceReflectionType from GSSRT_HIZ100 and the device can run
    // compute shaders), otherwise it uses GSSRT_FAST (the depth buffer only)
    static bool isEnabled();
    // ------------------------------------------------------------------------
    GEVulkanHiZDepth(GEVulkanDriver* vk, GEVulkanDeferredFBO* dfbo);
    // ------------------------------------------------------------------------
    ~GEVulkanHiZDepth();
    // ------------------------------------------------------------------------
    // Creates it for the viewport of the camera (again if the size changed)
    void prepare(GEVulkanCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    bool isReady() const                        { return m_hiz_depth != NULL; }
    // ------------------------------------------------------------------------
    // The depth buffer of the FBO needs to be written and in the read only
    // layout (after the g-buffer pass, with a barrier for compute shaders), it
    // is in shader read only layout after this
    void generate(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    // Same layout as GVDFP_DISPLACE_COLOR of the FBO: GVDFT_DISPLACE_COLOR
    // (the output of the tonemap pass, so this is the one of the previous
    // frame in the lighting pass), the depth (shadow sampler) and the HiZ
    // depth, NULL if prepare() didn't create it
    const VkDescriptorSet* getRenderingDescriptorSet() const
                   { return m_hiz_depth ? &m_rendering_descriptor_set : NULL; }
};

}

#endif
