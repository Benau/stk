#ifndef HEADER_GE_VULKAN_SHADOW_DRAW_CALL_HPP
#define HEADER_GE_VULKAN_SHADOW_DRAW_CALL_HPP

#include "ge_vulkan_draw_call.hpp"

#include <cassert>

namespace GE
{
enum GEVulkanShadowCameraCascade : unsigned;
class GEVulkanShadowFBO;

class GEVulkanShadowDrawCall : public GEVulkanDrawCall
{
protected:
    const GEVulkanShadowFBO* m_sfbo;

private:
    const GEVulkanShadowCameraCascade m_cascade;

    // ------------------------------------------------------------------------
    virtual bool isShadow() const                              { return true; }
    // ------------------------------------------------------------------------
    virtual bool skip(irr::scene::ISceneNode* node) const;
    // ------------------------------------------------------------------------
    virtual bool useDepthClamp() const                         { return true; }
    // ------------------------------------------------------------------------
    virtual VkRenderPass getRenderPassForPipelineCreation(GEVulkanDriver* vk,
                                                    GEVulkanPipelineType type);
    // ------------------------------------------------------------------------
    virtual uint32_t getSubpassForPipelineCreation(GEVulkanDriver* vk,
                                                   GEVulkanPipelineType type);
    // ------------------------------------------------------------------------
    virtual void prepare(GEVulkanCameraSceneNode* cam)       { assert(false); }
    // ------------------------------------------------------------------------
    virtual bool ignoreMaterial(irr::video::E_MATERIAL_TYPE mt) const;
    // ------------------------------------------------------------------------
    virtual void generateDynamicSPM(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    virtual const GEVulkanDrawCall* getMasterDrawCall() const;

public:
    // ------------------------------------------------------------------------
    GEVulkanShadowDrawCall(GEVulkanShadowFBO* sfbo,
                           GEVulkanShadowCameraCascade cc);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanShadowDrawCall() {}
    // ------------------------------------------------------------------------
    virtual void prepareShadow(unsigned layer);
    // ------------------------------------------------------------------------
    virtual bool doDepthOnlyRenderingFirst()                   { return true; }
    // ------------------------------------------------------------------------
    virtual const VkDescriptorSet* getEnvDescriptorSet(GEVulkanDriver* vk)
    {
        assert(false);
        return NULL;
    }
    // ------------------------------------------------------------------------
    virtual void setRenderState(bool state)                                  {}
    // ------------------------------------------------------------------------
    virtual bool getRenderState() const                        { return true; }

};   // GEVulkanShadowDrawCall

}   // namespace GE

#endif   // HEADER_GE_VULKAN_SHADOW_DRAW_CALL_HPP
