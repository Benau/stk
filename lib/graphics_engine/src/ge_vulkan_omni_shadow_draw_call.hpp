#ifndef HEADER_GE_VULKAN_OMNI_SHADOW_DRAW_CALL_HPP
#define HEADER_GE_VULKAN_OMNI_SHADOW_DRAW_CALL_HPP

#include "ge_vulkan_shadow_draw_call.hpp"
#include "ge_main.hpp"

namespace GE
{
// ---------------------------------------------------------------------------
class GEVulkanOmniShadowDrawCall : public GEVulkanShadowDrawCall
{
private:
    const unsigned m_layer_id;
    // ------------------------------------------------------------------------
    virtual bool skip(irr::scene::ISceneNode* node) const     { return false; }
    // ------------------------------------------------------------------------
    virtual bool useDepthClamp() const                        { return false; }
    // ------------------------------------------------------------------------
    virtual bool useDPShadow() const                           { return true; }
    // ------------------------------------------------------------------------
    // The atlas is a single array layer, every slot renders into a tile of it
    virtual uint32_t getSubpassForPipelineCreation(GEVulkanDriver* vk,
                                                   GEVulkanPipelineType type)
    {
        return 0;
    }

public:
    // ------------------------------------------------------------------------
    GEVulkanOmniShadowDrawCall(GEVulkanShadowFBO* sfbo, unsigned layer_id)
    : GEVulkanShadowDrawCall(sfbo, GVSCC_NEAR),
      m_layer_id(layer_id)
    {
    }
    // ------------------------------------------------------------------------
    virtual void prepareShadow(unsigned layer)
    {
        GEVulkanShadowDrawCall::prepareShadow(layer);
        setRenderState(true);
    }

};   // GEVulkanOmniShadowDrawCall

}   // namespace GE

#endif   // HEADER_GE_VULKAN_OMNI_SHADOW_DRAW_CALL_HPP
