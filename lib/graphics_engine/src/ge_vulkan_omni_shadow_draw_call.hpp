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
    // Each layer maps to exactly one array subpass index.
    virtual uint32_t getSubpassForPipelineCreation(GEVulkanDriver* vk,
                                                   GEVulkanPipelineType type)
    {
        return (uint32_t)m_layer_id;
    }

public:
    // ------------------------------------------------------------------------
    GEVulkanOmniShadowDrawCall(GEVulkanShadowFBO* sfbo, unsigned layer_id)
    : GEVulkanShadowDrawCall(sfbo, GVSCC_NEAR),
      m_layer_id(layer_id)
    {
    }

};   // GEVulkanOmniShadowDrawCall

}   // namespace GE

#endif   // HEADER_GE_VULKAN_OMNI_SHADOW_DRAW_CALL_HPP
