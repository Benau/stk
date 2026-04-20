#include "ge_vulkan_shadow_draw_call.hpp"

#include "ge_culling_tool.hpp"
#include "ge_material_manager.hpp"
#include "ge_render_info.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_shadow_fbo.hpp"
#include "ge_vulkan_skybox_renderer.hpp"

#include "mini_glm.hpp"
#include "ISceneNode.h"

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanShadowDrawCall::GEVulkanShadowDrawCall(GEVulkanShadowFBO* sfbo,
                                               GEVulkanShadowCameraCascade cc)
                      : GEVulkanDrawCall(), m_sfbo(sfbo), m_cascade(cc)
{
}   // GEVulkanShadowDrawCall

// ----------------------------------------------------------------------------
void GEVulkanShadowDrawCall::prepareShadow(unsigned layer)
{
    reset();
    m_culling_tool->initShadow(m_sfbo, layer);
    if (m_data_layout == VK_NULL_HANDLE)
    {
        createVulkanData();
        initNonPBRFallbackMaterials();
    }
}   // prepareShadow

// ----------------------------------------------------------------------------
bool GEVulkanShadowDrawCall::skip(irr::scene::ISceneNode* node) const
{
    // Skip small objects for far shadows
    if (m_cascade == GVSCC_FAR && node->getBoundingBox().getArea() < 20.0f)
        return true;
    return false;
}   // skip

// ----------------------------------------------------------------------------
VkRenderPass GEVulkanShadowDrawCall::getRenderPassForPipelineCreation(
                                                            GEVulkanDriver* vk,
                                                     GEVulkanPipelineType type)
{
    return m_sfbo->getRTTRenderPass();
}   // getRenderPassForPipelineCreation

// ----------------------------------------------------------------------------
bool GEVulkanShadowDrawCall::ignoreMaterial(
                                          irr::video::E_MATERIAL_TYPE mt) const
{
    return GEMaterialManager::getMaterial(mt)->isTransparent();
}   // ignoreMaterial

}
