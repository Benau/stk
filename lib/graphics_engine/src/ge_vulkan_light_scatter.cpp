#include "ge_vulkan_light_scatter.hpp"

#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanLightScatter::GEVulkanLightScatter(GEVulkanDriver* vk,
                              const irr::core::dimension2d<irr::u32>& size,
                              GEVulkanAttachmentTexture* depth)
                    : GEVulkanLowResEffect(vk, size, depth,
                      { "GEVulkanLightScatter", "light_scatter.comp",
                      sizeof(int32_t) * 4 + sizeof(float) * 8,
                      1/*half size*/, 1.0f, 0.25f, 2.5f })
{
}   // GEVulkanLightScatter

// ----------------------------------------------------------------------------
bool GEVulkanLightScatter::hasEffect(const GEVulkanDrawCall* dc) const
{
    return dc->hasLightScatter();
}   // hasEffect

// ----------------------------------------------------------------------------
void GEVulkanLightScatter::pushConstants(VkCommandBuffer cmd,
                                         const Viewport& vp)
{
    struct Constants
    {
        int32_t m_rect[4];
        // xyz is the position (world space) of the sun of the light shaft,
        // w its radius, all 0 if the track has none
        float m_light_shaft_position_radius[4];
        // rgb is the color and a the opacity of the sun of the light shaft,
        // a is 0 if the track has none
        float m_light_shaft_color_opacity[4];
    };
    Constants constants = {};
    for (unsigned i = 0; i < 4; i++)
        constants.m_rect[i] = vp.m_rect[i];
    if (m_vk->isLightShaftEnabled())
    {
        const irr::core::vector3df& p = m_vk->getLightShaftPosition();
        constants.m_light_shaft_position_radius[0] = p.X;
        constants.m_light_shaft_position_radius[1] = p.Y;
        constants.m_light_shaft_position_radius[2] = p.Z;
        constants.m_light_shaft_position_radius[3] =
            m_vk->getLightShaftRadius();
        const irr::core::vector3df& c = m_vk->getLightShaftColor();
        constants.m_light_shaft_color_opacity[0] = c.X;
        constants.m_light_shaft_color_opacity[1] = c.Y;
        constants.m_light_shaft_color_opacity[2] = c.Z;
        constants.m_light_shaft_color_opacity[3] = m_vk->getLightShaftOpacity();
    }
    vkCmdPushConstants(cmd, getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(constants), &constants);
}   // pushConstants

}
