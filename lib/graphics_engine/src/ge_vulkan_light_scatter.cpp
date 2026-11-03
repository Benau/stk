#include "ge_vulkan_light_scatter.hpp"
#include "ge_vulkan_driver.hpp"

#include "ge_vulkan_draw_call.hpp"

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
    // The sun of the light shaft (god rays) is read every frame, because the
    // track can enable or disable it at runtime. Everything is 0 (opacity
    // too) if there is none, which disables the sun scattering in the shader.
    // Not related to the shadows of the scattering, which only depend on the
    // shadow settings.
    struct Constants
    {
        int32_t m_rect[4];
        float m_light_shaft_position_radius[4];
        float m_light_shaft_color_opacity[4];
    } constants = {};
    for (unsigned i = 0; i < 4; i++)
        constants.m_rect[i] = vp.m_rect[i];
    if (vp.m_draw_call->hasGodRays())
    {
        const irr::core::vector3df& p = m_vk->getGodRaysPosition();
        constants.m_light_shaft_position_radius[0] = p.X;
        constants.m_light_shaft_position_radius[1] = p.Y;
        constants.m_light_shaft_position_radius[2] = p.Z;
        constants.m_light_shaft_position_radius[3] = m_vk->getGodRaysRadius();
        const irr::core::vector3df& c = m_vk->getGodRaysColor();
        constants.m_light_shaft_color_opacity[0] = c.X;
        constants.m_light_shaft_color_opacity[1] = c.Y;
        constants.m_light_shaft_color_opacity[2] = c.Z;
        constants.m_light_shaft_color_opacity[3] =
            (c.getLengthSQ() > 0.0f) ? 1.0f : 0.0f;
    }
    vkCmdPushConstants(cmd, getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(constants), &constants);
}   // pushConstants

}
