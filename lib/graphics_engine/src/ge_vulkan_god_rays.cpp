#include "ge_vulkan_god_rays.hpp"

#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"

#include <limits>

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanGodRays::GEVulkanGodRays(GEVulkanDriver* vk,
                              const irr::core::dimension2d<irr::u32>& size,
                              GEVulkanAttachmentTexture* depth)
               : GEVulkanLowResEffect(vk, size, depth,
                 { "GEVulkanGodRays", "god_rays.comp",
                 // Rect (4 ints), position and radius of the sun, color
                 // (4 floats each)
                 sizeof(int32_t) * 4 + sizeof(float) * 8,
                 2/*quarter size*/,
                 // The image is a quarter of the size, so a texel is twice as
                 // big as the one of the half size light scattering and the
                 // factor is a half of it. The width is not smaller than a
                 // quarter texel, and it has no maximum
                 0.5f, 0.25f, std::numeric_limits<float>::max() })
{
}   // GEVulkanGodRays

// ----------------------------------------------------------------------------
bool GEVulkanGodRays::hasEffect(const GEVulkanDrawCall* dc) const
{
    return dc->hasGodRays();
}   // hasEffect

// ----------------------------------------------------------------------------
void GEVulkanGodRays::pushConstants(VkCommandBuffer cmd, const Viewport& vp)
{
    // The same for all viewports, already multiplied by the opacity
    const irr::core::vector3df& sun = m_vk->getGodRaysPosition();
    const irr::core::vector3df& color = m_vk->getGodRaysColor();
    struct PushConstants
    {
        int32_t m_rect[4];
        float m_sun_radius[4];
        float m_color[4];
    } pc = {};
    for (unsigned i = 0; i < 4; i++)
        pc.m_rect[i] = vp.m_rect[i];
    pc.m_sun_radius[0] = sun.X;
    pc.m_sun_radius[1] = sun.Y;
    pc.m_sun_radius[2] = sun.Z;
    pc.m_sun_radius[3] = m_vk->getGodRaysRadius();
    pc.m_color[0] = color.X;
    pc.m_color[1] = color.Y;
    pc.m_color[2] = color.Z;
    vkCmdPushConstants(cmd, getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(pc), &pc);
}   // pushConstants

}
