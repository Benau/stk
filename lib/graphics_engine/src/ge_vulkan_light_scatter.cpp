#include "ge_vulkan_light_scatter.hpp"

#include "ge_vulkan_draw_call.hpp"

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanLightScatter::GEVulkanLightScatter(GEVulkanDriver* vk,
                              const irr::core::dimension2d<irr::u32>& size,
                              GEVulkanAttachmentTexture* depth)
                    : GEVulkanLowResEffect(vk, size, depth,
                      { "GEVulkanLightScatter", "light_scatter.comp",
                      sizeof(int32_t) * 4, 1/*half size*/, 1.0f, 0.25f, 2.5f })
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
    int32_t rect[4] = { vp.m_rect[0], vp.m_rect[1], vp.m_rect[2],
        vp.m_rect[3] };
    vkCmdPushConstants(cmd, getPipelineLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(rect), rect);
}   // pushConstants

}
