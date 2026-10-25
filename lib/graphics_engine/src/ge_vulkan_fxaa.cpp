#include "ge_vulkan_fxaa.hpp"

#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <stdexcept>

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanFXAA::GEVulkanFXAA(GEVulkanDriver* vk, VkDescriptorSetLayout io_layout,
    const irr::core::dimension2d<irr::u32>& capacity)
            : GEVulkanPostProcessing(vk, capacity),
              m_pipeline_layout(VK_NULL_HANDLE), m_pipeline(VK_NULL_HANDLE)
{
    VkDevice device = m_vk->getDevice();
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &io_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(device, &layout_info, NULL,
        &m_pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for "
            "GEVulkanFXAA");
    }

    VkComputePipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = GEVulkanShaderManager::getShader("fxaa.comp");
    info.stage.pName = "main";
    info.layout = m_pipeline_layout;
    if (vkCreateComputePipelines(device, m_vk->getPipelineCache(), 1, &info,
        NULL, &m_pipeline) != VK_SUCCESS)
    {
        m_pipeline = VK_NULL_HANDLE;
        destroy();
        throw std::runtime_error("vkCreateComputePipelines failed for "
            "fxaa.comp");
    }
}   // GEVulkanFXAA

// ----------------------------------------------------------------------------
GEVulkanFXAA::~GEVulkanFXAA()
{
    destroy();
}   // ~GEVulkanFXAA

// ----------------------------------------------------------------------------
void GEVulkanFXAA::destroy()
{
    VkDevice device = m_vk->getDevice();
    if (m_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_pipeline, NULL);
    m_pipeline = VK_NULL_HANDLE;
    if (m_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    m_pipeline_layout = VK_NULL_HANDLE;
}   // destroy

// ----------------------------------------------------------------------------
void GEVulkanFXAA::dispatch(VkCommandBuffer cmd,
                            const std::array<int32_t, 4>& rect, bool copy)
{
    PushConstants pc;
    for (unsigned i = 0; i < 4; i++)
        pc.m_rect[i] = rect[i];
    pc.m_copy = copy ? 1 : 0;
    vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(pc), &pc);
    vkCmdDispatch(cmd, (rect[2] - rect[0] + 7) / 8,
        (rect[3] - rect[1] + 7) / 8, 1);
}   // dispatch

// ----------------------------------------------------------------------------
void GEVulkanFXAA::apply(VkCommandBuffer cmd, VkDescriptorSet io_set,
    const irr::core::dimension2d<irr::u32>& size,
    const std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCamera*> >& p)
{
    // The size of the FBO, which getViewport clamps the viewports to
    m_size = size;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &io_set, 0, NULL);

    std::vector<std::array<int32_t, 4> > rects;
    for (auto& q : p)
    {
        Viewport vp;
        if (getViewport(q.first, q.second, &vp))
            rects.push_back(vp.m_rect);
    }

    // Output = input outside of the viewports, which is the whole image
    // unless one viewport is all of it (no splitscreen)
    const std::array<int32_t, 4> full = {{ 0, 0, (int32_t)size.Width,
        (int32_t)size.Height }};
    if (rects.size() != 1 || rects[0] != full)
    {
        dispatch(cmd, full, true/*copy*/);
        // The viewports write the pixels the copy wrote
        computeToComputeBarrier(cmd, 0);
    }

    // The viewports don't overlap and read the input only, so no barrier
    // between them
    for (const std::array<int32_t, 4>& rect : rects)
        dispatch(cmd, rect, false/*copy*/);
}   // apply

}
