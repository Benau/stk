#include "ge_vulkan_compute_final_pass.hpp"

#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanComputeFinalPass::GEVulkanComputeFinalPass(GEVulkanDriver* vk,
    VkDescriptorSetLayout io_layout,
    const irr::core::dimension2d<irr::u32>& capacity, bool needs_depth,
    const GEComputePassDesc& desc)
                        : GEVulkanPostProcessing(vk, capacity), m_desc(desc),
                          m_needs_depth(needs_depth),
                          m_data_layout(VK_NULL_HANDLE),
                          m_pipeline_layout(VK_NULL_HANDLE),
                          m_pipeline(VK_NULL_HANDLE),
                          m_copy_pipeline(VK_NULL_HANDLE)
{
    try
    {
        if (m_desc.m_push_size > GE_FINAL_PASS_MAX_PUSH_SIZE)
        {
            throw std::runtime_error("the push constants of " +
                m_desc.m_shader + " are too big");
        }
        std::vector<VkDescriptorSetLayout> layouts = { io_layout };
        if (m_desc.m_camera_set)
        {
            // The camera is set 1 in utils/camera.glsl
            m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
            layouts.push_back(m_data_layout);
        }
        VkPushConstantRange push_constant = {};
        push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_constant.offset = 0;
        push_constant.size = getPushSize();
        VkPipelineLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = layouts.size();
        layout_info.pSetLayouts = layouts.data();
        layout_info.pushConstantRangeCount = 1;
        layout_info.pPushConstantRanges = &push_constant;
        if (vkCreatePipelineLayout(m_vk->getDevice(), &layout_info, NULL,
            &m_pipeline_layout) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreatePipelineLayout failed for " +
                m_desc.m_shader);
        }
        m_pipeline = createPipeline(m_desc.m_shader.c_str());
        // Doesn't use the push constants and the camera
        m_copy_pipeline = createPipeline("final_pass_copy.comp");
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanComputeFinalPass

// ----------------------------------------------------------------------------
GEVulkanComputeFinalPass::~GEVulkanComputeFinalPass()
{
    destroy();
}   // ~GEVulkanComputeFinalPass

// ----------------------------------------------------------------------------
void GEVulkanComputeFinalPass::destroy()
{
    VkDevice device = m_vk->getDevice();
    if (m_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_pipeline, NULL);
    if (m_copy_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_copy_pipeline, NULL);
    if (m_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    if (m_data_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_data_layout, NULL);
    m_pipeline = m_copy_pipeline = VK_NULL_HANDLE;
    m_pipeline_layout = VK_NULL_HANDLE;
    m_data_layout = VK_NULL_HANDLE;
}   // destroy

// ----------------------------------------------------------------------------
VkPipeline GEVulkanComputeFinalPass::createPipeline(const char* shader) const
{
    VkComputePipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = GEVulkanShaderManager::getShader(shader);
    info.stage.pName = "main";
    info.layout = m_pipeline_layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(m_vk->getDevice(), m_vk->getPipelineCache(),
        1, &info, NULL, &pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string(
            "vkCreateComputePipelines failed for ") + shader);
    }
    return pipeline;
}   // createPipeline

// ----------------------------------------------------------------------------
void GEVulkanComputeFinalPass::collect(const std::vector<std::pair<
    GEVulkanDrawCall*, GEVulkanCamera*> >& p, std::vector<Item>* items) const
{
    items->clear();
    for (unsigned i = 0; i < p.size(); i++)
    {
        Item item;
        if (!getViewport(p[i].first, p[i].second, &item.m_viewport))
            continue;
        GEFinalPassView& view = item.m_view;
        for (unsigned j = 0; j < 4; j++)
            view.m_rect[j] = item.m_viewport.m_rect[j];
        view.m_height = item.m_viewport.m_height;
        view.m_scale = view.m_height / REFERENCE_HEIGHT;
        view.m_index = i;
        view.m_user_data = p[i].second->getUserData();
        view.m_pvm = p[i].second->getUBOData()->m_projection_view_matrix;
        view.m_prev_pvm = p[i].second->getPrevPVM();
        if (m_desc.m_is_active && !m_desc.m_is_active(view))
            continue;
        items->push_back(item);
    }
}   // collect

// ----------------------------------------------------------------------------
bool GEVulkanComputeFinalPass::isActive(const std::vector<std::pair<
    GEVulkanDrawCall*, GEVulkanCamera*> >& p) const
{
    // Without a condition the pass runs for any viewport
    if (!m_desc.m_is_active)
        return !p.empty();
    // The rect is the one clamped to the biggest FBO here, only the
    // viewports of the camera matter
    std::vector<Item> items;
    collect(p, &items);
    return !items.empty();
}   // isActive

// ----------------------------------------------------------------------------
void GEVulkanComputeFinalPass::apply(VkCommandBuffer cmd,
    VkDescriptorSet io_set, const irr::core::dimension2d<irr::u32>& size,
    const std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCamera*> >& p)
{
    // The size of the FBO, which getViewport clamps the viewports to
    m_size = size;
    std::vector<Item> items;
    collect(p, &items);

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &io_set, 0, NULL);

    // Output = input outside of the viewports, which is the whole image
    // unless one viewport is all of it (no splitscreen)
    const std::array<int32_t, 4> full = {{ 0, 0, (int32_t)size.Width,
        (int32_t)size.Height }};
    if (items.size() != 1 || items[0].m_viewport.m_rect != full)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_copy_pipeline);
        vkCmdDispatch(cmd, (size.Width + 7) / 8, (size.Height + 7) / 8, 1);
        // The viewports write the pixels the copy wrote
        if (!items.empty())
            computeToComputeBarrier(cmd, 0);
    }
    if (items.empty())
        return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    // The viewports don't overlap and only read what is inside of their rect,
    // so there is no barrier between them
    uint8_t constants[sizeof(Header) + GE_FINAL_PASS_MAX_PUSH_SIZE];
    for (const Item& item : items)
    {
        if (m_desc.m_camera_set)
        {
            // The camera of the viewport: the same unprojection as the pass
            // which wrote the depth, for any position on the screen
            item.m_viewport.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        }
        memset(constants, 0, sizeof(constants));
        Header* header = (Header*)constants;
        for (unsigned i = 0; i < 4; i++)
            header->m_rect[i] = item.m_view.m_rect[i];
        header->m_params[0] = item.m_view.m_height;
        header->m_params[1] = item.m_view.m_scale;
        header->m_params[2] = (float)item.m_view.m_index;
        if (m_desc.m_fill && m_desc.m_push_size != 0)
            m_desc.m_fill(item.m_view, constants + sizeof(Header));
        vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, getPushSize(), constants);
        const std::array<int32_t, 4>& r = item.m_viewport.m_rect;
        vkCmdDispatch(cmd, (r[2] - r[0] + 7) / 8, (r[3] - r[1] + 7) / 8, 1);
    }
}   // apply

}
