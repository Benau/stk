#include "ge_vulkan_post_processing.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <vector>

namespace GE
{
// ----------------------------------------------------------------------------
const float GEVulkanPostProcessing::REFERENCE_HEIGHT = 1080.0f;
// ----------------------------------------------------------------------------
irr::core::dimension2d<irr::u32> GEVulkanPostProcessing::halfSize(
    const irr::core::dimension2d<irr::u32>& size)
{
    return irr::core::dimension2d<irr::u32>(std::max(1u, (size.Width + 1) / 2),
        std::max(1u, (size.Height + 1) / 2));
}   // halfSize

// ----------------------------------------------------------------------------
std::array<int32_t, 4> GEVulkanPostProcessing::halfRect(
    const std::array<int32_t, 4>& r)
{
    // Covers all pixels of the rect: floor of the start, ceil of the end
    return {{ r[0] >> 1, r[1] >> 1, (r[2] + 1) >> 1, (r[3] + 1) >> 1 }};
}   // halfRect

// ----------------------------------------------------------------------------
std::array<int32_t, 4> GEVulkanPostProcessing::innerHalfRect(
    const std::array<int32_t, 4>& r)
{
    return {{ (r[0] + 1) >> 1, (r[1] + 1) >> 1, r[2] >> 1, r[3] >> 1 }};
}   // innerHalfRect

// ----------------------------------------------------------------------------
bool GEVulkanPostProcessing::getViewport(GEVulkanDrawCall* dc,
                                      GEVulkanCameraSceneNode* cam,
                                      Viewport* out) const
{
    const VkViewport vp = dc->getRenderViewport(m_vk, cam->getViewPort());
    const int32_t x0 = std::max(0, (int32_t)vp.x);
    const int32_t y0 = std::max(0, (int32_t)vp.y);
    const int32_t x1 = std::min((int32_t)m_size.Width,
        (int32_t)vp.x + (int32_t)vp.width);
    const int32_t y1 = std::min((int32_t)m_size.Height,
        (int32_t)vp.y + (int32_t)vp.height);
    if (x1 <= x0 || y1 <= y0)
        return false;
    // Same as GEVulkanDrawCall::getRenderViewport, before rotation
    const float scale = m_vk->getSeparateRTTTexture() ? 1.0f :
        getGEConfig()->m_render_scale;
    out->m_draw_call = dc;
    out->m_rect = {{ x0, y0, x1, y1 }};
    out->m_height = cam->getViewPort().getHeight() * scale;
    return true;
}   // getViewport

// ----------------------------------------------------------------------------
void GEVulkanPostProcessing::initializeGeneralImages(
                                            GEVulkanAttachmentTexture** images,
                                                          unsigned count) const
{
    VkCommandBuffer command_buffer =
        GEVulkanCommandLoader::beginSingleTimeCommands();

    for (unsigned i = 0; i < count; i++)
    {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = images[i]->getImage();
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;

        vkCmdPipelineBarrier(command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &barrier);
    }

    GEVulkanCommandLoader::endSingleTimeCommands(command_buffer);
}   // initializeGeneralImages

// ----------------------------------------------------------------------------
void GEVulkanPostProcessing::createImageDescriptorSetLayout(
                                           VkDescriptorSetLayout* layout) const
{
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {};

    bindings[0].binding = 0;
    bindings[0].descriptorCount = 1;
    bindings[0].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    bindings[1].binding = 1;
    bindings[1].descriptorCount = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_vk->getDevice(), &layout_info, NULL,
        layout) != VK_SUCCESS)
    {
        throw std::runtime_error(
            "vkCreateDescriptorSetLayout failed for "
            "GEVulkanPostProcessing");
    }
}   // createImageDescriptorSetLayout

// ----------------------------------------------------------------------------
void GEVulkanPostProcessing::createImageDescriptorPool(
                              unsigned set_count, VkDescriptorPool* pool) const
{
    std::array<VkDescriptorPoolSize, 2> pool_sizes = {};

    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount = set_count;

    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = set_count;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = set_count;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();

    if (vkCreateDescriptorPool(m_vk->getDevice(), &pool_info, NULL, pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error(
            "vkCreateDescriptorPool failed for GEVulkanPostProcessing");
    }
}   // createImageDescriptorPool

// ----------------------------------------------------------------------------
void GEVulkanPostProcessing::allocateImageDescriptorSets(VkDescriptorPool pool,
                                                  VkDescriptorSetLayout layout,
                                                  unsigned set_count,
                                                  VkDescriptorSet* sets) const
{
    std::vector<VkDescriptorSetLayout> layouts(set_count, layout);

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = pool;
    alloc_info.descriptorSetCount = set_count;
    alloc_info.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(m_vk->getDevice(), &alloc_info, sets) !=
        VK_SUCCESS)
    {
        throw std::runtime_error(
            "vkAllocateDescriptorSets failed for GEVulkanPostProcessing");
    }
}   // allocateImageDescriptorSets

// ----------------------------------------------------------------------------
void GEVulkanPostProcessing::writeImageDescriptorSet(VkDescriptorSet set,
                                                     VkImageView input_view,
                                                     VkSampler input_sampler,
                                                    VkImageLayout input_layout,
                                                       VkImageView output_view,
                                             VkImageLayout output_layout) const
{
    VkDescriptorImageInfo input = {};
    input.sampler = input_sampler;
    input.imageLayout = input_layout;
    input.imageView = input_view;

    VkDescriptorImageInfo output = {};
    output.imageLayout = output_layout;
    output.imageView = output_view;

    std::array<VkWriteDescriptorSet, 2> writes = {};

    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].descriptorCount = 1;
    writes[0].pImageInfo = &input;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].descriptorCount = 1;
    writes[1].pImageInfo = &output;

    vkUpdateDescriptorSets(m_vk->getDevice(), writes.size(), writes.data(),
        0, NULL);
}   // writeImageDescriptorSet

// ----------------------------------------------------------------------------
void GEVulkanPostProcessing::createBlurPipeline(const char* shader,
                                             VkDescriptorSetLayout layout,
                                             VkPipelineLayout* pipeline_layout,
                                             VkPipeline* pipeline) const
{
    VkDevice device = m_vk->getDevice();
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(BlurPushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
        pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string(
            "vkCreatePipelineLayout failed for ") + shader);
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = GEVulkanShaderManager::getShader(shader);
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = *pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
        NULL, pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string(
            "vkCreateComputePipelines failed for ") + shader);
    }
}   // createBlurPipeline

// ----------------------------------------------------------------------------
void GEVulkanPostProcessing::dispatchBlur(VkCommandBuffer cmd,
                                       VkPipelineLayout pipeline_layout,
                                       VkDescriptorSet set, BlurMode mode,
                                       const std::array<int32_t, 4>& src_rect,
                                       const std::array<int32_t, 4>& dst_rect,
                                       float scale, float blend,
                                       int32_t direction_x,
                                       int32_t direction_y) const
{
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline_layout, 0, 1, &set, 0, NULL);
    BlurPushConstants pc = {};
    for (unsigned i = 0; i < 4; i++)
    {
        pc.m_src_rect[i] = src_rect[i];
        pc.m_dst_rect[i] = dst_rect[i];
    }
    pc.m_direction[0] = direction_x;
    pc.m_direction[1] = direction_y;
    pc.m_mode = mode;
    pc.m_scale = scale;
    pc.m_blend = blend;
    vkCmdPushConstants(cmd, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(pc), &pc);
    const uint32_t w = (uint32_t)(dst_rect[2] - dst_rect[0]);
    const uint32_t h = (uint32_t)(dst_rect[3] - dst_rect[1]);
    vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
}   // dispatchBlur

// ----------------------------------------------------------------------------
void GEVulkanPostProcessing::computeToComputeBarrier(VkCommandBuffer cmd,
                                         VkPipelineStageFlags extra_dst_stage)
{
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | extra_dst_stage, 0, 1,
        &barrier, 0, NULL, 0, NULL);
}   // computeToComputeBarrier

}
