#include "ge_vulkan_post_processing.hpp"

#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_driver.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>

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
    // Both current users have a very small fixed number of sets, so avoid
    // making this helper depend on a std::vector
    std::array<VkDescriptorSetLayout, 5> layouts = {};
    if (set_count > layouts.size())
        throw std::runtime_error(
            "Too many descriptor sets for GEVulkanPostProcessing");

    for (unsigned i = 0; i < set_count; i++)
        layouts[i] = layout;

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

}
