#include "ge_vulkan_final_chain.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_anti_aliasing.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_fbo_texture.hpp"
#include "ge_vulkan_fxaa.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
// Targets which can have a set (the main one and the separate targets)
const unsigned MAX_TARGETS = 32;
// ----------------------------------------------------------------------------
void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout old_layout,
                  VkImageLayout new_layout, VkPipelineStageFlags src_stage,
                  VkAccessFlags src_access, VkPipelineStageFlags dst_stage,
                  VkAccessFlags dst_access)
{
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;
    vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, NULL, 0, NULL, 1,
        &barrier);
}   // imageBarrier
// ----------------------------------------------------------------------------
void writeIOSet(VkDevice device, VkDescriptorSet set, VkImageView src,
                VkSampler sampler, VkImageLayout src_layout, VkImageView dst)
{
    VkDescriptorImageInfo src_info = {};
    src_info.sampler = sampler;
    src_info.imageView = src;
    src_info.imageLayout = src_layout;
    VkDescriptorImageInfo dst_info = {};
    dst_info.imageView = dst;
    dst_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    std::array<VkWriteDescriptorSet, 2> writes = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &src_info;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set;
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &dst_info;
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, NULL);
}   // writeIOSet
}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanFinalTarget::~GEVulkanFinalTarget()
{
    delete m_image;
    if (m_pool)
    {
        for (VkDescriptorSet set : m_sets)
        {
            if (set != VK_NULL_HANDLE)
                vkFreeDescriptorSets(m_pool->m_device, m_pool->m_pool, 1, &set);
        }
    }
}   // ~GEVulkanFinalTarget

// ----------------------------------------------------------------------------
GEVulkanFinalChain::GEVulkanFinalChain(GEVulkanDriver* vk,
    const irr::core::dimension2d<irr::u32>& capacity)
                  : m_vk(vk), m_capacity(capacity), m_io_layout(VK_NULL_HANDLE)
{
    try
    {
        createIOLayout();
        // The order of the stages is the order they run
        switch (getGEConfig()->m_anti_aliasing)
        {
        case GAA_FXAA:
            m_stages.push_back(new GEVulkanFXAA(m_vk, m_io_layout,
                m_capacity));
            break;
        case GAA_CMAA2:
            m_stages.push_back(new GEVulkanAntiAliasing(m_vk, m_io_layout,
                m_capacity));
            break;
        default:
            break;
        }
    }
    catch (std::exception& e)
    {
        logMessage(GLL_ERROR, "GEVulkanFinalChain: %s, anti-aliasing is "
            "disabled", e.what());
        destroy();
    }
}   // GEVulkanFinalChain

// ----------------------------------------------------------------------------
GEVulkanFinalChain::~GEVulkanFinalChain()
{
    destroy();
}   // ~GEVulkanFinalChain

// ----------------------------------------------------------------------------
void GEVulkanFinalChain::destroy()
{
    // The pipeline layouts of the stages use the layout
    for (GEVulkanFinalStage* stage : m_stages)
        delete stage;
    m_stages.clear();
    if (m_io_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_vk->getDevice(), m_io_layout, NULL);
        m_io_layout = VK_NULL_HANDLE;
    }
    // The targets which are alive keep it
    if (m_pool)
        m_pool->m_chain_alive = false;
    m_pool.reset();
}   // destroy

// ----------------------------------------------------------------------------
void GEVulkanFinalChain::createIOLayout()
{
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layout_info, NULL,
        &m_io_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorSetLayout failed for "
            "the final chain");
    }

    std::array<VkDescriptorPoolSize, 2> sizes = {};
    sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[0].descriptorCount = MAX_TARGETS * 2;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = MAX_TARGETS * 2;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // The sets of a target are freed when it is destroyed
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.poolSizeCount = sizes.size();
    pool_info.pPoolSizes = sizes.data();
    pool_info.maxSets = MAX_TARGETS * 2;
    m_pool = std::make_shared<GEVulkanFinalPool>();
    m_pool->m_device = device;
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool->m_pool) !=
        VK_SUCCESS)
    {
        m_pool->m_pool = VK_NULL_HANDLE;
        throw std::runtime_error("vkCreateDescriptorPool failed for the "
            "final chain");
    }
}   // createIOLayout

// ----------------------------------------------------------------------------
void GEVulkanFinalChain::initTarget(GEVulkanFBOTexture* fbo)
{
    if (m_stages.empty() || fbo->getFinalTarget())
        return;
    const irr::core::dimension2d<irr::u32>& size = fbo->getSize();
    if (size.Width > m_capacity.Width || size.Height > m_capacity.Height)
        return;

    VkDevice device = m_vk->getDevice();
    GEVulkanFinalTarget* target = new GEVulkanFinalTarget;
    try
    {
        target->m_pool = m_pool;
        // rgba8 is the format which every device supports as storage image
        // (the one of the FBO is bgra8, which not all of them do)
        target->m_image = new GEVulkanAttachmentTexture(m_vk, size,
            VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);

        VkDescriptorSetAllocateInfo alloc_info = {};
        alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        alloc_info.descriptorPool = m_pool->m_pool;
        alloc_info.descriptorSetCount = 1;
        alloc_info.pSetLayouts = &m_io_layout;
        if (vkAllocateDescriptorSets(device, &alloc_info,
            &target->m_sets[0]) != VK_SUCCESS)
        {
            target->m_sets[0] = VK_NULL_HANDLE;
            throw std::runtime_error("no descriptor set for the final target");
        }
        // Only the first stage can run until m_sets[1] exists, see render()
        target->m_result_in_image = true;
        writeIOSet(device, target->m_sets[0], fbo->getRenderImageView(),
            // Linear, clamped (GEVulkanFXAA samples between the pixels, the
            // other stages use texelFetch, so it doesn't change them)
            m_vk->getSampler(GVS_SKYBOX),
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            (VkImageView)target->m_image->getTextureHandler());
    }
    catch (std::exception& e)
    {
        logMessage(GLL_WARNING, "GEVulkanFinalChain: no anti-aliasing for a "
            "target: %s", e.what());
        delete target;
        return;
    }
    fbo->setFinalTarget(target);
}   // initTarget

// ----------------------------------------------------------------------------
void GEVulkanFinalChain::render(VkCommandBuffer cmd, GEVulkanFBOTexture* fbo,
    const std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCamera*> >& p)
{
    GEVulkanFinalTarget* target = fbo->getFinalTarget();
    if (!target || m_stages.empty())
        return;

    // Only the first stage until the image of the FBO can be written (the
    // sets[1] of the targets are not created, see GEVulkanFinalChain)
    unsigned stage_count = m_stages.size();
    if (target->m_sets[1] == VK_NULL_HANDLE)
        stage_count = std::min(stage_count, 1u);

    VkImage fbo_image = fbo->getImage();
    VkImage extra_image = target->m_image->getImage();
    const VkPipelineStageFlags all_graphics_and_compute =
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_TRANSFER_BIT;

    // The FBO has been rendered, and what the last frame read or wrote of
    // m_image (the blit, or the stages) has to be finished. The content of
    // m_image is not needed: the first stage writes all of it
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, all_graphics_and_compute,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }
    imageBarrier(cmd, extra_image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_GENERAL, all_graphics_and_compute,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    // Ping-pong: the stage i reads the image of the FBO if i is even (and
    // writes m_image), m_image (and writes the image of the FBO) if it's odd
    for (unsigned i = 0; i < stage_count; i++)
    {
        const bool writes_fbo = (i % 2) == 1;
        if (writes_fbo)
        {
            // Written as storage image (general layout), the content is
            // not needed because the stage writes all of it
            imageBarrier(cmd, fbo_image, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT);
        }
        m_stages[i]->apply(cmd, target->m_sets[i % 2], fbo->getSize(), p);
        if (i + 1 < stage_count)
        {
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL,
                0, NULL);
        }
    }

    // Everything which samples the FBO after it expects the layout of the
    // end of its render pass
    const VkPipelineStageFlags users = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (target->m_result_in_image)
    {
        // The last stage wrote m_image, which is what samples the FBO reads
        // (GEVulkanFBOTexture::getImageView), the image of the FBO is not
        // touched. m_result_in_image has to be the parity of stage_count
        imageBarrier(cmd, extra_image, VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            users, VK_ACCESS_SHADER_READ_BIT);
    }
    else
    {
        // The last stage wrote the image of the FBO
        imageBarrier(cmd, fbo_image, VK_IMAGE_LAYOUT_GENERAL,
            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
            users, VK_ACCESS_SHADER_READ_BIT);
    }
}   // render

}
