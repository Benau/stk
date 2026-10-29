#include "ge_vulkan_final_chain.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_fbo_texture.hpp"

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
// See GEVulkanFinalTarget::m_sets
const unsigned SETS_PER_TARGET = 3;
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
                VkSampler sampler, VkImageLayout src_layout, VkImageView dst,
                VkImageView depth, VkSampler depth_sampler)
{
    VkDescriptorImageInfo src_info = {};
    src_info.sampler = sampler;
    src_info.imageView = src;
    src_info.imageLayout = src_layout;
    VkDescriptorImageInfo dst_info = {};
    dst_info.imageView = dst;
    dst_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo depth_info = {};
    depth_info.sampler = depth_sampler;
    depth_info.imageView = depth;
    depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    std::array<VkWriteDescriptorSet, 3> writes = {};
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
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[2].pImageInfo = &depth_info;
    // Binding 2 is only written if the FBO keeps its depth, only the stages
    // which need it read it
    vkUpdateDescriptorSets(device, depth != VK_NULL_HANDLE ? 3 : 2,
        writes.data(), 0, NULL);
}   // writeIOSet
}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanFinalTarget::~GEVulkanFinalTarget()
{
    for (GEVulkanAttachmentTexture* image : m_images)
        delete image;
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
    }
    catch (std::exception& e)
    {
        logMessage(GLL_ERROR, "GEVulkanFinalChain: %s, the final passes are "
            "disabled", e.what());
        destroy();
        return;
    }
    // Sorted by order, so for example the anti-aliasing is after the depth of
    // field and works on the edges of it too
    for (const GEFinalPassInfo& info : getEnabledFinalPasses())
    {
        if (m_stages.size() >= 32)
        {
            logMessage(GLL_WARNING, "GEVulkanFinalChain: too many passes, %s "
                "is ignored", info.m_name.c_str());
            continue;
        }
        try
        {
            GEVulkanFinalStage* stage = info.m_create ?
                info.m_create(m_vk, m_io_layout, m_capacity) : NULL;
            if (!stage)
                continue;
            Stage s;
            s.m_stage = stage;
            s.m_main_only = info.m_main_target_only;
            s.m_name = info.m_name;
            m_stages.push_back(s);
        }
        catch (std::exception& e)
        {
            // A stage which can't be created doesn't stop the other ones
            logMessage(GLL_ERROR, "GEVulkanFinalChain: %s is disabled: %s",
                info.m_name.c_str(), e.what());
        }
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
    for (Stage& stage : m_stages)
        delete stage.m_stage;
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
    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    // The depth of the FBO, for the stages which need it
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
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
    sizes[0].descriptorCount = MAX_TARGETS * SETS_PER_TARGET * 2;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    sizes[1].descriptorCount = MAX_TARGETS * SETS_PER_TARGET;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // The sets of a target are freed when it is destroyed
    pool_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.poolSizeCount = sizes.size();
    pool_info.pPoolSizes = sizes.data();
    pool_info.maxSets = MAX_TARGETS * SETS_PER_TARGET;
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
void GEVulkanFinalChain::initTarget(GEVulkanFBOTexture* fbo, bool main_target)
{
    if (m_stages.empty() || fbo->getFinalTarget())
        return;
    const irr::core::dimension2d<irr::u32>& size = fbo->getSize();
    if (size.Width > m_capacity.Width || size.Height > m_capacity.Height)
        return;

    // The stages which run for this FBO
    uint32_t mask = 0;
    unsigned count = 0;
    const bool has_depth = main_target && fbo->hasStoredDepth() &&
        fbo->getDepthTexture();
    for (unsigned i = 0; i < m_stages.size(); i++)
    {
        if (m_stages[i].m_stage->needsDepth() && !has_depth)
            continue;
        if (m_stages[i].m_main_only && !main_target)
            continue;
        mask |= 1u << i;
        count++;
    }
    if (count == 0)
        return;

    VkDevice device = m_vk->getDevice();
    GEVulkanFinalTarget* target = new GEVulkanFinalTarget;
    try
    {
        target->m_pool = m_pool;
        target->m_stage_mask = mask;
        target->m_stage_count = count;
        // rgba8 is the format which every device supports as storage image
        // (the one of the FBO is bgra8, which not all of them do). The second
        // image is only for the second stage and the next ones
        for (unsigned i = 0; i < std::min(count, 2u); i++)
        {
            target->m_images[i] = new GEVulkanAttachmentTexture(m_vk, size,
                VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_USAGE_STORAGE_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        const unsigned set_count = std::min(count, 3u);
        for (unsigned i = 0; i < set_count; i++)
        {
            VkDescriptorSetAllocateInfo alloc_info = {};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = m_pool->m_pool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &m_io_layout;
            if (vkAllocateDescriptorSets(device, &alloc_info,
                &target->m_sets[i]) != VK_SUCCESS)
            {
                target->m_sets[i] = VK_NULL_HANDLE;
                throw std::runtime_error("no descriptor set for the final "
                    "target");
            }
        }

        // Linear, clamped (GEVulkanFXAA samples between the pixels, the other
        // stages use texelFetch, so it doesn't change them)
        VkSampler sampler = m_vk->getSampler(GVS_SKYBOX);
        VkImageView image_0 = (VkImageView)target->m_images[0]->
            getTextureHandler();
        VkImageView image_1 = target->m_images[1] ?
            (VkImageView)target->m_images[1]->getTextureHandler() :
            VK_NULL_HANDLE;
        VkImageView depth = has_depth ?
            (VkImageView)fbo->getDepthTexture()->getTextureHandler() :
            VK_NULL_HANDLE;
        VkSampler depth_sampler = m_vk->getSampler(GVS_NEAREST);
        writeIOSet(device, target->m_sets[0], fbo->getRenderImageView(),
            sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, image_0, depth,
            depth_sampler);
        if (set_count > 1)
        {
            // Stage 1, 3...: images[0] -> images[1]
            writeIOSet(device, target->m_sets[1], image_0, sampler,
                VK_IMAGE_LAYOUT_GENERAL, image_1, depth, depth_sampler);
        }
        if (set_count > 2)
        {
            // Stage 2, 4...: images[1] -> images[0]
            writeIOSet(device, target->m_sets[2], image_1, sampler,
                VK_IMAGE_LAYOUT_GENERAL, image_0, depth, depth_sampler);
        }
    }
    catch (std::exception& e)
    {
        logMessage(GLL_WARNING, "GEVulkanFinalChain: no final passes for a "
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
    if (!target || m_stages.empty() || target->m_stage_count == 0)
        return;

    const VkPipelineStageFlags all_graphics_and_compute =
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_TRANSFER_BIT;

    // The FBO has been rendered, and what the last frame read or wrote of
    // the images (the blit, or the stages) has to be finished. The content of
    // the images is not needed: the first stage writes all of the first one,
    // and the second one writes all of the second
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
    for (GEVulkanAttachmentTexture* image : target->m_images)
    {
        if (!image)
            continue;
        imageBarrier(cmd, image->getImage(), VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_GENERAL, all_graphics_and_compute,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    // The stage k which runs is the first one if k is 0, so it reads the FBO,
    // otherwise the one the previous stage wrote (images[0] if k is odd,
    // images[1] if it's even), see GEVulkanFinalTarget::m_sets
    unsigned k = 0;
    for (unsigned i = 0; i < m_stages.size(); i++)
    {
        if ((target->m_stage_mask & (1u << i)) == 0)
            continue;
        const unsigned set = k == 0 ? 0 : (k % 2 == 1 ? 1 : 2);
        m_stages[i].m_stage->apply(cmd, target->m_sets[set], fbo->getSize(),
            p);
        k++;
        if (k < target->m_stage_count)
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
    // end of its render pass: it samples the last image the stages wrote
    // (GEVulkanFBOTexture::getImageView), the image of the FBO is not touched
    const VkPipelineStageFlags users = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    imageBarrier(cmd, target->getResultImage()->getImage(),
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
        users, VK_ACCESS_SHADER_READ_BIT);

    // The stages read the image and the depth of the FBO: the render passes
    // of the next frame (they write them again) have to wait for that
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 0,
        NULL);
}   // render

}
