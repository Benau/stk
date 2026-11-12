#include "ge_vulkan_stochastic_reflection.hpp"

#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
// Push constants of stochastic_reflection_mipmap.comp
struct MipmapPushConstants
{
    int32_t m_rect[4];
    int32_t m_origin[2];
    int32_t m_levels;
};

// ----------------------------------------------------------------------------
// Levels of the trace target of the stochastic reflection
// (STOCHASTIC_REFLECTION_LEVELS of utils/stochastic_reflection.glsl)
const unsigned g_stochastic_levels = 4;

// ----------------------------------------------------------------------------
// The count, the indirect dispatch of the trace and of the resolve, and a
// padding, before the texels of the list of the stochastic reflection, and the
// threads which trace a texel (STOCHASTIC_REFLECTION_THREADS), see
// utils/stochastic_reflection.glsl
const unsigned g_stochastic_list_header = 8;
const uint32_t g_stochastic_threads = 8;
const VkDeviceSize g_stochastic_trace_dispatch = sizeof(uint32_t);
const VkDeviceSize g_stochastic_resolve_dispatch = 4 * sizeof(uint32_t);

// ----------------------------------------------------------------------------
// The bindings of the set of the stochastic reflection, see
// utils/stochastic_reflection.glsl and its passes
enum StochasticBinding : unsigned
{
    SB_TRACE = 0,
    SB_LIST,
    SB_NORMAL,
    SB_DEPTH,
    SB_TRACE_LEVELS,
    SB_TRACE_SAMPLED,
    SB_RESOLVED,
    SB_COUNT
};

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanStochasticReflection::GEVulkanStochasticReflection(GEVulkanDriver* vk,
                                   const irr::core::dimension2d<irr::u32>& size,
                                   GEVulkanAttachmentTexture* depth,
                                   GEVulkanAttachmentTexture* normal, bool ssr,
                                   bool rt)
                            : GEVulkanSolidReflection(vk, size, ssr, rt),
                              m_trace_image(VK_NULL_HANDLE),
                              m_trace_memory(VK_NULL_HANDLE),
                              m_trace_view(VK_NULL_HANDLE), m_trace_levels(0),
                              m_mipmap_layout(VK_NULL_HANDLE),
                              m_mipmap_pipeline(VK_NULL_HANDLE),
                              m_resolve_layout(VK_NULL_HANDLE),
                              m_resolve_pipeline(VK_NULL_HANDLE)
{
    m_trace_level_views.fill(VK_NULL_HANDLE);
    // GEVulkanSolidReflection is destroyed by its destructor if this throws
    try
    {
        // The radiance premultiplied by the coverage needs the alpha, see
        // utils/stochastic_reflection.glsl (rgba16f is a storage format of
        // every device)
        const irr::core::dimension2d<irr::u32> half = halfSize(size);
        m_reflection = new GEVulkanAttachmentTexture(vk, half,
            VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        createTraceImage(half);
        initializeGeneralImages(&m_reflection, 1);
        // The memory of a new image can have anything (the one of an image
        // which was destroyed when the render target was created again, NaN
        // too), both targets start as 0
        VkCommandBuffer cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
        clearImage(cmd, m_reflection->getImage(), 1);
        clearImage(cmd, m_trace_image, m_trace_levels);
        GEVulkanCommandLoader::endSingleTimeCommands(cmd);
        // The header and every texel of the half size target
        createList(half, g_stochastic_list_header);
        createDescriptor(depth, normal);
        createClassifyPipeline("stochastic_reflection_classify.comp");
        createPipelines();
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanStochasticReflection

// ----------------------------------------------------------------------------
GEVulkanStochasticReflection::~GEVulkanStochasticReflection()
{
    destroy();
}   // ~GEVulkanStochasticReflection

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::destroy()
{
    m_vk->waitIdle();
    VkDevice device = m_vk->getDevice();
    if (m_mipmap_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_mipmap_pipeline, NULL);
    if (m_mipmap_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_mipmap_layout, NULL);
    if (m_resolve_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_resolve_pipeline, NULL);
    if (m_resolve_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_resolve_layout, NULL);
    m_mipmap_pipeline = VK_NULL_HANDLE;
    m_mipmap_layout = VK_NULL_HANDLE;
    m_resolve_pipeline = VK_NULL_HANDLE;
    m_resolve_layout = VK_NULL_HANDLE;
    for (VkImageView& view : m_trace_level_views)
    {
        if (view != VK_NULL_HANDLE)
            vkDestroyImageView(device, view, NULL);
        view = VK_NULL_HANDLE;
    }
    if (m_trace_view != VK_NULL_HANDLE)
        vkDestroyImageView(device, m_trace_view, NULL);
    m_trace_view = VK_NULL_HANDLE;
    if (m_trace_image != VK_NULL_HANDLE)
    {
        vmaDestroyImage(m_vk->getVmaAllocator(), m_trace_image,
            m_trace_memory);
    }
    m_trace_image = VK_NULL_HANDLE;
    m_trace_memory = VK_NULL_HANDLE;
}   // destroy

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::createTraceImage(
                                   const irr::core::dimension2d<irr::u32>& size)
{
    // The levels of the mipmap, fewer if the target is smaller than 8 texels
    const unsigned max_levels = (unsigned)std::floor(std::log2(
        (float)std::max(size.Width, size.Height))) + 1;
    m_trace_levels = std::min(g_stochastic_levels, max_levels);
    VkImageCreateInfo image_info = {};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.extent.width = size.Width;
    image_info.extent.height = size.Height;
    image_info.extent.depth = 1;
    image_info.mipLevels = m_trace_levels;
    image_info.arrayLayers = 1;
    image_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    // Cleared every frame (see render())
    image_info.usage = VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (vmaCreateImage(m_vk->getVmaAllocator(), &image_info, &alloc_info,
        &m_trace_image, &m_trace_memory, NULL) != VK_SUCCESS)
    {
        throw std::runtime_error("vmaCreateImage failed for the trace of "
            "GEVulkanStochasticReflection");
    }

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = m_trace_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = image_info.format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.levelCount = m_trace_levels;
    view_info.subresourceRange.layerCount = 1;
    if (vkCreateImageView(m_vk->getDevice(), &view_info, NULL,
        &m_trace_view) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateImageView failed for the trace of "
            "GEVulkanStochasticReflection");
    }
    view_info.subresourceRange.levelCount = 1;
    for (unsigned i = 0; i < m_trace_levels; i++)
    {
        view_info.subresourceRange.baseMipLevel = i;
        if (vkCreateImageView(m_vk->getDevice(), &view_info, NULL,
            &m_trace_level_views[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed for a level of "
                "the trace of GEVulkanStochasticReflection");
        }
    }

    // Always in the general layout, like the output
    VkCommandBuffer cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_trace_image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = m_trace_levels;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1,
        &barrier);
    GEVulkanCommandLoader::endSingleTimeCommands(cmd);
}   // createTraceImage

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::clearImage(VkCommandBuffer cmd,
                                              VkImage image, unsigned levels)
{
    // The image is in the general layout, which a transfer can write
    VkClearColorValue zero = {};
    VkImageSubresourceRange range = {};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = levels;
    range.layerCount = 1;
    vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1,
        &range);
}   // clearImage

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::createDescriptor(
                                              GEVulkanAttachmentTexture* depth,
                                              GEVulkanAttachmentTexture* normal)
{
    // See StochasticBinding
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, SB_COUNT> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    bindings[SB_TRACE].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[SB_LIST].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[SB_NORMAL].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[SB_DEPTH].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[SB_TRACE_LEVELS].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[SB_TRACE_LEVELS].descriptorCount = g_stochastic_levels;
    bindings[SB_TRACE_SAMPLED].descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[SB_RESOLVED].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layout_info, NULL, &m_layout) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorSetLayout failed for "
            "GEVulkanStochasticReflection");
    }

    std::array<VkDescriptorPoolSize, 3> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 2 + g_stochastic_levels;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = 1;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[2].descriptorCount = 3;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanStochasticReflection");
    }

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &m_layout;
    if (vkAllocateDescriptorSets(device, &alloc_info, &m_set) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanStochasticReflection");
    }

    VkDescriptorImageInfo trace_info = {};
    trace_info.imageView = m_trace_level_views[0];
    trace_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = m_list;
    buffer_info.range = VK_WHOLE_SIZE;
    VkDescriptorImageInfo normal_info = {};
    normal_info.sampler = m_vk->getSampler(GVS_NEAREST);
    normal_info.imageView = (VkImageView)normal->getTextureHandler();
    normal_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkDescriptorImageInfo depth_info = {};
    depth_info.sampler = m_vk->getSampler(GVS_NEAREST);
    depth_info.imageView = (VkImageView)depth->getTextureHandler();
    depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    // The levels which the target doesn't have are the last one, the mipmap
    // doesn't write them
    std::array<VkDescriptorImageInfo, g_stochastic_levels> level_infos = {};
    for (unsigned i = 0; i < level_infos.size(); i++)
    {
        level_infos[i].imageView =
            m_trace_level_views[std::min(i, m_trace_levels - 1)];
        level_infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorImageInfo sampled_info = {};
    sampled_info.sampler = m_vk->getSampler(GVS_SKYBOX);
    sampled_info.imageView = m_trace_view;
    sampled_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo resolved_info = {};
    resolved_info.imageView = getImageView();
    resolved_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    const std::array<const VkDescriptorImageInfo*, SB_COUNT> image_infos =
    {{
        &trace_info, NULL, &normal_info, &depth_info, level_infos.data(),
        &sampled_info, &resolved_info
    }};
    std::array<VkWriteDescriptorSet, SB_COUNT> writes = {};
    for (unsigned i = 0; i < writes.size(); i++)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = bindings[i].descriptorCount;
        writes[i].descriptorType = bindings[i].descriptorType;
        if (i == SB_LIST)
            writes[i].pBufferInfo = &buffer_info;
        else
            writes[i].pImageInfo = image_infos[i];
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, NULL);
}   // createDescriptor

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::createPipelines()
{
    // The resolve has the camera of the viewport, the same layout as every
    // draw call
    if (m_data_layout == VK_NULL_HANDLE)
        m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.size = sizeof(MipmapPushConstants);
    const std::array<VkDescriptorSetLayout, 2> resolve_layouts =
        {{ m_layout, m_data_layout }};
    const std::array<const char*, 2> shaders =
        {{ "stochastic_reflection_mipmap.comp",
        "stochastic_reflection_resolve.comp" }};
    std::array<VkPipelineLayout*, 2> layouts =
        {{ &m_mipmap_layout, &m_resolve_layout }};
    std::array<VkPipeline*, 2> pipelines =
        {{ &m_mipmap_pipeline, &m_resolve_pipeline }};
    for (unsigned i = 0; i < 2; i++)
    {
        VkPipelineLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        // The mipmap has the push constants, the resolve the camera
        layout_info.setLayoutCount = i == 0 ? 1 : resolve_layouts.size();
        layout_info.pSetLayouts = resolve_layouts.data();
        layout_info.pushConstantRangeCount = i == 0 ? 1 : 0;
        layout_info.pPushConstantRanges = &push_constant;
        if (vkCreatePipelineLayout(m_vk->getDevice(), &layout_info, NULL,
            layouts[i]) != VK_SUCCESS)
        {
            throw std::runtime_error(std::string("vkCreatePipelineLayout "
                "failed for ") + shaders[i]);
        }
        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module =
            GEVulkanShaderManager::getShader(shaders[i]);
        pipeline_info.stage.pName = "main";
        pipeline_info.layout = *layouts[i];
        if (vkCreateComputePipelines(m_vk->getDevice(),
            m_vk->getPipelineCache(), 1, &pipeline_info, NULL, pipelines[i]) !=
            VK_SUCCESS)
        {
            throw std::runtime_error(std::string("vkCreateComputePipelines "
                "failed for ") + shaders[i]);
        }
    }
}   // createPipelines

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::finishTrace(VkCommandBuffer cmd,
                                               GEVulkanDrawCall* dc,
                                               const std::array<int32_t, 4>&
                                               rect)
{
    // The trace wrote level 0
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
        NULL);

    // The 8x8 tiles of level 0 which have the texels of the viewport (the
    // texel t is the pixel 2t), from a multiple of 8 so a tile is a texel of
    // level 3
    MipmapPushConstants pc = {};
    for (unsigned i = 0; i < 4; i++)
        pc.m_rect[i] = rect[i];
    const int32_t first[2] = { rect[0] / 2 / 8 * 8, rect[1] / 2 / 8 * 8 };
    const int32_t end[2] = { (rect[2] + 1) / 2, (rect[3] + 1) / 2 };
    pc.m_origin[0] = first[0];
    pc.m_origin[1] = first[1];
    pc.m_levels = (int32_t)m_trace_levels;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_mipmap_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_mipmap_layout, 0, 1, &m_set, 0, NULL);
    vkCmdPushConstants(cmd, m_mipmap_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(pc), &pc);
    vkCmdDispatch(cmd, groups8(end[0] - first[0]), groups8(end[1] - first[1]),
        1);

    // The mip chain is complete
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
        NULL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_resolve_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_resolve_layout, 0, 1, &m_set, 0, NULL);
    dc->bindDataDescriptorSet(m_vk, cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_resolve_layout, 1);
    // The texels of the list, one thread each
    vkCmdDispatchIndirect(cmd, m_list, g_stochastic_resolve_dispatch);
}   // finishTrace

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::resetList(VkCommandBuffer cmd, bool first)
{
    if (!first)
    {
        // The resolve of the last viewport read the levels of the trace, the
        // mipmap of this one writes the ones of the texels which both
        // viewports share
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }
    else
    {
        // Once per frame, before the trace of the first viewport: the texels
        // which aren't traced are 0 (see utils/stochastic_reflection.glsl).
        // The barrier before (at the start of the frame) is after the resolve
        // of the last frame, the one after the list is before the trace
        clearImage(cmd, m_trace_image, m_trace_levels);
    }
    // No texel and no group of the dispatches (x), the trace has the threads
    // of a texel in y, see utils/stochastic_reflection.glsl
    const std::array<uint32_t, g_stochastic_list_header> header =
        {{ 0, 0, g_stochastic_threads, 1, 0, 1, 1, 0 }};
    vkCmdUpdateBuffer(cmd, m_list, 0, sizeof(header), header.data());
}   // resetList

}
