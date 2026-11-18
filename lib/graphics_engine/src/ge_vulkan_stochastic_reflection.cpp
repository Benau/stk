#include "ge_vulkan_stochastic_reflection.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
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
// Push constants of stochastic_reflection_temporal.comp
struct TemporalPushConstants
{
    uint32_t m_viewport_index;
    uint32_t m_history;
};

// ----------------------------------------------------------------------------
// The viewports of a frame which have the temporal filter, the other ones have
// no history (more than any splitscreen has)
const unsigned g_stochastic_max_viewports = 8;

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
    SB_RAY_LENGTH,
    SB_COUNT
};

// ----------------------------------------------------------------------------
// The bindings of the set of stochastic_reflection_temporal.comp
enum TemporalBinding : unsigned
{
    TB_CURRENT = 0,
    TB_HISTORY_IN,
    TB_HISTORY_SURFACE_IN,
    TB_HISTORY_OUT,
    TB_HISTORY_SURFACE_OUT,
    TB_PREV_CAMERAS,
    TB_COUNT
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
                              m_ray_length(NULL), m_current(NULL),
                              m_prev_camera_buffer(VK_NULL_HANDLE),
                              m_prev_camera_memory(VK_NULL_HANDLE),
                              m_resolve_set(VK_NULL_HANDLE),
                              m_temporal_layout(VK_NULL_HANDLE),
                              m_temporal_pool(VK_NULL_HANDLE),
                              m_mipmap_layout(VK_NULL_HANDLE),
                              m_mipmap_pipeline(VK_NULL_HANDLE),
                              m_resolve_layout(VK_NULL_HANDLE),
                              m_resolve_pipeline(VK_NULL_HANDLE),
                              m_temporal_pipeline_layout(VK_NULL_HANDLE),
                              m_temporal_pipeline(VK_NULL_HANDLE), m_frame(0),
                              m_history_index(0), m_temporal(true),
                              m_viewport_index(0), m_viewport_history(false)
{
    m_trace_level_views.fill(VK_NULL_HANDLE);
    m_history.fill(NULL);
    m_history_surface.fill(NULL);
    m_temporal_sets.fill(VK_NULL_HANDLE);
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
        // The targets of the temporal filter, created whatever the frame cache
        // is (it can be changed at any time)
        const VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        m_ray_length = new GEVulkanAttachmentTexture(vk, half,
            VK_FORMAT_R32_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT);
        m_current = new GEVulkanAttachmentTexture(vk, half,
            VK_FORMAT_R16G16B16A16_SFLOAT, usage, VK_IMAGE_ASPECT_COLOR_BIT);
        for (unsigned i = 0; i < 2; i++)
        {
            m_history[i] = new GEVulkanAttachmentTexture(vk, half,
                VK_FORMAT_R16G16B16A16_SFLOAT, usage,
                VK_IMAGE_ASPECT_COLOR_BIT);
            m_history_surface[i] = new GEVulkanAttachmentTexture(vk, half,
                VK_FORMAT_R32G32_UINT, usage, VK_IMAGE_ASPECT_COLOR_BIT);
        }
        std::array<GEVulkanAttachmentTexture*, 7> images =
        {{
            m_reflection, m_ray_length, m_current, m_history[0], m_history[1],
            m_history_surface[0], m_history_surface[1]
        }};
        initializeGeneralImages(images.data(), images.size());
        // The memory of a new image can have anything (the one of an image
        // which was destroyed when the render target was created again, NaN
        // too), all the targets start as 0 (no history)
        VkCommandBuffer cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
        for (GEVulkanAttachmentTexture* image : images)
            clearImage(cmd, image->getImage(), 1);
        clearImage(cmd, m_trace_image, m_trace_levels);
        GEVulkanCommandLoader::endSingleTimeCommands(cmd);
        // The header and every texel of the half size target
        createList(half, g_stochastic_list_header);
        createDescriptor(depth, normal);
        createTemporalDescriptors(half);
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
    if (m_temporal_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_temporal_pipeline, NULL);
    if (m_temporal_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_temporal_pipeline_layout, NULL);
    m_mipmap_pipeline = VK_NULL_HANDLE;
    m_mipmap_layout = VK_NULL_HANDLE;
    m_resolve_pipeline = VK_NULL_HANDLE;
    m_resolve_layout = VK_NULL_HANDLE;
    m_temporal_pipeline = VK_NULL_HANDLE;
    m_temporal_pipeline_layout = VK_NULL_HANDLE;
    // The sets are freed with their pool
    if (m_temporal_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_temporal_pool, NULL);
    if (m_temporal_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_temporal_layout, NULL);
    m_temporal_pool = VK_NULL_HANDLE;
    m_temporal_layout = VK_NULL_HANDLE;
    m_temporal_sets.fill(VK_NULL_HANDLE);
    if (m_prev_camera_buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_vk->getVmaAllocator(), m_prev_camera_buffer,
            m_prev_camera_memory);
    }
    m_prev_camera_buffer = VK_NULL_HANDLE;
    m_prev_camera_memory = VK_NULL_HANDLE;
    delete m_ray_length;
    delete m_current;
    m_ray_length = NULL;
    m_current = NULL;
    for (unsigned i = 0; i < 2; i++)
    {
        delete m_history[i];
        delete m_history_surface[i];
        m_history[i] = NULL;
        m_history_surface[i] = NULL;
    }
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
std::vector<std::pair<uint32_t, VkDescriptorType> >
    GEVulkanStochasticReflection::getTraceOutputBindings() const
{
    // The trace target and the list are SB_TRACE and SB_LIST, the ones of
    // GEVulkanSolidReflection, see stochastic_reflection_ssr.comp
    std::vector<std::pair<uint32_t, VkDescriptorType> > bindings =
        GEVulkanSolidReflection::getTraceOutputBindings();
    bindings.emplace_back(SB_RAY_LENGTH, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    return bindings;
}   // getTraceOutputBindings

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
    bindings[SB_RAY_LENGTH].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
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

    // m_set and m_resolve_set
    std::array<VkDescriptorPoolSize, 3> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = (3 + g_stochastic_levels) * 2;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = 2;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[2].descriptorCount = 3 * 2;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 2;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanStochasticReflection");
    }

    const std::array<VkDescriptorSetLayout, 2> layouts =
        {{ m_layout, m_layout }};
    std::array<VkDescriptorSet, 2> sets = {};
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = sets.size();
    alloc_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &alloc_info, sets.data()) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanStochasticReflection");
    }
    m_set = sets[0];
    m_resolve_set = sets[1];

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
    // The output of the resolve is the output of the lighting pass, or
    // m_current which the temporal filter reads
    std::array<VkDescriptorImageInfo, 2> resolved_infos = {};
    resolved_infos[0].imageView = getImageView();
    resolved_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    resolved_infos[1].imageView = (VkImageView)m_current->getTextureHandler();
    resolved_infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkDescriptorImageInfo ray_length_info = {};
    ray_length_info.imageView = (VkImageView)m_ray_length->getTextureHandler();
    ray_length_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    for (unsigned set = 0; set < sets.size(); set++)
    {
        const std::array<const VkDescriptorImageInfo*, SB_COUNT> image_infos =
        {{
            &trace_info, NULL, &normal_info, &depth_info, level_infos.data(),
            &sampled_info, &resolved_infos[set], &ray_length_info
        }};
        std::array<VkWriteDescriptorSet, SB_COUNT> writes = {};
        for (unsigned i = 0; i < writes.size(); i++)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = sets[set];
            writes[i].dstBinding = i;
            writes[i].descriptorCount = bindings[i].descriptorCount;
            writes[i].descriptorType = bindings[i].descriptorType;
            if (i == SB_LIST)
                writes[i].pBufferInfo = &buffer_info;
            else
                writes[i].pImageInfo = image_infos[i];
        }
        vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0,
            NULL);
    }
}   // createDescriptor

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::createTemporalDescriptors(
                                   const irr::core::dimension2d<irr::u32>& size)
{
    VmaAllocationCreateInfo buffer_alloc = {};
    buffer_alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    if (!m_vk->createBuffer(sizeof(GEVulkanPrevCamera) *
        g_stochastic_max_viewports, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, buffer_alloc, m_prev_camera_buffer,
        m_prev_camera_memory))
    {
        throw std::runtime_error("createBuffer failed for the cameras of "
            "GEVulkanStochasticReflection");
    }

    // See TemporalBinding
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, TB_COUNT> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = i == TB_PREV_CAMERAS ?
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER :
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layout_info, NULL,
        &m_temporal_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorSetLayout failed for "
            "the temporal filter of GEVulkanStochasticReflection");
    }

    std::array<VkDescriptorPoolSize, 2> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = (TB_COUNT - 1) * m_temporal_sets.size();
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = m_temporal_sets.size();
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = m_temporal_sets.size();
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_temporal_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for the "
            "temporal filter of GEVulkanStochasticReflection");
    }

    const std::array<VkDescriptorSetLayout, 2> layouts =
        {{ m_temporal_layout, m_temporal_layout }};
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_temporal_pool;
    alloc_info.descriptorSetCount = m_temporal_sets.size();
    alloc_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &alloc_info, m_temporal_sets.data()) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for the "
            "temporal filter of GEVulkanStochasticReflection");
    }

    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = m_prev_camera_buffer;
    buffer_info.range = VK_WHOLE_SIZE;
    for (unsigned set = 0; set < m_temporal_sets.size(); set++)
    {
        // Read the history of this set, write the other one
        const unsigned other = (set + 1) % m_temporal_sets.size();
        const std::array<GEVulkanAttachmentTexture*, TB_COUNT - 1> images =
        {{
            m_current, m_history[set], m_history_surface[set],
            m_history[other], m_history_surface[other]
        }};
        std::array<VkDescriptorImageInfo, TB_COUNT - 1> image_infos = {};
        std::array<VkWriteDescriptorSet, TB_COUNT> writes = {};
        for (unsigned i = 0; i < writes.size(); i++)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = m_temporal_sets[set];
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = bindings[i].descriptorType;
            if (i == TB_PREV_CAMERAS)
            {
                writes[i].pBufferInfo = &buffer_info;
                continue;
            }
            image_infos[i].imageView =
                (VkImageView)images[i]->getTextureHandler();
            image_infos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            writes[i].pImageInfo = &image_infos[i];
        }
        vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0,
            NULL);
    }
}   // createTemporalDescriptors

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
    VkPushConstantRange temporal_push_constant = {};
    temporal_push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    temporal_push_constant.size = sizeof(TemporalPushConstants);
    const std::array<VkDescriptorSetLayout, 3> set_layouts =
        {{ m_layout, m_data_layout, m_temporal_layout }};
    const std::array<const char*, 3> shaders =
        {{ "stochastic_reflection_mipmap.comp",
        "stochastic_reflection_resolve.comp",
        "stochastic_reflection_temporal.comp" }};
    std::array<VkPipelineLayout*, 3> layouts =
        {{ &m_mipmap_layout, &m_resolve_layout, &m_temporal_pipeline_layout }};
    std::array<VkPipeline*, 3> pipelines =
        {{ &m_mipmap_pipeline, &m_resolve_pipeline, &m_temporal_pipeline }};
    for (unsigned i = 0; i < shaders.size(); i++)
    {
        VkPipelineLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        // The mipmap has the push constants, the resolve the camera, the
        // temporal filter both and its set
        layout_info.setLayoutCount = i + 1;
        layout_info.pSetLayouts = set_layouts.data();
        layout_info.pushConstantRangeCount = i == 1 ? 0 : 1;
        layout_info.pPushConstantRanges = i == 0 ? &push_constant :
            &temporal_push_constant;
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
                                               const Viewport& vp,
                                               GEVulkanCamera* cam)
{
    GEVulkanDrawCall* dc = vp.m_draw_call;
    const std::array<int32_t, 4>& rect = vp.m_rect;
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
    // Into m_current with the temporal filter
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_resolve_layout, 0, 1, m_temporal ? &m_resolve_set : &m_set, 0,
        NULL);
    dc->bindDataDescriptorSet(m_vk, cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_resolve_layout, 1);
    // The texels of the list, one thread each
    vkCmdDispatchIndirect(cmd, m_list, g_stochastic_resolve_dispatch);
    if (!m_temporal)
        return;

    // The resolve wrote m_current, the temporal filter reads the texels
    // around, and writes the output and the history
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
        NULL);
    TemporalPushConstants temporal_pc = {};
    temporal_pc.m_viewport_index = m_viewport_index;
    temporal_pc.m_history = m_viewport_history ? 1 : 0;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_temporal_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_temporal_pipeline_layout, 0, 1, &m_set, 0, NULL);
    dc->bindDataDescriptorSet(m_vk, cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_temporal_pipeline_layout, 1);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_temporal_pipeline_layout, 2, 1, &m_temporal_sets[m_history_index],
        0, NULL);
    vkCmdPushConstants(cmd, m_temporal_pipeline_layout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(temporal_pc), &temporal_pc);
    vkCmdDispatchIndirect(cmd, m_list, g_stochastic_resolve_dispatch);

    // The next frame reads the history of the viewport with its camera of
    // this frame
    ViewportHistory history;
    history.m_camera = cam;
    history.m_rect = rect;
    history.m_prev_camera.set(cam->getUBOData());
    m_viewports.push_back(history);
    m_viewport_index++;
}   // finishTrace

// ----------------------------------------------------------------------------
void GEVulkanStochasticReflection::resetList(VkCommandBuffer cmd, bool first,
                                             const Viewport& vp,
                                             GEVulkanCamera* cam)
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
        // The history written by the last frame is read by this one, and the
        // cameras of its viewports are the ones it was rendered with
        if (!m_viewports.empty())
            m_history_index = (m_history_index + 1) % m_temporal_sets.size();
        m_prev_viewports.swap(m_viewports);
        m_viewports.clear();
        m_viewport_index = 0;
        if (m_temporal)
        {
            // Never 0 (no temporal filter)
            if (++m_frame == 0)
                m_frame = 1;
            // The texels which aren't written in this frame can't be reused
            // by the next one (the history of the surface is 0)
            clearImage(cmd, m_history_surface[(m_history_index + 1) %
                m_temporal_sets.size()]->getImage(), 1);
        }
        else
            m_prev_viewports.clear();
    }
    // The history of the viewport is the one of its camera in the last frame,
    // if it had the same rect (with as many viewports, the one which was
    // filtered at the same index)
    m_viewport_history = false;
    if (m_temporal && m_viewport_index < g_stochastic_max_viewports)
    {
        for (const ViewportHistory& prev : m_prev_viewports)
        {
            if (prev.m_camera != cam || prev.m_rect != vp.m_rect)
                continue;
            m_viewport_history = true;
            vkCmdUpdateBuffer(cmd, m_prev_camera_buffer,
                sizeof(GEVulkanPrevCamera) * m_viewport_index,
                sizeof(GEVulkanPrevCamera), &prev.m_prev_camera);
            break;
        }
    }
    // No texel and no group of the dispatches (x), the trace has the threads
    // of a texel in y, see utils/stochastic_reflection.glsl, and the frame of
    // the temporal filter
    const std::array<uint32_t, g_stochastic_list_header> header =
        {{ 0, 0, g_stochastic_threads, 1, 0, 1, 1, m_temporal ? m_frame : 0 }};
    vkCmdUpdateBuffer(cmd, m_list, 0, sizeof(header), header.data());
}   // resetList

}
