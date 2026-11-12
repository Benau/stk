#include "ge_vulkan_solid_reflection.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_deferred_fbo.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_hiz_depth.hpp"
#include "ge_vulkan_rt_scene.hpp"
#include "ge_vulkan_shader_manager.hpp"
#include "ge_vulkan_skybox_renderer.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
// Groups of 8x8 threads which cover a size
uint32_t groups8(int32_t size)
{
    return (uint32_t)(std::max(size, 0) + 7) / 8;
}   // groups8

// ----------------------------------------------------------------------------
// Push constants of solid_reflection_classify.comp (and
// stochastic_reflection_classify.comp)
struct ClassifyPushConstants
{
    int32_t m_rect[4];
};

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
GEVulkanSolidReflection::GEVulkanSolidReflection(GEVulkanDriver* vk,
                                   const irr::core::dimension2d<irr::u32>& size,
                                   GEVulkanAttachmentTexture* depth,
                                   GEVulkanAttachmentTexture* normal, bool ssr,
                                   bool rt, bool stochastic)
                       : GEVulkanPostProcessing(vk, size), m_reflection(NULL),
                         m_trace_image(VK_NULL_HANDLE),
                         m_trace_memory(VK_NULL_HANDLE),
                         m_trace_view(VK_NULL_HANDLE), m_trace_levels(0),
                         m_list(VK_NULL_HANDLE), m_list_memory(VK_NULL_HANDLE),
                         m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
                         m_set(VK_NULL_HANDLE),
                         m_classify_layout(VK_NULL_HANDLE),
                         m_classify_pipeline(VK_NULL_HANDLE),
                         m_mipmap_layout(VK_NULL_HANDLE),
                         m_mipmap_pipeline(VK_NULL_HANDLE),
                         m_resolve_layout(VK_NULL_HANDLE),
                         m_resolve_pipeline(VK_NULL_HANDLE),
                         m_has_ssr(ssr), m_has_rt(rt),
                         m_stochastic(stochastic),
                         m_data_layout(VK_NULL_HANDLE), m_options()
{
    m_trace_layouts.fill(VK_NULL_HANDLE);
    m_trace_pipelines.fill(VK_NULL_HANDLE);
    m_trace_level_views.fill(VK_NULL_HANDLE);
    try
    {
        if (m_stochastic)
        {
            // The radiance premultiplied by the coverage needs the alpha, see
            // utils/stochastic_reflection.glsl (rgba16f is a storage format
            // of every device). Cleared once when it's created
            m_reflection = new GEVulkanAttachmentTexture(vk, halfSize(size),
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
            createTraceImage(halfSize(size));
        }
        else
        {
            // The specular radiance is never negative and has no alpha, the
            // same format as the light scattering (the shader is compiled
            // with SHADER_STORAGE_IMAGE_EXTENDED_FORMATS if it can be
            // written)
            m_reflection = new GEVulkanAttachmentTexture(vk, size,
                GEVulkanFeatures::supportsShaderStorageImageExtendedFormats() ?
                VK_FORMAT_B10G11R11_UFLOAT_PACK32 :
                VK_FORMAT_R16G16B16A16_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }
        initializeGeneralImages(&m_reflection, 1);
        // The memory of a new image can have anything (the one of an image
        // which was destroyed when the render target was created again, NaN
        // too), both targets start as 0
        if (m_stochastic)
        {
            VkCommandBuffer cmd =
                GEVulkanCommandLoader::beginSingleTimeCommands();
            clearStochasticImages(cmd, m_reflection->getImage(), 1);
            clearStochasticImages(cmd, m_trace_image, m_trace_levels);
            GEVulkanCommandLoader::endSingleTimeCommands(cmd);
        }
        // The count, the indirect dispatch and every pixel of the FBO (every
        // texel of the half size target if m_stochastic, one viewport at a
        // time uses it)
        const irr::core::dimension2d<irr::u32> list_size =
            m_stochastic ? halfSize(size) : size;
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (!m_vk->createBuffer(sizeof(uint32_t) * ((m_stochastic ?
            g_stochastic_list_header : 4) + list_size.Width * list_size.Height),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT, alloc_info, m_list,
            m_list_memory))
        {
            throw std::runtime_error("createBuffer failed for "
                "GEVulkanSolidReflection");
        }
        if (m_stochastic)
            createStochasticDescriptor(depth, normal);
        else
            createDescriptor(depth, normal);
        createClassifyPipeline();
        if (m_stochastic)
            createStochasticPipelines();
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanSolidReflection

// ----------------------------------------------------------------------------
GEVulkanSolidReflection::~GEVulkanSolidReflection()
{
    destroy();
}   // ~GEVulkanSolidReflection

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::destroy()
{
    m_vk->waitIdle();
    VkDevice device = m_vk->getDevice();
    destroyTracePipelines();
    if (m_data_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_data_layout, NULL);
    m_data_layout = VK_NULL_HANDLE;
    if (m_classify_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_classify_pipeline, NULL);
    if (m_classify_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_classify_layout, NULL);
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
    // The set is freed with its pool
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_pool, NULL);
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_layout, NULL);
    if (m_list != VK_NULL_HANDLE)
        vmaDestroyBuffer(m_vk->getVmaAllocator(), m_list, m_list_memory);
    delete m_reflection;
    m_reflection = NULL;
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
    m_classify_pipeline = VK_NULL_HANDLE;
    m_classify_layout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
    m_list = VK_NULL_HANDLE;
    m_list_memory = VK_NULL_HANDLE;
}   // destroy

// ----------------------------------------------------------------------------
VkImageView GEVulkanSolidReflection::getImageView() const
{
    return (VkImageView)m_reflection->getTextureHandler();
}   // getImageView

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::createDescriptor(GEVulkanAttachmentTexture* depth,
                                              GEVulkanAttachmentTexture* normal)
{
    // 0 = the output (storage image), 1 = the list, 2 = the normal and 3 = the
    // depth of the g-buffer
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, 4> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = i == 0 ?
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : i == 1 ?
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER :
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layout_info, NULL, &m_layout) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorSetLayout failed for "
            "GEVulkanSolidReflection");
    }

    std::array<VkDescriptorPoolSize, 3> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[0].descriptorCount = 1;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[1].descriptorCount = 1;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[2].descriptorCount = 2;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanSolidReflection");
    }

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &m_layout;
    if (vkAllocateDescriptorSets(device, &alloc_info, &m_set) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanSolidReflection");
    }

    std::array<VkDescriptorImageInfo, 3> image_infos = {};
    image_infos[0].imageView = getImageView();
    image_infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[1].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[1].imageView = (VkImageView)normal->getTextureHandler();
    image_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_infos[2].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[2].imageView = (VkImageView)depth->getTextureHandler();
    image_infos[2].imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    VkDescriptorBufferInfo buffer_info = {};
    buffer_info.buffer = m_list;
    buffer_info.range = VK_WHOLE_SIZE;
    std::array<VkWriteDescriptorSet, 4> writes = {};
    for (unsigned i = 0; i < writes.size(); i++)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings[i].descriptorType;
        if (i == 1)
            writes[i].pBufferInfo = &buffer_info;
        else
            writes[i].pImageInfo = &image_infos[i == 0 ? 0 : i - 1];
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, NULL);
}   // createDescriptor

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::createClassifyPipeline()
{
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.size = sizeof(ClassifyPushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &m_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(m_vk->getDevice(), &pipeline_layout_info, NULL,
        &m_classify_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for "
            "GEVulkanSolidReflection");
    }
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = GEVulkanShaderManager::getShader(
        m_stochastic ? "stochastic_reflection_classify.comp" :
        "solid_reflection_classify.comp");
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_classify_layout;
    if (vkCreateComputePipelines(m_vk->getDevice(), m_vk->getPipelineCache(),
        1, &pipeline_info, NULL, &m_classify_pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateComputePipelines failed for "
            "solid_reflection_classify.comp");
    }
}   // createClassifyPipeline

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::createTraceImage(
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
            "GEVulkanSolidReflection");
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
            "GEVulkanSolidReflection");
    }
    view_info.subresourceRange.levelCount = 1;
    for (unsigned i = 0; i < m_trace_levels; i++)
    {
        view_info.subresourceRange.baseMipLevel = i;
        if (vkCreateImageView(m_vk->getDevice(), &view_info, NULL,
            &m_trace_level_views[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed for a level of "
                "the trace of GEVulkanSolidReflection");
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
void GEVulkanSolidReflection::clearStochasticImages(VkCommandBuffer cmd,
                                                    VkImage image,
                                                    unsigned levels)
{
    // The image is in the general layout, which a transfer can write
    VkClearColorValue zero = {};
    VkImageSubresourceRange range = {};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = levels;
    range.layerCount = 1;
    vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1,
        &range);
}   // clearStochasticImages

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::createStochasticDescriptor(
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
            "GEVulkanSolidReflection");
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
            "GEVulkanSolidReflection");
    }

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &m_layout;
    if (vkAllocateDescriptorSets(device, &alloc_info, &m_set) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanSolidReflection");
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
}   // createStochasticDescriptor

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::createStochasticPipelines()
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
}   // createStochasticPipelines

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::dispatchStochasticResolve(VkCommandBuffer cmd,
                                                        GEVulkanDrawCall* dc,
                                                        const int* rect)
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
}   // dispatchStochasticResolve

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::destroyTracePipelines()
{
    for (unsigned i = 0; i < TP_COUNT; i++)
    {
        if (m_trace_pipelines[i] != VK_NULL_HANDLE)
            vkDestroyPipeline(m_vk->getDevice(), m_trace_pipelines[i], NULL);
        if (m_trace_layouts[i] != VK_NULL_HANDLE)
        {
            vkDestroyPipelineLayout(m_vk->getDevice(), m_trace_layouts[i],
                NULL);
        }
        m_trace_pipelines[i] = VK_NULL_HANDLE;
        m_trace_layouts[i] = VK_NULL_HANDLE;
    }
}   // destroyTracePipelines

// ----------------------------------------------------------------------------
bool GEVulkanSolidReflection::prepareTracePipelines(GEVulkanDrawCall* dc)
{
    const GEVulkanPipelineConstants::Options& options =
        dc->getPipelineConstantOptions();
    const bool created = m_trace_layouts[TP_SSR] != VK_NULL_HANDLE ||
        m_trace_layouts[TP_RT] != VK_NULL_HANDLE;
    // The layout of the mesh textures is created again when its pool grows
    if (created && (!(options == m_options) ||
        (m_trace_layouts[TP_RT] != VK_NULL_HANDLE &&
        m_mesh_texture_layout.expired())))
    {
        m_vk->waitIdle();
        destroyTracePipelines();
    }
    if (m_trace_layouts[TP_SSR] == VK_NULL_HANDLE &&
        m_trace_layouts[TP_RT] == VK_NULL_HANDLE)
    {
        m_options = options;
        try
        {
            createTracePipelines(dc);
        }
        catch (const std::exception& e)
        {
            GE::logMessage(GE::GLL_ERROR, "%s", e.what());
            destroyTracePipelines();
            return false;
        }
    }
    return true;
}   // prepareTracePipelines

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::createTracePipelines(GEVulkanDrawCall* dc)
{
    if (m_data_layout == VK_NULL_HANDLE)
        m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    GEVulkanFBOTexture* rtt = m_vk->getRTTTexture();
    // The sets of the fragment shaders which they were before (the lighting
    // pass), and the output and the list, see solid_reflection_ssr.comp and
    // solid_reflection_rt.comp
    std::array<std::vector<VkDescriptorSetLayout>, TP_COUNT> layouts;
    if (m_has_ssr)
    {
        layouts[TP_SSR] = rtt->getAllDescriptorSetLayouts(m_data_layout,
            GVDFP_SOLID_SSR);
    }
    std::shared_ptr<VkDescriptorSetLayout> mesh_texture_layout;
    if (m_has_rt && m_vk->getRTScene())
    {
        mesh_texture_layout =
            m_vk->getMeshTextureDescriptor()->getRTDescriptorSetLayout();
        m_mesh_texture_layout = mesh_texture_layout;
        layouts[TP_RT] =
        {
            m_vk->getRTScene()->getDescriptorSetLayout(),
            m_data_layout,
            rtt->getDescriptorSetLayout(GVDFP_HDR),
            m_vk->getSkyBoxRenderer()->getEnvDescriptorSetLayout(),
            *mesh_texture_layout
        };
    }
    const std::array<const char*, TP_COUNT> shaders = m_stochastic ?
        std::array<const char*, TP_COUNT>
        {{ "stochastic_reflection_ssr.comp",
        "stochastic_reflection_rt.comp" }} :
        std::array<const char*, TP_COUNT>
        {{ "solid_reflection_ssr.comp", "solid_reflection_rt.comp" }};
    // The same specialization constants as the graphics pipelines of the draw
    // call (the image based lighting, the shadows and the iterations of the
    // HiZ trace)
    GEVulkanPipelineConstants constants(m_vk, m_options);
    for (unsigned i = 0; i < TP_COUNT; i++)
    {
        if (layouts[i].empty())
            continue;
        layouts[i].push_back(m_layout);
        VkPipelineLayoutCreateInfo layout_info = {};
        layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        layout_info.setLayoutCount = layouts[i].size();
        layout_info.pSetLayouts = layouts[i].data();
        if (vkCreatePipelineLayout(m_vk->getDevice(), &layout_info, NULL,
            &m_trace_layouts[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreatePipelineLayout failed for "
                "the trace of GEVulkanSolidReflection");
        }
        VkComputePipelineCreateInfo pipeline_info = {};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeline_info.stage.module =
            GEVulkanShaderManager::getShader(shaders[i]);
        pipeline_info.stage.pName = "main";
        pipeline_info.stage.pSpecializationInfo = constants.getInfo();
        pipeline_info.layout = m_trace_layouts[i];
        if (vkCreateComputePipelines(m_vk->getDevice(),
            m_vk->getPipelineCache(), 1, &pipeline_info, NULL,
            &m_trace_pipelines[i]) != VK_SUCCESS)
        {
            throw std::runtime_error(std::string("vkCreateComputePipelines "
                "failed for ") + shaders[i]);
        }
    }
}   // createTracePipelines

// ----------------------------------------------------------------------------
const VkDescriptorSet* GEVulkanSolidReflection::getSSRDescriptorSet(
                                                   GEVulkanDrawCall* dc) const
{
    GEVulkanHiZDepth* hiz = dc->getHiZDepth();
    return hiz ? hiz->getRenderingDescriptorSet() :
        m_vk->getRTTTexture()->getDescriptorSet(GVDFP_SOLID_SSR);
}   // getSSRDescriptorSet

// ----------------------------------------------------------------------------
bool GEVulkanSolidReflection::canTrace(GEVulkanDrawCall* dc, bool rt) const
{
    if (m_trace_pipelines[rt ? TP_RT : TP_SSR] == VK_NULL_HANDLE ||
        !dc->hasDeferredOutput())
        return false;
    if (rt)
    {
        GEVulkanRTScene* rt_scene = m_vk->getRTScene();
        return rt_scene && rt_scene->isReady();
    }
    const VkDescriptorSet* set = getSSRDescriptorSet(dc);
    return set != NULL && *set != VK_NULL_HANDLE;
}   // canTrace

// ----------------------------------------------------------------------------
void GEVulkanSolidReflection::dispatchTrace(VkCommandBuffer cmd,
                                            GEVulkanDrawCall* dc, bool rt)
{
    VkPipelineLayout layout = m_trace_layouts[rt ? TP_RT : TP_SSR];
    unsigned set_index = 0;
    if (rt)
    {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
            1, m_vk->getRTScene()->getDescriptorSet(
            m_vk->getCurrentBufferIdx()), 0, NULL);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 4,
            1, m_vk->getMeshTextureDescriptor()->getRTDescriptorSet(), 0,
            NULL);
        set_index = 5;
    }
    else
    {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0,
            1, getSSRDescriptorSet(dc), 0, NULL);
        set_index = 4;
    }
    // Camera, g-buffer (the normal has the gloss), the specular IBL (the same
    // as the lighting pass has in its lit color) and the output and the list
    dc->bindDataDescriptorSet(m_vk, cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        layout, 1);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 2, 1,
        m_vk->getRTTTexture()->getDescriptorSet(GVDFP_HDR), 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 3, 1,
        dc->getEnvDescriptorSet(m_vk), 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout,
        set_index, 1, &m_set, 0, NULL);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_trace_pipelines[rt ? TP_RT : TP_SSR]);
    // The count of the groups is after the count of the pixels (the same
    // offset for the stochastic reflection, the dispatch of its trace)
    vkCmdDispatchIndirect(cmd, m_list, sizeof(uint32_t));
}   // dispatchTrace

// ----------------------------------------------------------------------------
bool GEVulkanSolidReflection::isTraced(GEVulkanDrawCall* dc) const
{
    return std::find(m_traced.begin(), m_traced.end(), dc) != m_traced.end();
}   // isTraced

// ----------------------------------------------------------------------------
bool GEVulkanSolidReflection::render(VkCommandBuffer cmd, bool rt, bool ssr,
                                     const std::vector<std::pair<
                                     GEVulkanDrawCall*, GEVulkanCamera*> >& p)
{
    m_traced.clear();
    if (!rt && !ssr)
        return false;
    // The g-buffer is written, the tonemap pass of the last frame wrote the
    // history of the screen space reflection, the lighting pass of the last
    // frame read the output, and the trace of the last viewport (or frame)
    // read the list (indirect dispatch too)
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    }

    // With the options of the first viewport, the same as its environment
    bool prepared = false;
    for (auto& q : p)
    {
        Viewport vp;
        if (getViewport(q.first, q.second, &vp))
        {
            prepared = prepareTracePipelines(q.first);
            break;
        }
    }

    bool traced = false;
    bool first = true;
    for (auto& q : p)
    {
        Viewport vp;
        if (!getViewport(q.first, q.second, &vp))
            continue;
        // The lighting pass keeps the specular IBL of the viewports which
        // aren't traced (isTraced)
        if (!prepared || !canTrace(q.first, rt))
            continue;
        // The trace of the last viewport read the list
        if (!first)
        {
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0,
                NULL);
            // The resolve of the last viewport read the levels of the
            // trace, the mipmap of this one writes the ones of the texels
            // which both viewports share
            if (m_stochastic)
            {
                barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                    VK_ACCESS_SHADER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0,
                    NULL, 0, NULL);
            }
        }
        // Once per frame, before the trace of the first viewport: the
        // texels which aren't traced are 0 (see
        // utils/stochastic_reflection.glsl). The barrier before (above or at
        // the start of the frame) is after the resolve of the last frame,
        // the one after the list is before the trace
        if (first && m_stochastic)
            clearStochasticImages(cmd, m_trace_image, m_trace_levels);
        first = false;
        if (m_stochastic)
        {
            // No texel and no group of the dispatches (x), the trace has the
            // threads of a texel in y, see utils/stochastic_reflection.glsl
            const std::array<uint32_t, g_stochastic_list_header> header =
                {{ 0, 0, g_stochastic_threads, 1, 0, 1, 1, 0 }};
            vkCmdUpdateBuffer(cmd, m_list, 0, sizeof(header), header.data());
        }
        else
        {
            // No pixel and no group of the dispatch (x), it has y and z of 1
            vkCmdFillBuffer(cmd, m_list, 0, 2 * sizeof(uint32_t), 0);
            vkCmdFillBuffer(cmd, m_list, 2 * sizeof(uint32_t),
                2 * sizeof(uint32_t), 1);
        }
        {
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL,
                0, NULL);
        }

        ClassifyPushConstants pc = {};
        for (unsigned i = 0; i < 4; i++)
            pc.m_rect[i] = vp.m_rect[i];
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_classify_pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_classify_layout, 0, 1, &m_set, 0, NULL);
        vkCmdPushConstants(cmd, m_classify_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pc), &pc);
        if (m_stochastic)
        {
            // The texels of the half size target whose pixel can be in the
            // rect (the texel t is the pixel 2t)
            vkCmdDispatch(cmd,
                groups8((vp.m_rect[2] + 1) / 2 - vp.m_rect[0] / 2),
                groups8((vp.m_rect[3] + 1) / 2 - vp.m_rect[1] / 2), 1);
        }
        else
        {
            vkCmdDispatch(cmd, groups8(vp.m_rect[2] - vp.m_rect[0]),
                groups8(vp.m_rect[3] - vp.m_rect[1]), 1);
        }

        // The list is complete, and its size is the indirect dispatch
        {
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &barrier, 0, NULL,
                0, NULL);
        }
        dispatchTrace(cmd, q.first, rt);
        if (m_stochastic)
            dispatchStochasticResolve(cmd, q.first, vp.m_rect.data());
        m_traced.push_back(q.first);
        traced = true;
    }

    // Sampled by the lighting pass
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
        NULL);
    return traced;
}   // render

}
