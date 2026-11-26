#include "ge_vulkan_surfel_gi.hpp"
#include "ge_vulkan_features.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_environment_map.hpp"
#include "ge_vulkan_rt_scene.hpp"
#include "ge_vulkan_shader_manager.hpp"
#include "ge_vulkan_skybox_renderer.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
// Groups of 64 threads which cover count items
uint32_t groups64(uint32_t count)
{
    return (count + 63) / 64;
}   // groups64

// ----------------------------------------------------------------------------
// Groups of 8x8 threads which cover a size
uint32_t groups8(int32_t size)
{
    return (uint32_t)(std::max(size, 0) + 7) / 8;
}   // groups8

unsigned probeLevels(const GEVulkanCamera& camera)
{
    using irr::core::vector3df;
    using irr::core::aabbox3df;
    const auto* frustum = camera.getViewFrustum();
    const vector3df eye = camera.getAbsolutePosition();
    const float radius = std::sqrt(std::max({eye.getDistanceFromSQ(frustum->getFarLeftDown()),
        eye.getDistanceFromSQ(frustum->getFarRightDown()), eye.getDistanceFromSQ(frustum->getFarLeftUp()),
        eye.getDistanceFromSQ(frustum->getFarRightUp())}));
    const vector3df extent(radius, radius, radius);
    const aabbox3df sphere(eye - extent, eye + extent);
    const auto* scene = camera.getSceneBounds();
    float spacing = 2.0f;
    for (unsigned levels = 1; ; ++levels, spacing *= 4.0f)
    {
        const vector3df cell(std::floor(eye.X / spacing), std::floor(eye.Y / spacing), std::floor(eye.Z / spacing));
        const vector3df half(7.4f, 3.4f, 7.4f);
        const aabbox3df interior((cell - half) * spacing, (cell + half) * spacing);
        if ((scene && scene->isFullInside(interior)) || sphere.isFullInside(interior) || !std::isfinite(radius))
            return levels;
    }
}

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanSurfelGI::GEVulkanSurfelGI(GEVulkanDriver* vk,
                                   const irr::core::dimension2d<irr::u32>& size,
                                   GEVulkanAttachmentTexture* depth,
                                   GEVulkanAttachmentTexture* normal)
                : GEVulkanPostProcessing(vk, size), m_buffers(), m_gi(NULL),
                  m_contact(NULL),
                  m_representatives(NULL), m_layout(VK_NULL_HANDLE),
                  m_pool(VK_NULL_HANDLE), m_set(VK_NULL_HANDLE),
                  m_data_layout(VK_NULL_HANDLE),
                  m_pipeline_layout(VK_NULL_HANDLE), m_options(),
                  m_pipelines(), m_frame(0), m_time(0), m_last_render(0),
                  m_rt_build_count(0), m_needs_reset(true), m_surfel_count(0),
                  m_cleared(false)
{
    for (Buffer& b : m_buffers)
    {
        b.m_buffer = VK_NULL_HANDLE;
        b.m_memory = VK_NULL_HANDLE;
    }
    m_pipelines.fill(VK_NULL_HANDLE);
    m_depth_step = depth->getInternalFormat() == VK_FORMAT_D16_UNORM ? 1.0f / 65535.0f :
        depth->getInternalFormat() == VK_FORMAT_D24_UNORM_S8_UINT ? 1.0f / 16777215.0f : 0.0f;
    if (!m_vk->getRTScene())
        throw std::runtime_error("No ray traced scene for GEVulkanSurfelGI");
    try
    {
        m_gi = new GEVulkanAttachmentTexture(vk, size,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        const VkFormat contact_format = GEVulkanFeatures::supportsShaderStorageImageExtendedFormats() ?
            VK_FORMAT_R16G16_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        m_contact = new GEVulkanAttachmentTexture(vk, size, contact_format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        initializeGeneralImages(&m_gi, 1);
        initializeGeneralImages(&m_contact, 1);
        m_representatives = new GEVulkanAttachmentTexture(vk, halfSize(size),
            VK_FORMAT_R32G32B32A32_UINT, VK_IMAGE_USAGE_STORAGE_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        initializeGeneralImages(&m_representatives, 1);
        createBuffers();
        VkCommandBuffer init = GEVulkanCommandLoader::beginSingleTimeCommands();
        vkCmdFillBuffer(init, getProbeField(), 0, 80, 0);
        GEVulkanCommandLoader::endSingleTimeCommands(init);
        createDescriptors(depth, normal);
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanSurfelGI

// ----------------------------------------------------------------------------
GEVulkanSurfelGI::~GEVulkanSurfelGI()
{
    destroy();
}   // ~GEVulkanSurfelGI

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::destroyPipelines()
{
    VkDevice device = m_vk->getDevice();
    for (VkPipeline& pipeline : m_pipelines)
    {
        if (pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device, pipeline, NULL);
        pipeline = VK_NULL_HANDLE;
    }
    if (m_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    m_pipeline_layout = VK_NULL_HANDLE;
}   // destroyPipelines

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::destroy()
{
    m_vk->waitIdle();
    VkDevice device = m_vk->getDevice();
    destroyPipelines();
    if (m_data_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_data_layout, NULL);
    // The set is freed with its pool
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_pool, NULL);
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_layout, NULL);
    for (Buffer& b : m_buffers)
    {
        if (b.m_buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(m_vk->getVmaAllocator(), b.m_buffer, b.m_memory);
        b.m_buffer = VK_NULL_HANDLE;
        b.m_memory = VK_NULL_HANDLE;
    }
    delete m_gi;
    m_gi = NULL;
    delete m_contact;
    m_contact = NULL;
    delete m_representatives;
    m_representatives = NULL;
    m_data_layout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
}   // destroy

// ----------------------------------------------------------------------------
VkImageView GEVulkanSurfelGI::getImageView() const
{
    return (VkImageView)m_gi->getTextureHandler();
}   // getImageView

VkImageView GEVulkanSurfelGI::getContactView() const
{
    return (VkImageView)m_contact->getTextureHandler();
}

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createBuffers()
{
    const std::array<VkDeviceSize, BI_COUNT> sizes =
    {{
        sizeof(Surfel) * MAX_SURFELS,
        sizeof(int32_t) + sizeof(uint32_t) * MAX_SURFELS,
        sizeof(uint32_t) * CELL_COUNT,
        sizeof(uint32_t) * (CELL_COUNT * (CELL_CAPACITY + 1) + 2 * MAX_SURFELS),
        sizeof(uint32_t) * CELL_COUNT,
        sizeof(uint32_t) * (4 + MAX_SURFELS),
        sizeof(uint32_t) * (5 * TRACE_LIST_COUNT +
            TRACE_LIST_COUNT * MAX_SURFELS),
        80 + sizeof(Probe) * PROBE_LEVEL_SIZE * m_probe_layers,
        sizeof(uint32_t) * (8 + PRIMARY_RAY_BUDGET * (4 + 8) + MAX_CAMERAS * PROBE_BANDS * (1 + PROBE_BATCH)),
        64 * MAX_SURFELS
    }};
    for (unsigned i = 0; i < BI_COUNT; i++)
    {
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (i == BI_PROBES)
            usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (i == BI_TRACE_LIST || i == BI_SPAWN_QUEUE || i == BI_RAY_WORK)
            usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (!m_vk->createBuffer(sizes[i], usage, alloc_info,
            m_buffers[i].m_buffer, m_buffers[i].m_memory))
        {
            throw std::runtime_error("createBuffer failed for "
                "GEVulkanSurfelGI");
        }
    }
}   // createBuffers

void GEVulkanSurfelGI::prepareProbeField(const std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCamera*> >& views)
{
    unsigned cameras = 0, levels = 1;
    for (const auto& view : views)
    {
        Viewport viewport;
        if (!getViewport(view.first, view.second, &viewport))
            continue;
        m_probe_levels[cameras] = probeLevels(*view.second);
        levels = std::max(levels, m_probe_levels[cameras]);
        if (++cameras == MAX_CAMERAS)
            break;
    }
    const VkDeviceSize layer_size = sizeof(Probe) * PROBE_LEVEL_SIZE;
    const auto& limits = m_vk->getPhysicalDeviceProperties().limits;
    const unsigned capacity = unsigned((limits.maxStorageBufferRange - 80) / layer_size);
    const unsigned layers = std::min(cameras * levels, capacity);
    if (layers <= m_probe_layers)
        return;
    VmaAllocationCreateInfo allocation = {};
    allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    Buffer field = {};
    if (!m_vk->createBuffer(80 + layer_size * layers, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, allocation, field.m_buffer, field.m_memory))
        return;
    m_vk->waitIdle();
    VkCommandBuffer cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
    VkMemoryBarrier readable = {};
    readable.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    readable.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    readable.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &readable, 0, nullptr, 0, nullptr);
    VkBufferCopy copy = {0, 0, 80 + layer_size * m_probe_layers};
    vkCmdCopyBuffer(cmd, getProbeField(), field.m_buffer, 1, &copy);
    vkCmdFillBuffer(cmd, field.m_buffer, copy.size, VK_WHOLE_SIZE, 0);
    VkMemoryBarrier ready = {};
    ready.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    ready.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    ready.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &ready, 0, nullptr, 0, nullptr);
    GEVulkanCommandLoader::endSingleTimeCommands(cmd);
    VkDescriptorBufferInfo info = {field.m_buffer, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_set;
    write.dstBinding = 3 + BI_PROBES;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.descriptorCount = 1;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(m_vk->getDevice(), 1, &write, 0, nullptr);
    std::swap(field, m_buffers[BI_PROBES]);
    vmaDestroyBuffer(m_vk->getVmaAllocator(), field.m_buffer, field.m_memory);
    m_probe_layers = layers;
}

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createDescriptors(GEVulkanAttachmentTexture* depth,
                                         GEVulkanAttachmentTexture* normal)
{
    const unsigned binding_count = 3 + BI_COUNT + 2;
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, binding_count> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = i < 2 ?
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
            i == 2 || i >= 3 + BI_COUNT ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
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
            "GEVulkanSurfelGI");
    }

    std::array<VkDescriptorPoolSize, 3> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount = 2;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = 3;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[2].descriptorCount = BI_COUNT;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanSurfelGI");
    }

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &m_layout;
    if (vkAllocateDescriptorSets(device, &alloc_info, &m_set) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanSurfelGI");
    }

    std::array<VkDescriptorImageInfo, 5> image_infos = {};
    image_infos[0].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[0].imageView = (VkImageView)depth->getTextureHandler();
    image_infos[0].imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    image_infos[1].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[1].imageView = (VkImageView)normal->getTextureHandler();
    image_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_infos[2].imageView = getContactView();
    image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[3].imageView = (VkImageView)m_representatives->getTextureHandler();
    image_infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[4].imageView = getImageView();
    image_infos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    std::array<VkDescriptorBufferInfo, BI_COUNT> buffer_infos = {};
    for (unsigned i = 0; i < BI_COUNT; i++)
    {
        buffer_infos[i].buffer = m_buffers[i].m_buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = VK_WHOLE_SIZE;
    }
    std::array<VkWriteDescriptorSet, binding_count> writes = {};
    for (unsigned i = 0; i < writes.size(); i++)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings[i].descriptorType;
        if (i < 3)
            writes[i].pImageInfo = &image_infos[i];
        else if (i < 3 + BI_COUNT)
            writes[i].pBufferInfo = &buffer_infos[i - 3];
        else
            writes[i].pImageInfo = &image_infos[i - BI_COUNT];
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, NULL);

}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createPipeline(const char* shader, VkPipeline* pipeline,
                              const VkSpecializationInfo* specialization) const
{
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = GEVulkanShaderManager::getShader(shader);
    pipeline_info.stage.pName = "main";
    pipeline_info.stage.pSpecializationInfo = specialization;
    pipeline_info.layout = m_pipeline_layout;
    if (vkCreateComputePipelines(m_vk->getDevice(), m_vk->getPipelineCache(),
        1, &pipeline_info, NULL, pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkCreateComputePipelines "
            "failed for GEVulkanSurfelGI ") + shader);
    }
}   // createPipeline

// ----------------------------------------------------------------------------
bool GEVulkanSurfelGI::isFrameCache()
{
    return getGEConfig()->m_frame_cache;
}   // isFrameCache

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createPipelines()
{
    if (m_data_layout == VK_NULL_HANDLE)
        m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    std::shared_ptr<VkDescriptorSetLayout> mesh_texture_layout =
        m_vk->getMeshTextureDescriptor()->getRTDescriptorSetLayout();
    m_mesh_texture_layout = mesh_texture_layout;
    std::array<VkDescriptorSetLayout, 5> layouts =
    {{
        m_vk->getRTScene()->getDescriptorSetLayout(),
        m_data_layout,
        m_layout,
        *mesh_texture_layout,
        m_vk->getSkyBoxRenderer()->getEnvDescriptorSetLayout()
    }};
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(SurfelPushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = layouts.size();
    pipeline_layout_info.pSetLayouts = layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(m_vk->getDevice(), &pipeline_layout_info, NULL,
        &m_pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for "
            "GEVulkanSurfelGI");
    }

    // The trace shades the hits like the reflection, so it has the same
    // specialization constants as the draw call (GEVulkanPipelineConstants):
    // image based lighting and the number of levels of the specular map (the
    // environment maps only exist then), and the shadow types, without them
    // the sun of a hit would never have a shadow. A ray which misses samples
    // the skybox, which doesn't need them
    GEVulkanPipelineConstants constants(m_vk, m_options);
    createPipeline("surfel_gi_clear.comp", &m_pipelines[PI_CLEAR]);
    createPipeline("surfel_gi_materials.comp", &m_pipelines[PI_MATERIALS]);
    createPipeline("surfel_gi_shadow.comp", &m_pipelines[PI_SHADOW]);
    createPipeline("surfel_gi_trace_resolve.comp", &m_pipelines[PI_TRACE_RESOLVE]);
    createPipeline("surfel_gi_update.comp", &m_pipelines[PI_UPDATE]);
    createPipeline("surfel_gi_trace.comp", &m_pipelines[PI_TRACE],
        constants.getInfo());
    createPipeline("surfel_gi_apply.comp", &m_pipelines[PI_APPLY]);
    createPipeline("surfel_gi_spawn.comp", &m_pipelines[PI_SPAWN]);
    createPipeline("surfel_gi_spawn_bin.comp", &m_pipelines[PI_SPAWN_BIN]);
    createPipeline("surfel_gi_trace_schedule.comp", &m_pipelines[PI_TRACE_SCHEDULE]);
    createPipeline("surfel_gi_probe_trace.comp", &m_pipelines[PI_PROBE_TRACE], constants.getInfo());
    createPipeline("surfel_gi_probe_filter.comp", &m_pipelines[PI_PROBE_FILTER]);
    createPipeline("surfel_gi_share.comp", &m_pipelines[PI_SHARE]);
    createPipeline("surfel_gi_upscale.comp", &m_pipelines[PI_UPSCALE]);
}   // createPipelines

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::clearOutput(VkCommandBuffer cmd)
{
    if (m_cleared)
        return;
    m_cleared = true;
    // What the lighting pass of the last frame read has to be finished, and
    // the clear of the last frame (in another command buffer) too
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    VkClearColorValue black = {};
    vkCmdFillBuffer(cmd, getProbeField(), sizeof(SurfelPushConstants::m_cameras), sizeof(uint32_t), 0);
    VkImageSubresourceRange range = {};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(cmd, m_gi->getImage(), VK_IMAGE_LAYOUT_GENERAL,
        &black, 1, &range);
    vkCmdClearColorImage(cmd, m_contact->getImage(), VK_IMAGE_LAYOUT_GENERAL,
        &black, 1, &range);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
        NULL);
}   // clearOutput

// ----------------------------------------------------------------------------
bool GEVulkanSurfelGI::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    GEVulkanRTScene* rt_scene = m_vk->getRTScene();
    std::vector<Viewport> viewports;
    std::vector<GEVulkanCamera*> cameras;
    for (auto& q : p)
    {
        Viewport vp;
        if (getViewport(q.first, q.second, &vp))
        {
            viewports.push_back(vp);
            cameras.push_back(q.second);
        }
    }
    // Nothing to trace in the menus or before the scene is built
    if (viewports.empty() || !rt_scene || !rt_scene->isReady())
    {
        clearOutput(cmd);
        return false;
    }

    // The layout of the mesh textures is created again when its pool grows
    if (m_pipeline_layout != VK_NULL_HANDLE && m_mesh_texture_layout.expired())
    {
        m_vk->waitIdle();
        destroyPipelines();
    }
    // The options of the draw call changed (for example image based lighting
    // or the shadow map was toggled): the trace has the specialization
    // constants of them, and the irradiance of the surfels was traced with the
    // other ambient or shadow
    if (m_pipeline_layout != VK_NULL_HANDLE &&
        !(m_options == p[0].first->getPipelineConstantOptions()))
    {
        m_vk->waitIdle();
        destroyPipelines();
        m_needs_reset = true;
    }
    if (m_pipeline_layout == VK_NULL_HANDLE)
    {
        try
        {
            m_options = p[0].first->getPipelineConstantOptions();
            createPipelines();
        }
        catch (const std::exception& e)
        {
            GE::logMessage(GE::GLL_ERROR, "%s", e.what());
            destroyPipelines();
        }
    }
    if (m_pipelines[PI_APPLY] == VK_NULL_HANDLE)
    {
        clearOutput(cmd);
        return false;
    }
    SurfelPushConstants pc = {};
    pc.m_frame = ++m_frame;
    if (!pc.m_frame)
    {
        pc.m_frame = ++m_frame;
        m_needs_reset = true;
    }
    pc.m_geometry_count = rt_scene->getActiveGeometries(m_vk->getCurrentBufferIdx());
    pc.m_depth_step = m_depth_step;
    // The time of the frame, at most a tenth of a second (a hitch, or the
    // first frame) so a long pause doesn't fill in everything at once
    const uint64_t now = getMonoTimeMs();
    uint64_t dt_ms = 16;
    if (m_frame > 1)
        dt_ms = std::min<uint64_t>(now - m_last_render, 100);
    m_last_render = now;
    pc.m_steps = (float)dt_ms * (60.0f / 1000.0f);
    const unsigned camera_count =
        std::min<unsigned>((unsigned)cameras.size(), MAX_CAMERAS);
    pc.m_camera_count = camera_count;
    // Every camera has its own budget, the ids above the pool in use are not
    // in the free stack, so it's cleared when the count of them changes
    pc.m_surfel_count = SURFELS_PER_CAMERA * camera_count;
    if (pc.m_surfel_count != m_surfel_count)
    {
        m_surfel_count = pc.m_surfel_count;
        m_needs_reset = true;
    }
    for (unsigned i = 0; i < camera_count; i++)
    {
        const irr::core::matrix4& inverse_view =
            cameras[i]->getUBOData()->m_inverse_view_matrix;
        pc.m_cameras[i][0] = inverse_view[12];
        pc.m_cameras[i][1] = inverse_view[13];
        pc.m_cameras[i][2] = inverse_view[14];
        pc.m_cameras[i][3] = float(m_probe_levels[i]);
        const unsigned prior = unsigned(m_grid_cameras[i][3]);
        if ((prior > m_grid_probe_limit || m_probe_levels[i] > m_probe_layers / camera_count) &&
            (prior != m_probe_levels[i] || m_grid_probe_limit != m_probe_layers / camera_count))
            m_needs_reset = true;
    }
    if (m_needs_reset || m_rt_build_count != rt_scene->getBuildCount())
    {
        pc.m_cache |= RESET_CACHE;
        m_needs_reset = false;
        m_rt_build_count = rt_scene->getBuildCount();
    }
    memcpy(m_grid_cameras, pc.m_cameras, sizeof(m_grid_cameras));
    m_grid_probe_limit = m_probe_layers / camera_count;
    m_time += (uint32_t)dt_ms;
    m_cleared = false;
    pc.m_time = m_time;
    const unsigned viewport_count =
        std::min<unsigned>((unsigned)viewports.size(), MAX_CAMERAS);
    if (isFrameCache())
        pc.m_cache |= CACHE_FRAME;

    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }

    const unsigned buffer_idx = m_vk->getCurrentBufferIdx();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, rt_scene->getDescriptorSet(buffer_idx), 0,
        NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 3, 1,
        m_vk->getMeshTextureDescriptor()->getRTDescriptorSet(), 0, NULL);
    // The light of the hits and the sky of the misses are the ones of the
    // first viewport
    GEVulkanDrawCall* first_dc = viewports[0].m_draw_call;
    const VkDescriptorSet* env = first_dc->getEnvDescriptorSet(m_vk);
    pc.m_has_skybox =
        env != m_vk->getSkyBoxRenderer()->getDummyEnvDescriptorSet();
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 4, 1, env, 0, NULL);
    first_dc->bindDataDescriptorSet(m_vk, cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 1);
    vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(pc), &pc);

    if (pc.m_geometry_count > 0)
    {
        const uint32_t groups = (pc.m_geometry_count + 3) / 4;
        const uint32_t columns = std::min(groups, 65535u);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_MATERIALS]);
        vkCmdDispatch(cmd, columns, (groups + columns - 1) / columns, 1);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_CLEAR]);
    vkCmdDispatch(cmd,
        groups64(std::max<uint32_t>(CELL_COUNT, pc.m_surfel_count)), 1, 1);
    computeToComputeBarrier(cmd, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_UPDATE]);
    vkCmdDispatch(cmd, groups64(pc.m_surfel_count), 1, 1);
    computeToComputeBarrier(cmd, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_SPAWN]);
    for (unsigned index = 0; index < viewport_count; ++index)
    {
        const Viewport& vp = viewports[index];
        if (isEmptyRect(vp.m_rect))
            continue;
        vp.m_draw_call->bindDataDescriptorSet(m_vk, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        std::copy(vp.m_rect.begin(), vp.m_rect.end(), pc.m_rect);
        vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups8(vp.m_rect[2] - vp.m_rect[0]),
            groups8(vp.m_rect[3] - vp.m_rect[1]), 1);
        computeToComputeBarrier(cmd, 0);
    }
    auto indirect_ready = [cmd]()
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 1, &barrier, 0, NULL, 0, NULL);
    };
    indirect_ready();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_SPAWN_BIN]);
    vkCmdDispatchIndirect(cmd, m_buffers[BI_SPAWN_QUEUE].m_buffer, 0);
    computeToComputeBarrier(cmd, 0);
    first_dc->bindDataDescriptorSet(m_vk, cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_TRACE_SCHEDULE]);
    vkCmdDispatch(cmd, groups64(PROBE_LEVEL_SIZE), camera_count, 1);
    computeToComputeBarrier(cmd, 0);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_PROBE_TRACE]);
    vkCmdDispatch(cmd, PROBE_BATCH, camera_count, 1);
    auto resolve_shadows = [&](uint32_t phase)
    {
        indirect_ready();
        pc.m_trace_list = phase;
        vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_SHADOW]);
        vkCmdDispatchIndirect(cmd, m_buffers[BI_RAY_WORK].m_buffer, phase * 4 * sizeof(uint32_t));
        computeToComputeBarrier(cmd, 0);
    };
    resolve_shadows(0);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_PROBE_FILTER]);
    vkCmdDispatch(cmd, PROBE_BATCH, camera_count, 1);
    indirect_ready();
    auto dispatch_lists = [&](PipelineIndex pipeline)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[pipeline]);
        for (uint32_t list = 0; list < TRACE_LIST_COUNT; list++)
        {
            pc.m_trace_list = list;
            vkCmdPushConstants(cmd, m_pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatchIndirect(cmd, m_buffers[BI_TRACE_LIST].m_buffer,
                list * 3 * sizeof(uint32_t));
        }
    };
    dispatch_lists(PI_TRACE);
    resolve_shadows(1);
    dispatch_lists(PI_TRACE_RESOLVE);
    computeToComputeBarrier(cmd, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_SHARE]);
    vkCmdDispatch(cmd, groups64(pc.m_surfel_count), 1, 1);
    computeToComputeBarrier(cmd, 0);

    for (unsigned v = 0; v < viewport_count; v++)
    {
        const Viewport& vp = viewports[v];
        if (isEmptyRect(vp.m_rect))
            continue;
        vp.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        for (unsigned i = 0; i < 4; i++)
            pc.m_rect[i] = vp.m_rect[i];
        vkCmdPushConstants(cmd, m_pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        const int32_t width = vp.m_rect[2] - vp.m_rect[0];
        const int32_t height = vp.m_rect[3] - vp.m_rect[1];
        const uint32_t tiles_x = groups8((width + 1) / 2);
        const uint32_t tiles_y = groups8((height + 1) / 2);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelines[PI_APPLY]);
        vkCmdDispatch(cmd, tiles_x, tiles_y, 1);
        computeToComputeBarrier(cmd, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[PI_UPSCALE]);
        vkCmdDispatch(cmd, groups8(width), groups8(height), 1);
        computeToComputeBarrier(cmd, 0);
    }

    // Sampled by the lighting pass, the surfels are read by the next frame
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }
    // The depth and the normal are read by the compute shaders (execution
    // dependency only), so the g-buffer of the next frame has to wait for it
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 0,
        NULL);
    return true;
}   // render

}
