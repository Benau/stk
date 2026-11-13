#include "ge_vulkan_surfel_gi.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
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

// ----------------------------------------------------------------------------
// Before a transfer write (fill or update) of a buffer which is written by
// the transfer in every frame and used by the compute shaders. The barriers
// after the last write only make it visible to the compute shaders (and the
// indirect dispatches), so the write of the next frame, which is in another
// command buffer, needs the last one to be available to the transfer too
void transferBarrier(VkCommandBuffer cmd, bool indirect_read)
{
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    if (indirect_read)
        barrier.srcAccessMask |= VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
}   // transferBarrier

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanSurfelGI::GEVulkanSurfelGI(GEVulkanDriver* vk,
                                   const irr::core::dimension2d<irr::u32>& size,
                                   GEVulkanAttachmentTexture* depth,
                                   GEVulkanAttachmentTexture* normal)
                : GEVulkanPostProcessing(vk, size), m_buffers(), m_gi(NULL),
                  m_history(), m_layout(VK_NULL_HANDLE),
                  m_pool(VK_NULL_HANDLE), m_sets(),
                  m_data_layout(VK_NULL_HANDLE),
                  m_pipeline_layout(VK_NULL_HANDLE), m_options(),
                  m_pipelines(), m_frame(0), m_time(0), m_last_render(0),
                  m_rt_build_count(0), m_needs_reset(true), m_surfel_count(0),
                  m_cleared(false), m_prev_cameras(), m_prev_camera_count(0),
                  m_history_index(0)
{
    for (Buffer& b : m_buffers)
    {
        b.m_buffer = VK_NULL_HANDLE;
        b.m_memory = VK_NULL_HANDLE;
    }
    m_pipelines.fill(VK_NULL_HANDLE);
    m_history.fill(NULL);
    m_sets.fill(VK_NULL_HANDLE);
    if (!m_vk->getRTScene())
        throw std::runtime_error("No ray traced scene for GEVulkanSurfelGI");
    try
    {
        m_gi = new GEVulkanAttachmentTexture(vk, size,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        initializeGeneralImages(&m_gi, 1);
        // Only read and written by surfel_gi_apply.comp (it's never read
        // before it's written, see CACHE_HISTORY)
        for (GEVulkanAttachmentTexture*& history : m_history)
        {
            history = new GEVulkanAttachmentTexture(vk, size,
                VK_FORMAT_R32G32_UINT, VK_IMAGE_USAGE_STORAGE_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }
        initializeGeneralImages(m_history.data(), m_history.size());
        createBuffers();
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
    for (GEVulkanAttachmentTexture*& history : m_history)
    {
        delete history;
        history = NULL;
    }
    m_data_layout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
    m_sets.fill(VK_NULL_HANDLE);
}   // destroy

// ----------------------------------------------------------------------------
VkImageView GEVulkanSurfelGI::getImageView() const
{
    return (VkImageView)m_gi->getTextureHandler();
}   // getImageView

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createBuffers()
{
    // Only the gpu reads and writes them, surfel_gi_clear.comp fills them on
    // the first frame (the cameras of the last frame are updated by the
    // command buffer)
    const std::array<VkDeviceSize, BI_COUNT> sizes =
    {{
        sizeof(Surfel) * MAX_SURFELS,
        sizeof(int32_t) + sizeof(uint32_t) * MAX_SURFELS,
        sizeof(uint32_t) * CELL_COUNT,
        sizeof(uint32_t) * CELL_COUNT * CELL_CAPACITY,
        sizeof(uint32_t) * CELL_COUNT,
        sizeof(PrevCamera) * MAX_CAMERAS,
        sizeof(uint32_t) * (4 + m_size.Width * m_size.Height),
        sizeof(uint32_t) * ((m_size.Width + 7) / 8) * ((m_size.Height + 7) / 8),
        sizeof(uint32_t) * (4 * TRACE_LIST_COUNT +
            TRACE_LIST_COUNT * MAX_SURFELS)
    }};
    for (unsigned i = 0; i < BI_COUNT; i++)
    {
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (i == BI_PREV_CAMERA || i == BI_TILE_SPAWN)
            usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (i == BI_TRACE_LIST)
            usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (i == BI_MISS)
        {
            usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        }
        if (!m_vk->createBuffer(sizes[i], usage, alloc_info,
            m_buffers[i].m_buffer, m_buffers[i].m_memory))
        {
            throw std::runtime_error("createBuffer failed for "
                "GEVulkanSurfelGI");
        }
    }
}   // createBuffers

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createDescriptors(GEVulkanAttachmentTexture* depth,
                                         GEVulkanAttachmentTexture* normal)
{
    // The g-buffer (combined image samplers), the output, the buffers and the
    // history which is read and written (storage images)
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

    const unsigned set_count = m_sets.size();
    std::array<VkDescriptorPoolSize, 3> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount = 2 * set_count;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = 3 * set_count;
    pool_sizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_sizes[2].descriptorCount = BI_COUNT * set_count;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = set_count;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanSurfelGI");
    }

    std::array<VkDescriptorSetLayout, 2> set_layouts = {{ m_layout, m_layout }};
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = set_count;
    alloc_info.pSetLayouts = set_layouts.data();
    if (vkAllocateDescriptorSets(device, &alloc_info, m_sets.data()) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanSurfelGI");
    }

    for (unsigned set = 0; set < set_count; set++)
    {
        std::array<VkDescriptorImageInfo, 5> image_infos = {};
        image_infos[0].sampler = m_vk->getSampler(GVS_NEAREST);
        image_infos[0].imageView = (VkImageView)depth->getTextureHandler();
        image_infos[0].imageLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        image_infos[1].sampler = m_vk->getSampler(GVS_NEAREST);
        image_infos[1].imageView = (VkImageView)normal->getTextureHandler();
        image_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        image_infos[2].imageView = getImageView();
        image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        // Read the history of this set, write the other one
        image_infos[3].imageView =
            (VkImageView)m_history[set]->getTextureHandler();
        image_infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        image_infos[4].imageView =
            (VkImageView)m_history[(set + 1) % set_count]->getTextureHandler();
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
            writes[i].dstSet = m_sets[set];
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
    }
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
    return getGEConfig()->m_ray_tracing.m_gi_frame_cache;
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
    createPipeline("surfel_gi_update.comp", &m_pipelines[PI_UPDATE]);
    createPipeline("surfel_gi_trace.comp", &m_pipelines[PI_TRACE],
        constants.getInfo());
    createPipeline("surfel_gi_apply.comp", &m_pipelines[PI_APPLY]);
    createPipeline("surfel_gi_gather.comp", &m_pipelines[PI_GATHER]);
    createPipeline("surfel_gi_spawn.comp", &m_pipelines[PI_SPAWN]);
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
    VkImageSubresourceRange range = {};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    vkCmdClearColorImage(cmd, m_gi->getImage(), VK_IMAGE_LAYOUT_GENERAL,
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
void GEVulkanSurfelGI::clearMissList(VkCommandBuffer cmd)
{
    // The passes of the last viewport (or the last frame) read them, and the
    // fills of the last frame have to be finished
    transferBarrier(cmd, true);
    // No pixel and no group of the dispatch (x), the dispatch has y and z of
    // 1, and no spawn in the tiles
    vkCmdFillBuffer(cmd, m_buffers[BI_MISS].m_buffer, 0,
        2 * sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmd, m_buffers[BI_MISS].m_buffer, 2 * sizeof(uint32_t),
        2 * sizeof(uint32_t), 1);
    vkCmdFillBuffer(cmd, m_buffers[BI_TILE_SPAWN].m_buffer, 0, VK_WHOLE_SIZE,
        0xFFFFFFFF);
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
        NULL);
}   // clearMissList

// ----------------------------------------------------------------------------
bool GEVulkanSurfelGI::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    GEVulkanRTScene* rt_scene = m_vk->getRTScene();
    // The history is only written if this frame is traced
    const unsigned prev_camera_count = m_prev_camera_count;
    m_prev_camera_count = 0;
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
    m_cleared = false;

    SurfelPushConstants pc = {};
    pc.m_frame = ++m_frame;
    // The time of the frame, at most a tenth of a second (a hitch, or the
    // first frame) so a long pause doesn't fill in everything at once
    const uint64_t now = getMonoTimeMs();
    uint64_t dt_ms = 16;
    if (m_frame > 1)
        dt_ms = std::min<uint64_t>(now - m_last_render, 100);
    m_last_render = now;
    m_time += (uint32_t)dt_ms;
    pc.m_time = m_time;
    pc.m_steps = (float)dt_ms * (60.0f / 1000.0f);
    const unsigned camera_count =
        std::min<unsigned>((unsigned)p.size(), MAX_CAMERAS);
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
            p[i].second->getUBOData()->m_inverse_view_matrix;
        pc.m_cameras[i][0] = inverse_view[12];
        pc.m_cameras[i][1] = inverse_view[13];
        pc.m_cameras[i][2] = inverse_view[14];
        pc.m_cameras[i][3] = 1.0f;
    }
    // The surfels are on the triangles of the static BLAS, which is new
    if (m_needs_reset || m_rt_build_count != rt_scene->getBuildCount())
    {
        pc.m_reset = 1;
        m_needs_reset = false;
        m_rt_build_count = rt_scene->getBuildCount();
    }
    // The history has the irradiance of the surfels of the last frame, seen
    // by the cameras of its viewports, which can be reused if they are the
    // same viewports (so a pixel is in the same one) and the surfels were not
    // cleared
    const bool cache = isFrameCache();
    const unsigned viewport_count =
        std::min<unsigned>((unsigned)viewports.size(), MAX_CAMERAS);
    if (cache)
    {
        pc.m_cache |= CACHE_FRAME;
        if (pc.m_reset == 0 && prev_camera_count == viewport_count)
            pc.m_cache |= CACHE_HISTORY | CACHE_DEFER;
    }

    // The cameras which the history was rendered with, after the apply of
    // the last frame read them
    if ((pc.m_cache & CACHE_HISTORY) != 0)
    {
        transferBarrier(cmd, false);
        vkCmdUpdateBuffer(cmd, m_buffers[BI_PREV_CAMERA].m_buffer, 0,
            sizeof(PrevCamera) * viewport_count, m_prev_cameras.data());
    }

    // The depth and the normal of the g-buffer are written, the scene is
    // built, the cameras of the last frame are updated, and what the last
    // frame read or wrote has to be finished
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
    // The history of the last frame is read, the other one written
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_sets[m_history_index], 0, NULL);
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

    // 1. Empty cells (and pool)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_CLEAR]);
    vkCmdDispatch(cmd,
        groups64(std::max<uint32_t>(CELL_COUNT, pc.m_surfel_count)), 1, 1);
    computeToComputeBarrier(cmd, 0);

    // 2. Recycle, resize and register in the cells (and the lists of the trace)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_UPDATE]);
    vkCmdDispatch(cmd, groups64(pc.m_surfel_count), 1, 1);
    computeToComputeBarrier(cmd, 0);

    // 3. Rays of the surfels, with the frame cache the ones of the lists of
    // the update (their sizes are the indirect dispatches)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_TRACE]);
    if (cache)
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
        for (uint32_t list = 0; list < TRACE_LIST_COUNT; list++)
        {
            pc.m_trace_list = list;
            vkCmdPushConstants(cmd, m_pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatchIndirect(cmd, m_buffers[BI_TRACE_LIST].m_buffer,
                list * 3 * sizeof(uint32_t));
        }
    }
    else
        vkCmdDispatch(cmd, groups64(pc.m_surfel_count), 1, 1);
    computeToComputeBarrier(cmd, 0);

    // 4. Irradiance of the pixels and new surfels, per viewport
    const bool defer = (pc.m_cache & CACHE_DEFER) != 0;
    for (unsigned v = 0; v < viewport_count; v++)
    {
        const Viewport& vp = viewports[v];
        if (isEmptyRect(vp.m_rect))
            continue;
        vp.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        for (unsigned i = 0; i < 4; i++)
            pc.m_rect[i] = vp.m_rect[i];
        pc.m_viewport_index = v;
        vkCmdPushConstants(cmd, m_pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        const uint32_t tiles_x = groups8(vp.m_rect[2] - vp.m_rect[0]);
        const uint32_t tiles_y = groups8(vp.m_rect[3] - vp.m_rect[1]);
        if (defer)
            clearMissList(cmd);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelines[PI_APPLY]);
        vkCmdDispatch(cmd, tiles_x, tiles_y, 1);
        if (!defer)
            continue;
        // The list of the misses is complete, and its size is the indirect
        // dispatch
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
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelines[PI_GATHER]);
        vkCmdDispatchIndirect(cmd, m_buffers[BI_MISS].m_buffer,
            sizeof(uint32_t));
        computeToComputeBarrier(cmd, 0);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelines[PI_SPAWN]);
        vkCmdDispatch(cmd, groups64(tiles_x * tiles_y), 1, 1);
        computeToComputeBarrier(cmd, 0);
    }

    // The next frame reads the history written now, with the cameras of
    // this frame
    if (cache)
    {
        m_history_index = (m_history_index + 1) % m_sets.size();
        m_prev_camera_count = viewport_count;
        for (unsigned i = 0; i < viewport_count; i++)
        {
            const GEVulkanCameraUBO* ubo = cameras[i]->getUBOData();
            PrevCamera& prev = m_prev_cameras[i];
            memcpy(prev.m_projection_view_matrix,
                ubo->m_projection_view_matrix.pointer(),
                sizeof(prev.m_projection_view_matrix));
            prev.m_viewport[0] = ubo->m_viewport.UpperLeftCorner.X;
            prev.m_viewport[1] = ubo->m_viewport.UpperLeftCorner.Y;
            prev.m_viewport[2] = ubo->m_viewport.LowerRightCorner.X;
            prev.m_viewport[3] = ubo->m_viewport.LowerRightCorner.Y;
            prev.m_position[0] = ubo->m_inverse_view_matrix[12];
            prev.m_position[1] = ubo->m_inverse_view_matrix[13];
            prev.m_position[2] = ubo->m_inverse_view_matrix[14];
            prev.m_position[3] = 1.0f;
        }
    }

    // Sampled by the lighting pass, the surfels are read by the next frame
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
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
