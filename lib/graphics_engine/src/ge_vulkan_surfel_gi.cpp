#include "ge_vulkan_surfel_gi.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_environment_map.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_rt_scene.hpp"
#include "ge_vulkan_shader_manager.hpp"
#include "ge_vulkan_skybox_renderer.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include <algorithm>
#include <cmath>
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

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanSurfelGI::GEVulkanSurfelGI(GEVulkanDriver* vk,
                                   const irr::core::dimension2d<irr::u32>& size,
                                   GEVulkanAttachmentTexture* depth,
                                   GEVulkanAttachmentTexture* normal)
                : GEVulkanPostProcessing(vk, size), m_buffers(), m_gi(NULL),
                  m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
                  m_set(VK_NULL_HANDLE), m_data_layout(VK_NULL_HANDLE),
                  m_pipeline_layout(VK_NULL_HANDLE), m_pipelines(),
                  m_frame(0), m_rt_build_count(0), m_needs_reset(true),
                  m_cleared(false)
{
    for (Buffer& b : m_buffers)
    {
        b.m_buffer = VK_NULL_HANDLE;
        b.m_memory = VK_NULL_HANDLE;
    }
    m_pipelines.fill(VK_NULL_HANDLE);
    if (!m_vk->getRTScene())
        throw std::runtime_error("No ray traced scene for GEVulkanSurfelGI");
    try
    {
        m_gi = new GEVulkanAttachmentTexture(vk, size,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        initializeGeneralImages(&m_gi, 1);
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

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createBuffers()
{
    // Only the gpu reads and writes them, surfel_gi_clear.comp fills them on
    // the first frame
    const std::array<VkDeviceSize, BI_COUNT> sizes =
    {{
        sizeof(Surfel) * MAX_SURFELS,
        sizeof(int32_t) + sizeof(uint32_t) * MAX_SURFELS,
        sizeof(uint32_t) * CELL_COUNT,
        sizeof(uint32_t) * CELL_COUNT * CELL_CAPACITY
    }};
    for (unsigned i = 0; i < BI_COUNT; i++)
    {
        VmaAllocationCreateInfo alloc_info = {};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        if (!m_vk->createBuffer(sizes[i], VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            alloc_info, m_buffers[i].m_buffer, m_buffers[i].m_memory))
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
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, 3 + BI_COUNT> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = i < 2 ?
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : i == 2 ?
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE :
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
    pool_sizes[1].descriptorCount = 1;
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
    if (vkAllocateDescriptorSets(device, &alloc_info, &m_set) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanSurfelGI");
    }

    std::array<VkDescriptorImageInfo, 3> image_infos = {};
    image_infos[0].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[0].imageView = (VkImageView)depth->getTextureHandler();
    image_infos[0].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    image_infos[1].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[1].imageView = (VkImageView)normal->getTextureHandler();
    image_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_infos[2].imageView = getImageView();
    image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    std::array<VkDescriptorBufferInfo, BI_COUNT> buffer_infos = {};
    for (unsigned i = 0; i < BI_COUNT; i++)
    {
        buffer_infos[i].buffer = m_buffers[i].m_buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = VK_WHOLE_SIZE;
    }
    std::array<VkWriteDescriptorSet, 3 + BI_COUNT> writes = {};
    for (unsigned i = 0; i < writes.size(); i++)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = bindings[i].descriptorType;
        if (i < 3)
            writes[i].pImageInfo = &image_infos[i];
        else
            writes[i].pBufferInfo = &buffer_infos[i - 3];
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

    // The trace shades the hits like the reflection, which reads the image
    // based lighting setting and the number of levels of the specular map, the
    // same as GEVulkanDrawCall::createPipeline (the environment maps only
    // exist then). A ray which misses samples the skybox, which doesn't need
    // them
    struct Constants
    {
        VkBool32 m_ibl;
        float m_specular_levels_minus_one;
    };
    Constants constants = {};
    constants.m_ibl = getGEConfig()->m_pbr && getGEConfig()->m_ibl &&
        GEVulkanFeatures::supportsComputeInMainQueue();
    constants.m_specular_levels_minus_one = std::floor(std::log2(
        GEVulkanEnvironmentMap::getSpecularEnvironmentMapSize().Width));
    std::array<VkSpecializationMapEntry, 2> entries = {};
    entries[0].constantID = 0;
    entries[0].offset = offsetof(Constants, m_ibl);
    entries[0].size = sizeof(VkBool32);
    entries[1].constantID = 1;
    entries[1].offset = offsetof(Constants, m_specular_levels_minus_one);
    entries[1].size = sizeof(float);
    VkSpecializationInfo specialization = {};
    specialization.mapEntryCount = entries.size();
    specialization.pMapEntries = entries.data();
    specialization.dataSize = sizeof(Constants);
    specialization.pData = &constants;

    createPipeline("surfel_gi_clear.comp", &m_pipelines[PI_CLEAR]);
    createPipeline("surfel_gi_update.comp", &m_pipelines[PI_UPDATE]);
    createPipeline("surfel_gi_trace.comp", &m_pipelines[PI_TRACE],
        &specialization);
    createPipeline("surfel_gi_apply.comp", &m_pipelines[PI_APPLY]);
}   // createPipelines

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::clearOutput(VkCommandBuffer cmd)
{
    if (m_cleared)
        return;
    m_cleared = true;
    // What the lighting pass of the last frame read has to be finished
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
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
bool GEVulkanSurfelGI::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    GEVulkanRTScene* rt_scene = m_vk->getRTScene();
    std::vector<Viewport> viewports;
    for (auto& q : p)
    {
        Viewport vp;
        if (getViewport(q.first, q.second, &vp))
            viewports.push_back(vp);
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
    if (m_pipeline_layout == VK_NULL_HANDLE)
    {
        try
        {
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
    const unsigned camera_count = std::min<unsigned>((unsigned)p.size(), 4);
    pc.m_camera_count = camera_count;
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

    // The depth and the normal of the g-buffer are written, the scene is
    // built, and what the last frame read or wrote has to be finished
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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

    // 1. Empty cells (and pool)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_CLEAR]);
    vkCmdDispatch(cmd, groups64(std::max<uint32_t>(CELL_COUNT, MAX_SURFELS)),
        1, 1);
    computeToComputeBarrier(cmd, 0);

    // 2. Recycle, resize and register in the cells
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_UPDATE]);
    vkCmdDispatch(cmd, groups64(MAX_SURFELS), 1, 1);
    computeToComputeBarrier(cmd, 0);

    // 3. Rays of the surfels
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_TRACE]);
    vkCmdDispatch(cmd, groups64(MAX_SURFELS), 1, 1);
    computeToComputeBarrier(cmd, 0);

    // 4. Irradiance of the pixels and new surfels, per viewport
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_APPLY]);
    for (const Viewport& vp : viewports)
    {
        if (isEmptyRect(vp.m_rect))
            continue;
        vp.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        for (unsigned i = 0; i < 4; i++)
            pc.m_rect[i] = vp.m_rect[i];
        vkCmdPushConstants(cmd, m_pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, groups8(vp.m_rect[2] - vp.m_rect[0]),
            groups8(vp.m_rect[3] - vp.m_rect[1]), 1);
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
