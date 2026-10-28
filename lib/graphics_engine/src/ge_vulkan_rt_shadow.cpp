#include "ge_vulkan_rt_shadow.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_array_texture.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_rt_scene.hpp"
#include "ge_vulkan_shader_manager.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace GE
{
namespace
{
// Same as the push constants of rt_shadow.comp and rt_shadow_clear.comp
struct RTShadowPushConstants
{
    // x0, y0, x1, y1 (exclusive) in texels of the image
    int32_t m_rect[4];
    float m_scale;
    // Bits of GEShadowType which are traced
    int32_t m_types;
    // Number of layers of lights (the layers after the sun)
    int32_t m_lights;
};

// ----------------------------------------------------------------------------
// Number of groups (8x8 threads) which cover the area of a rect
uint32_t groups(int32_t size)
{
    return (uint32_t)(std::max(size, 0) + 7) / 8;
}   // groups

// ----------------------------------------------------------------------------
// Size of the image of a render target, at least 1 texel
irr::core::dimension2d<irr::u32> scaledSize(
                                    const irr::core::dimension2d<irr::u32>& size,
                                    float scale)
{
    return irr::core::dimension2d<irr::u32>(
        std::max(1u, (irr::u32)std::ceil((float)size.Width * scale)),
        std::max(1u, (irr::u32)std::ceil((float)size.Height * scale)));
}   // scaledSize

// ----------------------------------------------------------------------------
// The texels of the image which belong to a rect of the render target (the
// one of a viewport), same as rtShadowTexelRect of rt_shadow_grid.glsl: the
// texel t samples the pixel floor((t + 0.5) / scale), so it's in the rect if
// x0 <= (t + 0.5) / scale < x1. The rects of the viewports don't overlap so
// the texel rects don't
std::array<int32_t, 4> texelRect(const std::array<int32_t, 4>& rect,
                                 float scale,
                                 const irr::core::dimension2d<irr::u32>& max)
{
    std::array<int32_t, 4> out = {};
    for (unsigned i = 0; i < 4; i++)
        out[i] = (int32_t)std::ceil((float)rect[i] * scale - 0.5f);
    out[0] = std::max(out[0], 0);
    out[1] = std::max(out[1], 0);
    out[2] = std::min(out[2], (int32_t)max.Width);
    out[3] = std::min(out[3], (int32_t)max.Height);
    return out;
}   // texelRect

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanRTShadow::GEVulkanRTShadow(GEVulkanDriver* vk,
                                   const irr::core::dimension2d<irr::u32>& size,
                                   GEVulkanAttachmentTexture* depth,
                                   GEVulkanAttachmentTexture* normal,
                                   bool active)
                : GEVulkanPostProcessing(vk, size), m_layers(1),
                  m_scale(active ? getRTShadowScale() : 1.0f), m_mask(NULL),
                  m_storage_view(VK_NULL_HANDLE), m_layout(VK_NULL_HANDLE),
                  m_pool(VK_NULL_HANDLE), m_set(VK_NULL_HANDLE),
                  m_data_layout(VK_NULL_HANDLE),
                  m_pipeline_layout(VK_NULL_HANDLE),
                  m_pipeline(VK_NULL_HANDLE), m_clear_pipeline(VK_NULL_HANDLE)
{
    try
    {
        // Visibility in [0, 1] (x) and linear depth (y) of the surface of the
        // texel, RG32F is a storage format without the extended ones
        VkFormat format = VK_FORMAT_R32G32_SFLOAT;
        if (GEVulkanFeatures::supportsShaderStorageImageExtendedFormats())
            format = VK_FORMAT_R16G16_SFLOAT;
        // 4 is the minimum size of the image
        const irr::core::dimension2d<irr::u32> mask_size = active ?
            scaledSize(size, m_scale) : irr::core::dimension2d<irr::u32>(4, 4);
        // The lights which have a shadow are the first ones, the limit is the
        // same as the shadow maps of the lights
        if (active && (getRTShadowType() & GST_POINTLIGHT) != 0)
        {
            m_layers += std::min(getGEConfig()->m_point_shadow_limit,
                getGEConfig()->m_max_pointlights);
        }
        m_mask = new GEVulkanArrayTexture(format, VK_IMAGE_VIEW_TYPE_2D_ARRAY,
            mask_size, m_layers, irr::video::SColor(0),
            VK_IMAGE_LAYOUT_GENERAL, false/*create_mipmap*/);
        if (active)
        {
            createDescriptors(depth, normal);
            createPipelines();
        }
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanRTShadow

// ----------------------------------------------------------------------------
GEVulkanRTShadow::~GEVulkanRTShadow()
{
    destroy();
}   // ~GEVulkanRTShadow

// ----------------------------------------------------------------------------
void GEVulkanRTShadow::destroy()
{
    m_vk->waitIdle();
    VkDevice device = m_vk->getDevice();
    for (VkPipeline* pipeline : { &m_pipeline, &m_clear_pipeline })
    {
        if (*pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device, *pipeline, NULL);
        *pipeline = VK_NULL_HANDLE;
    }
    if (m_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    if (m_data_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_data_layout, NULL);
    // Descriptor sets are freed with their pool
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_pool, NULL);
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_layout, NULL);
    if (m_storage_view != VK_NULL_HANDLE)
        vkDestroyImageView(device, m_storage_view, NULL);
    delete m_mask;
    m_mask = NULL;
    m_pipeline_layout = VK_NULL_HANDLE;
    m_data_layout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
    m_storage_view = VK_NULL_HANDLE;
}   // destroy

// ----------------------------------------------------------------------------
VkImageView GEVulkanRTShadow::getImageView() const
{
    return (VkImageView)m_mask->getTextureHandler();
}   // getImageView

// ----------------------------------------------------------------------------
void GEVulkanRTShadow::createDescriptors(GEVulkanAttachmentTexture* depth,
                                         GEVulkanAttachmentTexture* normal)
{
    VkDevice device = m_vk->getDevice();

    VkImageViewCreateInfo view_info = {};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = m_mask->getImage();
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    view_info.format = m_mask->getInternalFormat();
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = m_layers;
    if (vkCreateImageView(device, &view_info, NULL, &m_storage_view) !=
        VK_SUCCESS)
        throw std::runtime_error("vkCreateImageView failed for GEVulkanRTShadow");

    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = i < 2 ?
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
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
            "GEVulkanRTShadow");
    }

    std::array<VkDescriptorPoolSize, 2> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount = 2;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = 1;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanRTShadow");
    }

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &m_layout;
    if (vkAllocateDescriptorSets(device, &alloc_info, &m_set) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanRTShadow");
    }

    std::array<VkDescriptorImageInfo, 3> infos = {};
    infos[0].sampler = m_vk->getSampler(GVS_NEAREST);
    infos[0].imageView = (VkImageView)depth->getTextureHandler();
    infos[0].imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    infos[1].sampler = m_vk->getSampler(GVS_NEAREST);
    infos[1].imageView = (VkImageView)normal->getTextureHandler();
    infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    infos[2].imageView = m_storage_view;
    infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    std::array<VkWriteDescriptorSet, 3> writes = {};
    for (unsigned i = 0; i < writes.size(); i++)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = i < 2 ?
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[i].pImageInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, NULL);
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanRTShadow::createPipeline(const char* shader,
                                      VkPipeline* pipeline) const
{
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = GEVulkanShaderManager::getShader(shader);
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_pipeline_layout;
    if (vkCreateComputePipelines(m_vk->getDevice(), m_vk->getPipelineCache(),
        1, &pipeline_info, NULL, pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkCreateComputePipelines "
            "failed for GEVulkanRTShadow ") + shader);
    }
}   // createPipeline

// ----------------------------------------------------------------------------
void GEVulkanRTShadow::createPipelines()
{
    GEVulkanRTScene* rt_scene = m_vk->getRTScene();
    if (!rt_scene)
        throw std::runtime_error("No ray traced scene for GEVulkanRTShadow");
    m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(RTShadowPushConstants);
    // The layouts of the scene and of the mesh textures live as long as the
    // driver objects which the swapchain related objects (this one) are
    // created again with when they change (updateDriver, reloadShaders)
    std::array<VkDescriptorSetLayout, 4> layouts =
    {{
        rt_scene->getDescriptorSetLayout(),
        m_data_layout,
        m_layout,
        *m_vk->getMeshTextureDescriptor()->getRTDescriptorSetLayout()
    }};
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
            "GEVulkanRTShadow");
    }
    createPipeline("rt_shadow.comp", &m_pipeline);
    createPipeline("rt_shadow_clear.comp", &m_clear_pipeline);
}   // createPipelines

// ----------------------------------------------------------------------------
bool GEVulkanRTShadow::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    const GEShadowType types = getRTShadowType();
    if (!isActive() || types == GST_NONE)
        return false;
    std::vector<Viewport> viewports;
    for (auto& q : p)
    {
        Viewport vp;
        if (getViewport(q.first, q.second, &vp))
            viewports.push_back(vp);
    }
    if (viewports.empty())
        return false;

    // The depth and the normal of the g-buffer are written, the scene is
    // built (the barrier of it has the compute stage too), and what the last
    // frame read or wrote of the image has to be finished
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

    GEVulkanRTScene* rt_scene = m_vk->getRTScene();
    // Nothing to trace in the menus or before the scene is built, it's all lit
    const bool traced = rt_scene && rt_scene->isReady();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        traced ? m_pipeline : m_clear_pipeline);
    if (traced)
    {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipeline_layout, 0, 1,
            rt_scene->getDescriptorSet(m_vk->getCurrentBufferIdx()), 0, NULL);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipeline_layout, 3, 1,
            m_vk->getMeshTextureDescriptor()->getRTDescriptorSet(), 0, NULL);
    }
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    RTShadowPushConstants pc = {};
    pc.m_scale = m_scale;
    pc.m_types = (int32_t)types;
    // The layers after the sun are the lights
    pc.m_lights = (int32_t)m_layers - 1;
    const irr::core::dimension2d<irr::u32> mask_size =
        scaledSize(m_size, m_scale);
    for (const Viewport& vp : viewports)
    {
        const std::array<int32_t, 4> rect =
            texelRect(vp.m_rect, m_scale, mask_size);
        if (rect[2] <= rect[0] || rect[3] <= rect[1])
            continue;
        if (traced)
        {
            // The camera and the global light of the viewport
            vp.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
                VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        }
        for (unsigned i = 0; i < 4; i++)
            pc.m_rect[i] = rect[i];
        vkCmdPushConstants(cmd, m_pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        // The layers are the z, the ones which are not used return at once
        vkCmdDispatch(cmd, groups(rect[2] - rect[0]),
            groups(rect[3] - rect[1]), m_layers);
    }

    // Sampled by the lighting pass
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }
    // The depth and the normal are read by the compute shader (execution
    // dependency only), so the g-buffer of the next frame has to wait for it
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 0,
        NULL);
    return true;
}   // render

}
