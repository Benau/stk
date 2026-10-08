#include "ge_vulkan_light_scatter.hpp"

#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_main.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <stdexcept>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
struct BlurPushConstants
{
    // x0, y0, x1, y1 (exclusive) in pixels of the scatter image
    int32_t m_rect[4];
    // (1, 0) = horizontal, (0, 1) = vertical
    int32_t m_direction[2];
    // Multiplies the distance of the samples
    float m_scale;
};
}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanLightScatter::GEVulkanLightScatter(GEVulkanDriver* vk,
                              const irr::core::dimension2d<irr::u32>& size,
                              GEVulkanAttachmentTexture* depth)
                    : GEVulkanPostProcessing(vk, size),
                      m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
                      m_blur_pipeline_layout(VK_NULL_HANDLE),
                      m_blur_pipeline(VK_NULL_HANDLE),
                      m_data_layout(VK_NULL_HANDLE),
                      m_pipeline_layout(VK_NULL_HANDLE),
                      m_pipeline(VK_NULL_HANDLE)
{
    m_sets.fill(VK_NULL_HANDLE);
    // No alpha is needed and the values are never negative, B10G11R11 has the
    // range and half of the memory of RGBA16F (storage image support of it
    // is optional)
    VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
    if (GEVulkanFeatures::supportsShaderStorageImageExtendedFormats())
        format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    const irr::core::dimension2d<irr::u32> half = halfSize(size);
    const VkImageUsageFlags usage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    for (GEVulkanAttachmentTexture*& t : m_scatter)
    {
        t = new GEVulkanAttachmentTexture(vk, half, format, usage,
            VK_IMAGE_ASPECT_COLOR_BIT);
    }

    // The images stay in general layout forever
    initializeGeneralImages(m_scatter.data(), m_scatter.size());

    createDescriptors(depth);
    createBlurPipeline();
    createPipeline();
}   // GEVulkanLightScatter

// ----------------------------------------------------------------------------
GEVulkanLightScatter::~GEVulkanLightScatter()
{
    VkDevice device = m_vk->getDevice();
    if (m_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_pipeline, NULL);
    if (m_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    if (m_data_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_data_layout, NULL);
    if (m_blur_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_blur_pipeline, NULL);
    if (m_blur_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_blur_pipeline_layout, NULL);
    // Descriptor sets are freed with their pool
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_pool, NULL);
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_layout, NULL);
    for (GEVulkanAttachmentTexture* t : m_scatter)
        delete t;
}   // ~GEVulkanLightScatter

// ----------------------------------------------------------------------------
VkImageView GEVulkanLightScatter::getImageView() const
{
    return (VkImageView)m_scatter[0]->getTextureHandler();
}   // getImageView

// ----------------------------------------------------------------------------
void GEVulkanLightScatter::createDescriptors(GEVulkanAttachmentTexture* depth)
{
    createImageDescriptorSetLayout(&m_layout);
    createImageDescriptorPool(m_sets.size(), &m_pool);
    allocateImageDescriptorSets(m_pool, m_layout, m_sets.size(),
        m_sets.data());

    // Index of m_scatter, -1 is the depth
    const int inputs[3] = { -1, 0, 1 };
    const int outputs[3] = { 0, 1, 0 };
    for (unsigned i = 0; i < m_sets.size(); i++)
    {
        VkSampler sampler;
        VkImageLayout input_layout;
        VkImageView input_view;

        if (inputs[i] < 0)
        {
            sampler = m_vk->getSampler(GVS_NEAREST);
            input_layout =
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            input_view =
                (VkImageView)depth->getTextureHandler();
        }
        else
        {
            // Linear for the merged taps of the blur
            sampler = m_vk->getSampler(GVS_SKYBOX);
            input_layout = VK_IMAGE_LAYOUT_GENERAL;
            input_view =
                (VkImageView)m_scatter[inputs[i]]->getTextureHandler();
        }

        writeImageDescriptorSet(m_sets[i], input_view, sampler, input_layout,
            (VkImageView)m_scatter[outputs[i]]->getTextureHandler(),
            VK_IMAGE_LAYOUT_GENERAL);
    }
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanLightScatter::createBlurPipeline()
{
    VkDevice device = m_vk->getDevice();
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(BlurPushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &m_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
        &m_blur_pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for blur in "
            "GEVulkanLightScatter");
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module =
        GEVulkanShaderManager::getShader("light_scatter_blur.comp");
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_blur_pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
        NULL, &m_blur_pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateComputePipelines failed for blur in "
            "GEVulkanLightScatter");
    }
}   // createBlurPipeline

// ----------------------------------------------------------------------------
void GEVulkanLightScatter::createPipeline()
{
    VkDevice device = m_vk->getDevice();
    m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(int32_t) * 4;
    // The data descriptor set layout of draw calls are all defined the same,
    // so the sets of any draw call are compatible
    std::array<VkDescriptorSetLayout, 2> layouts =
    {{
        m_layout,
        m_data_layout
    }};
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = layouts.size();
    pipeline_layout_info.pSetLayouts = layouts.data();
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
        &m_pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for "
            "GEVulkanLightScatter");
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module =
        GEVulkanShaderManager::getShader("light_scatter.comp");
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
        NULL, &m_pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateComputePipelines failed for "
            "GEVulkanLightScatter");
    }
}   // createPipeline

// ----------------------------------------------------------------------------
bool GEVulkanLightScatter::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCameraSceneNode*> >& p)
{
    std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCameraSceneNode*> > list;
    for (auto& q : p)
    {
        if (q.first->hasLightScatter())
            list.push_back(q);
    }
    if (list.empty())
        return false;

    // The depth of the g-buffer is written, and what the last frame read or
    // wrote of the images has to be finished
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }

    struct Viewport
    {
        // x0, y0, x1, y1 in pixels of the FBO
        std::array<int32_t, 4> m_rect;
        // Height of the viewport in pixels of the FBO, without rotation
        float m_height;
    };
    std::vector<Viewport> viewports;
    std::vector<GEVulkanDrawCall*> draw_calls;
    // Same as GEVulkanDrawCall::getRenderViewport, before rotation
    const float scale = m_vk->getSeparateRTTTexture() ? 1.0f :
        getGEConfig()->m_render_scale;
    for (auto& q : list)
    {
        // The same viewport as prepareViewport, in pixels of the FBO
        const VkViewport vp = q.first->getRenderViewport(m_vk,
            q.second->getViewPort());
        const int32_t x0 = std::max(0, (int32_t)vp.x);
        const int32_t y0 = std::max(0, (int32_t)vp.y);
        const int32_t x1 = std::min((int32_t)m_size.Width,
            (int32_t)vp.x + (int32_t)vp.width);
        const int32_t y1 = std::min((int32_t)m_size.Height,
            (int32_t)vp.y + (int32_t)vp.height);
        if (x1 > x0 && y1 > y0)
        {
            viewports.push_back({ {{ x0, y0, x1, y1 }},
                q.second->getViewPort().getHeight() * scale });
            draw_calls.push_back(q.first);
        }
    }
    if (viewports.empty())
        return false;

    VkMemoryBarrier compute_barrier = {};
    compute_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    compute_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compute_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    // 1. Every viewport marches its rays, with its own camera and lights
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &m_sets[0], 0, NULL);
    for (unsigned i = 0; i < viewports.size(); i++)
    {
        draw_calls[i]->bindDataDescriptorSet(m_vk, cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        const std::array<int32_t, 4>& r = viewports[i].m_rect;
        int32_t rect[4] = { r[0], r[1], r[2], r[3] };
        vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(rect), rect);
        // The area in the half size image, covers all the pixels of the rect
        const uint32_t w = (uint32_t)(((r[2] + 1) >> 1) - (r[0] >> 1));
        const uint32_t h = (uint32_t)(((r[3] + 1) >> 1) - (r[1] >> 1));
        vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    }
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &compute_barrier, 0, NULL,
        0, NULL);

    // 2. Horizontal and then vertical blur, one dispatch per viewport. The
    // barriers are between all of them so it also avoids writing the shared
    // border pixels of two viewports at the same time
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blur_pipeline);
    for (unsigned pass = 0; pass < 2; pass++)
    {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_blur_pipeline_layout, 0, 1, &m_sets[pass + 1], 0, NULL);
        for (const Viewport& vp : viewports)
        {
            BlurPushConstants pc = {};
            pc.m_rect[0] = vp.m_rect[0] >> 1;
            pc.m_rect[1] = vp.m_rect[1] >> 1;
            pc.m_rect[2] = (vp.m_rect[2] + 1) >> 1;
            pc.m_rect[3] = (vp.m_rect[3] + 1) >> 1;
            pc.m_direction[0] = pass == 0 ? 1 : 0;
            pc.m_direction[1] = pass == 0 ? 0 : 1;
            pc.m_scale = std::min(std::max(vp.m_height / REFERENCE_HEIGHT,
                0.25f), 2.5f);
            vkCmdPushConstants(cmd, m_blur_pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            const uint32_t w = (uint32_t)(pc.m_rect[2] - pc.m_rect[0]);
            const uint32_t h = (uint32_t)(pc.m_rect[3] - pc.m_rect[1]);
            vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
        }
        if (pass == 0)
        {
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &compute_barrier,
                0, NULL, 0, NULL);
        }
    }

    // 3. Sampled by the lighting pass
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }
    // The depth is read by the compute shader (execution dependency only),
    // so the g-buffer of the next frame has to wait for it
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, 0, NULL, 0, NULL, 0,
        NULL);
    return true;
}   // render

}
