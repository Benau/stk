#include "ge_vulkan_god_rays.hpp"

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
// ----------------------------------------------------------------------------
GEVulkanGodRays::GEVulkanGodRays(GEVulkanDriver* vk,
                              const irr::core::dimension2d<irr::u32>& size,
                              GEVulkanAttachmentTexture* depth)
               : GEVulkanPostProcessing(vk, size),
                 m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
                 m_data_layout(VK_NULL_HANDLE),
                 m_pipeline_layout(VK_NULL_HANDLE),
                 m_pipeline(VK_NULL_HANDLE)
{
    m_rays.fill(NULL);
    m_sets.fill(VK_NULL_HANDLE);
    // No alpha is needed and the values are never negative, B10G11R11 has the
    // range and half of the memory of RGBA16F (storage image support of it
    // is optional)
    VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
    if (GEVulkanFeatures::supportsShaderStorageImageExtendedFormats())
        format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    try
    {
        const irr::core::dimension2d<irr::u32> quarter = quarterSize(size);
        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        for (GEVulkanAttachmentTexture*& t : m_rays)
        {
            t = new GEVulkanAttachmentTexture(vk, quarter, format, usage,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }

        // The images stay in general layout forever
        initializeGeneralImages(m_rays.data(), m_rays.size());

        createDescriptors(depth);
        createBlurPipeline();
        createPipeline();
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanGodRays

// ----------------------------------------------------------------------------
GEVulkanGodRays::~GEVulkanGodRays()
{
    destroy();
}   // ~GEVulkanGodRays

// ----------------------------------------------------------------------------
void GEVulkanGodRays::destroy()
{
    VkDevice device = m_vk->getDevice();
    if (m_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_pipeline, NULL);
    if (m_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    if (m_data_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_data_layout, NULL);
    destroyBlurPipelines(&m_blur_pipelines);
    // Descriptor sets are freed with their pool
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_pool, NULL);
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_layout, NULL);
    for (GEVulkanAttachmentTexture* t : m_rays)
        delete t;
}   // destroy

// ----------------------------------------------------------------------------
irr::core::dimension2d<irr::u32> GEVulkanGodRays::quarterSize(
    const irr::core::dimension2d<irr::u32>& size)
{
    return irr::core::dimension2d<irr::u32>(std::max(1u, (size.Width + 3) / 4),
        std::max(1u, (size.Height + 3) / 4));
}   // quarterSize

// ----------------------------------------------------------------------------
std::array<int32_t, 4> GEVulkanGodRays::quarterRect(
    const std::array<int32_t, 4>& r)
{
    return {{ r[0] >> 2, r[1] >> 2, (r[2] + 3) >> 2, (r[3] + 3) >> 2 }};
}   // quarterRect

// ----------------------------------------------------------------------------
VkImageView GEVulkanGodRays::getImageView() const
{
    return (VkImageView)m_rays[0]->getTextureHandler();
}   // getImageView

// ----------------------------------------------------------------------------
void GEVulkanGodRays::createDescriptors(GEVulkanAttachmentTexture* depth)
{
    createImageDescriptorSetLayout(&m_layout);
    createImageDescriptorPool(m_sets.size(), &m_pool);
    allocateImageDescriptorSets(m_pool, m_layout, m_sets.size(),
        m_sets.data());

    // Index of m_rays, -1 is the depth
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
                (VkImageView)m_rays[inputs[i]]->getTextureHandler();
        }

        writeImageDescriptorSet(m_sets[i], input_view, sampler, input_layout,
            (VkImageView)m_rays[outputs[i]]->getTextureHandler(),
            VK_IMAGE_LAYOUT_GENERAL);
    }
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanGodRays::createBlurPipeline()
{
    createBlurPipelines("blur_hdr.comp", m_layout,
        blurModeBit(BM_GAUSSIAN), &m_blur_pipelines);
}   // createBlurPipeline

// ----------------------------------------------------------------------------
void GEVulkanGodRays::createPipeline()
{
    VkDevice device = m_vk->getDevice();
    m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    // Rect (4 ints), position and radius of the sun, color (4 floats each)
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(int32_t) * 4 + sizeof(float) * 8;
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
            "GEVulkanGodRays");
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module =
        GEVulkanShaderManager::getShader("god_rays.comp");
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_pipeline_layout;
    if (vkCreateComputePipelines(device, m_vk->getPipelineCache(), 1,
        &pipeline_info, NULL, &m_pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateComputePipelines failed for "
            "GEVulkanGodRays");
    }
}   // createPipeline

// ----------------------------------------------------------------------------
bool GEVulkanGodRays::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    if (!m_vk->isGodRaysEnabled() || p.empty())
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

    // The same for all viewports, already multiplied by the opacity
    const irr::core::vector3df& sun = m_vk->getGodRaysPosition();
    const irr::core::vector3df& color = m_vk->getGodRaysColor();
    struct PushConstants
    {
        int32_t m_rect[4];
        float m_sun_radius[4];
        float m_color[4];
    } pc = {};
    pc.m_sun_radius[0] = sun.X;
    pc.m_sun_radius[1] = sun.Y;
    pc.m_sun_radius[2] = sun.Z;
    pc.m_sun_radius[3] = m_vk->getGodRaysRadius();
    pc.m_color[0] = color.X;
    pc.m_color[1] = color.Y;
    pc.m_color[2] = color.Z;

    // 1. Every viewport marches its rays, with its own camera
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &m_sets[0], 0, NULL);
    for (unsigned i = 0; i < viewports.size(); i++)
    {
        viewports[i].m_draw_call->bindDataDescriptorSet(m_vk, cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        const std::array<int32_t, 4>& r = viewports[i].m_rect;
        for (unsigned j = 0; j < 4; j++)
            pc.m_rect[j] = r[j];
        vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
            0, sizeof(pc), &pc);
        // The area in the quarter size image, covers all the pixels of the
        // rect
        const std::array<int32_t, 4> q = quarterRect(r);
        const uint32_t w = (uint32_t)(q[2] - q[0]);
        const uint32_t h = (uint32_t)(q[3] - q[1]);
        vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    }
    computeToComputeBarrier(cmd, 0);

    // 2. Horizontal and then vertical blur, one dispatch per viewport, which
    // only reads inside of its own rect (a pixel at the border can be written
    // by two viewports, each with only its own pixels)
    VkPipeline bound = VK_NULL_HANDLE;
    for (unsigned pass = 0; pass < 2; pass++)
    {
        for (const Viewport& vp : viewports)
        {
            // Covers all the pixels of the rect, in the quarter size image
            const std::array<int32_t, 4> rect = quarterRect(vp.m_rect);
            // Independent of the size of the screen, the viewport (splitscreen)
            // and the render scale (m_height has both): the width is the
            // same fraction of the height of the viewport as the light
            // scattering, which is half of the size. The image is a quarter
            // of the size so a texel is twice as big, and the width is
            // not smaller than a quarter texel
            const float scale = std::max(0.5f * vp.m_height /
                REFERENCE_HEIGHT, 0.25f);
            dispatchBlur(cmd, m_blur_pipelines, &bound, m_sets[pass + 1],
                BM_GAUSSIAN, rect, rect, scale, 0.0f,
                pass == 0 ? 1 : 0, pass == 0 ? 0 : 1);
        }
        if (pass == 0)
            computeToComputeBarrier(cmd, 0);
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
