#include "ge_vulkan_low_res_effect.hpp"

#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_main.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanLowResEffect::GEVulkanLowResEffect(GEVulkanDriver* vk,
                              const irr::core::dimension2d<irr::u32>& size,
                              GEVulkanAttachmentTexture* depth,
                              const Params& params)
                    : GEVulkanPostProcessing(vk, size), m_params(params),
                      m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
                      m_data_layout(VK_NULL_HANDLE),
                      m_pipeline_layout(VK_NULL_HANDLE),
                      m_pipeline(VK_NULL_HANDLE)
{
    m_images.fill(NULL);
    m_sets.fill(VK_NULL_HANDLE);
    // No alpha is needed and the values are never negative, B10G11R11 has the
    // range and half of the memory of RGBA16F (storage image support of it
    // is optional)
    VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
    if (GEVulkanFeatures::supportsShaderStorageImageExtendedFormats())
        format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    try
    {
        const irr::core::dimension2d<irr::u32> small =
            downscaledSize(size, m_params.m_shift);
        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        for (GEVulkanAttachmentTexture*& t : m_images)
        {
            t = new GEVulkanAttachmentTexture(vk, small, format, usage,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }

        // The images stay in general layout forever
        initializeGeneralImages(m_images.data(), m_images.size());

        createDescriptors(depth);
        createBlurPipeline();
        createPipeline();
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanLowResEffect

// ----------------------------------------------------------------------------
GEVulkanLowResEffect::~GEVulkanLowResEffect()
{
    destroy();
}   // ~GEVulkanLowResEffect

// ----------------------------------------------------------------------------
void GEVulkanLowResEffect::destroy()
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
    for (GEVulkanAttachmentTexture* t : m_images)
        delete t;
}   // destroy

// ----------------------------------------------------------------------------
irr::core::dimension2d<irr::u32> GEVulkanLowResEffect::downscaledSize(
    const irr::core::dimension2d<irr::u32>& size, unsigned shift)
{
    const irr::u32 add = (1u << shift) - 1;
    return irr::core::dimension2d<irr::u32>(
        std::max(1u, (size.Width + add) >> shift),
        std::max(1u, (size.Height + add) >> shift));
}   // downscaledSize

// ----------------------------------------------------------------------------
std::array<int32_t, 4> GEVulkanLowResEffect::downscaledRect(
    const std::array<int32_t, 4>& r, unsigned shift)
{
    const int32_t add = (1 << shift) - 1;
    return {{ r[0] >> shift, r[1] >> shift, (r[2] + add) >> shift,
        (r[3] + add) >> shift }};
}   // downscaledRect

// ----------------------------------------------------------------------------
VkImageView GEVulkanLowResEffect::getImageView() const
{
    return (VkImageView)m_images[0]->getTextureHandler();
}   // getImageView

// ----------------------------------------------------------------------------
void GEVulkanLowResEffect::createDescriptors(GEVulkanAttachmentTexture* depth)
{
    createImageDescriptorSetLayout(&m_layout);
    createImageDescriptorPool(m_sets.size(), &m_pool);
    allocateImageDescriptorSets(m_pool, m_layout, m_sets.size(),
        m_sets.data());

    // Index of m_images, -1 is the depth
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
                (VkImageView)m_images[inputs[i]]->getTextureHandler();
        }

        writeImageDescriptorSet(m_sets[i], input_view, sampler, input_layout,
            (VkImageView)m_images[outputs[i]]->getTextureHandler(),
            VK_IMAGE_LAYOUT_GENERAL);
    }
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanLowResEffect::createBlurPipeline()
{
    createBlurPipelines("blur_hdr.comp", m_layout,
        blurModeBit(BM_GAUSSIAN), &m_blur_pipelines);
}   // createBlurPipeline

// ----------------------------------------------------------------------------
void GEVulkanLowResEffect::createPipeline()
{
    VkDevice device = m_vk->getDevice();
    m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = m_params.m_push_constant_size;
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
        throw std::runtime_error(std::string("vkCreatePipelineLayout failed "
            "for ") + m_params.m_name);
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module =
        GEVulkanShaderManager::getShader(m_params.m_shader);
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_pipeline_layout;
    if (vkCreateComputePipelines(device, m_vk->getPipelineCache(), 1,
        &pipeline_info, NULL, &m_pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkCreateComputePipelines "
            "failed for ") + m_params.m_name);
    }
}   // createPipeline

// ----------------------------------------------------------------------------
bool GEVulkanLowResEffect::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    std::vector<Viewport> viewports;
    for (auto& q : p)
    {
        Viewport vp;
        if (hasEffect(q.first) && getViewport(q.first, q.second, &vp))
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

    // 1. Every viewport marches, with its own camera (and lights)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &m_sets[0], 0, NULL);
    for (const Viewport& vp : viewports)
    {
        vp.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        pushConstants(cmd, vp);
        // The area in the image, covers all the pixels of the rect
        const std::array<int32_t, 4> r =
            downscaledRect(vp.m_rect, m_params.m_shift);
        const uint32_t w = (uint32_t)(r[2] - r[0]);
        const uint32_t h = (uint32_t)(r[3] - r[1]);
        vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
    }
    computeToComputeBarrier(cmd, 0);

    // 2. Horizontal and then vertical blur, one dispatch per viewport, which
    // only reads inside of its own rect (a pixel at the border can be
    // written by two viewports, each with only its own pixels)
    VkPipeline bound = VK_NULL_HANDLE;
    for (unsigned pass = 0; pass < 2; pass++)
    {
        for (const Viewport& vp : viewports)
        {
            // Covers all the pixels of the rect, in the image
            const std::array<int32_t, 4> rect =
                downscaledRect(vp.m_rect, m_params.m_shift);
            // Independent of the size of the screen, the viewport
            // (splitscreen) and the render scale (m_height has both)
            const float scale = std::min(std::max(vp.m_height /
                REFERENCE_HEIGHT * m_params.m_blur_scale_factor,
                m_params.m_blur_scale_min), m_params.m_blur_scale_max);
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
