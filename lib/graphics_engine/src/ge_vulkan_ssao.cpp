#include "ge_vulkan_ssao.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_array_texture.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace GE
{
namespace
{
// Same as the push constants of utils/ssao_common.glsl
struct SSAOPushConstants
{
    int32_t m_rect[4];
    float m_params[4];
    int32_t m_misc[4];
};

// ----------------------------------------------------------------------------
// Number of groups (8x8 threads) which cover the area of a rect
uint32_t groups(int32_t size)
{
    return (uint32_t)(std::max(size, 0) + 7) / 8;
}   // groups

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanSSAO::GEVulkanSSAO(GEVulkanDriver* vk,
                           const irr::core::dimension2d<irr::u32>& size,
                           GEVulkanAttachmentTexture* depth,
                           GEVulkanAttachmentTexture* normal,
                           GEVulkanAttachmentTexture* light)
            : GEVulkanPostProcessing(vk, size), m_depth(NULL), m_ao(NULL),
              m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
              m_data_layout(VK_NULL_HANDLE), m_pipeline_layout(VK_NULL_HANDLE),
              m_depth_pipeline(VK_NULL_HANDLE), m_down_pipeline(VK_NULL_HANDLE),
              m_ao_pipeline(VK_NULL_HANDLE), m_blur_pipeline(VK_NULL_HANDLE),
              m_upsample_pipeline(VK_NULL_HANDLE),
              m_ssil_pipeline(VK_NULL_HANDLE),
              m_ssil_blur_pipeline(VK_NULL_HANDLE)
{
    m_depth_views.fill(VK_NULL_HANDLE);
    m_ao_half.fill(NULL);
    m_ssil_half.fill(NULL);
    m_sets.fill(VK_NULL_HANDLE);
    try
    {
        // Linear depth of the sky is large, 32 bit float is needed for the
        // precision far away
        m_depth = new GEVulkanArrayTexture(VK_FORMAT_R32_SFLOAT,
            VK_IMAGE_VIEW_TYPE_2D, size, 1, irr::video::SColor(0),
            VK_IMAGE_LAYOUT_GENERAL);
        if (m_depth->getMipmapLevels() < DEPTH_LEVELS)
            throw std::runtime_error("FBO is too small for GEVulkanSSAO");

        VkDevice device = m_vk->getDevice();
        for (unsigned i = 0; i < m_depth_views.size(); i++)
        {
            VkImageViewCreateInfo view_info = {};
            view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            view_info.image = m_depth->getImage();
            view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            view_info.format = m_depth->getInternalFormat();
            view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            view_info.subresourceRange.baseMipLevel = i;
            view_info.subresourceRange.levelCount = 1;
            view_info.subresourceRange.baseArrayLayer = 0;
            view_info.subresourceRange.layerCount = 1;
            if (vkCreateImageView(device, &view_info, NULL,
                &m_depth_views[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("vkCreateImageView failed for "
                    "GEVulkanSSAO");
            }
        }

        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const irr::core::dimension2d<irr::u32> half = halfSize(size);
        // Visibility in [0, 1], R8 needs the extended storage formats
        VkFormat ao_format = VK_FORMAT_R8G8B8A8_UNORM;
        if (GEVulkanFeatures::supportsShaderStorageImageExtendedFormats())
            ao_format = VK_FORMAT_R8_UNORM;
        for (GEVulkanAttachmentTexture*& t : m_ao_half)
        {
            t = new GEVulkanAttachmentTexture(vk, half, ao_format, usage,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }
        m_ao = new GEVulkanAttachmentTexture(vk, size, ao_format, usage,
            VK_IMAGE_ASPECT_COLOR_BIT);

        // The indirect light is hdr, rg16f isn't a storage format
        for (GEVulkanAttachmentTexture*& t : m_ssil_half)
        {
            t = new GEVulkanAttachmentTexture(vk, half,
                VK_FORMAT_R16G16B16A16_SFLOAT, usage,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }

        // The images stay in general layout forever (m_depth was created in
        // it)
        std::array<GEVulkanAttachmentTexture*, 5> images =
            {{ m_ao_half[0], m_ao_half[1], m_ao, m_ssil_half[0],
            m_ssil_half[1] }};
        initializeGeneralImages(images.data(), images.size());

        createDescriptors(depth, normal, light);
        createPipelines();
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanSSAO

// ----------------------------------------------------------------------------
GEVulkanSSAO::~GEVulkanSSAO()
{
    destroy();
}   // ~GEVulkanSSAO

// ----------------------------------------------------------------------------
void GEVulkanSSAO::destroy()
{
    VkDevice device = m_vk->getDevice();
    for (VkPipeline* pipeline : { &m_depth_pipeline, &m_down_pipeline,
        &m_ao_pipeline, &m_blur_pipeline, &m_upsample_pipeline,
        &m_ssil_pipeline, &m_ssil_blur_pipeline })
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
    for (VkImageView view : m_depth_views)
    {
        if (view != VK_NULL_HANDLE)
            vkDestroyImageView(device, view, NULL);
    }
    delete m_ao;
    for (GEVulkanAttachmentTexture* t : m_ao_half)
        delete t;
    for (GEVulkanAttachmentTexture* t : m_ssil_half)
        delete t;
    delete m_depth;
    m_ao = NULL;
    m_ao_half.fill(NULL);
    m_ssil_half.fill(NULL);
    m_depth = NULL;
    m_depth_views.fill(VK_NULL_HANDLE);
    m_pipeline_layout = VK_NULL_HANDLE;
    m_data_layout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
}   // destroy

// ----------------------------------------------------------------------------
VkImageView GEVulkanSSAO::getImageView() const
{
    return (VkImageView)m_ao->getTextureHandler();
}   // getImageView

// ----------------------------------------------------------------------------
VkImageView GEVulkanSSAO::getSSILImageView() const
{
    return (VkImageView)m_ssil_half[0]->getTextureHandler();
}   // getSSILImageView

// ----------------------------------------------------------------------------
void GEVulkanSSAO::createDescriptors(GEVulkanAttachmentTexture* depth,
                                     GEVulkanAttachmentTexture* normal,
                                     GEVulkanAttachmentTexture* light)
{
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, 5> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        // 4 is the light of ssil.comp, the other inputs are 0 and 1
        bindings[i].descriptorType = (i < 2 || i == 4) ?
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layout_info, NULL,
        &m_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorSetLayout failed for "
            "GEVulkanSSAO");
    }

    std::array<VkDescriptorPoolSize, 2> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount = DS_COUNT * 3;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = DS_COUNT * 2;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = DS_COUNT;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanSSAO");
    }

    std::array<VkDescriptorSetLayout, DS_COUNT> layouts;
    layouts.fill(m_layout);
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = m_sets.size();
    alloc_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &alloc_info, m_sets.data()) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanSSAO");
    }

    const VkImageView depth_view = (VkImageView)depth->getTextureHandler();
    const VkImageView normal_view = (VkImageView)normal->getTextureHandler();
    const VkImageView chain_view = (VkImageView)m_depth->getTextureHandler();
    const VkImageView ao_half_0 =
        (VkImageView)m_ao_half[0]->getTextureHandler();
    const VkImageView ao_half_1 =
        (VkImageView)m_ao_half[1]->getTextureHandler();
    const VkImageView ao_full = (VkImageView)m_ao->getTextureHandler();
    const VkImageView light_view = (VkImageView)light->getTextureHandler();
    const VkImageView ssil_half_0 =
        (VkImageView)m_ssil_half[0]->getTextureHandler();
    const VkImageView ssil_half_1 =
        (VkImageView)m_ssil_half[1]->getTextureHandler();

    struct Input
    {
        VkImageView m_view;
        VkImageLayout m_layout;
    };
    const Input depth_input =
        { depth_view, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
    const Input normal_input =
        { normal_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    const Input chain_input = { chain_view, VK_IMAGE_LAYOUT_GENERAL };
    const Input half_0_input = { ao_half_0, VK_IMAGE_LAYOUT_GENERAL };
    const Input half_1_input = { ao_half_1, VK_IMAGE_LAYOUT_GENERAL };
    const Input light_input =
        { light_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    const Input ssil_0_input = { ssil_half_0, VK_IMAGE_LAYOUT_GENERAL };
    const Input ssil_1_input = { ssil_half_1, VK_IMAGE_LAYOUT_GENERAL };

    // Binding 1, 3 and 4 are not used by every shader, the same as binding 0
    // and 2 then
    auto write = [this](DescriptorSet set, const Input& a, const Input& b,
                        const Input& light_in, VkImageView out_0,
                        VkImageView out_1)
    {
        const Input* inputs[3] = { &a, &b, &light_in };
        std::array<VkDescriptorImageInfo, 5> infos = {};
        for (unsigned i = 0; i < infos.size(); i++)
        {
            if (i >= 2 && i < 4)
                continue;
            const Input& input = *inputs[i < 2 ? i : 2];
            infos[i].sampler = m_vk->getSampler(GVS_NEAREST);
            infos[i].imageView = input.m_view;
            infos[i].imageLayout = input.m_layout;
        }
        infos[2].imageView = out_0;
        infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        infos[3].imageView = out_1;
        infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        std::array<VkWriteDescriptorSet, 5> writes = {};
        for (unsigned i = 0; i < writes.size(); i++)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = m_sets[set];
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = (i < 2 || i == 4) ?
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &infos[i];
        }
        vkUpdateDescriptorSets(m_vk->getDevice(), writes.size(),
            writes.data(), 0, NULL);
    };
    write(DS_DEPTH, depth_input, depth_input, light_input, m_depth_views[0],
        m_depth_views[1]);
    write(DS_DOWN_1, chain_input, chain_input, light_input, m_depth_views[2],
        m_depth_views[2]);
    write(DS_DOWN_2, chain_input, chain_input, light_input, m_depth_views[3],
        m_depth_views[3]);
    write(DS_AO, chain_input, normal_input, light_input, ao_half_0, ao_half_0);
    write(DS_BLUR_H, half_0_input, chain_input, light_input, ao_half_1,
        ao_half_1);
    write(DS_BLUR_V, half_1_input, chain_input, light_input, ao_half_0,
        ao_half_0);
    write(DS_UPSAMPLE, half_0_input, chain_input, light_input, ao_full,
        ao_full);
    write(DS_SSIL, chain_input, normal_input, light_input, ssil_half_0,
        ssil_half_0);
    write(DS_SSIL_BLUR_H, ssil_0_input, chain_input, light_input, ssil_half_1,
        ssil_half_1);
    write(DS_SSIL_BLUR_V, ssil_1_input, chain_input, light_input, ssil_half_0,
        ssil_half_0);
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanSSAO::createPipeline(const char* shader,
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
            "failed for GEVulkanSSAO ") + shader);
    }
}   // createPipeline

// ----------------------------------------------------------------------------
void GEVulkanSSAO::createPipelines()
{
    m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(SSAOPushConstants);
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
    if (vkCreatePipelineLayout(m_vk->getDevice(), &pipeline_layout_info, NULL,
        &m_pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for "
            "GEVulkanSSAO");
    }
    createPipeline("ssao_depth.comp", &m_depth_pipeline);
    createPipeline("ssao_depth_down.comp", &m_down_pipeline);
    createPipeline("ssao.comp", &m_ao_pipeline);
    createPipeline("ssao_blur.comp", &m_blur_pipeline);
    createPipeline("ssao_upsample.comp", &m_upsample_pipeline);
    createPipeline("ssil.comp", &m_ssil_pipeline);
    createPipeline("ssil_blur.comp", &m_ssil_blur_pipeline);
}   // createPipelines

// ----------------------------------------------------------------------------
bool GEVulkanSSAO::getViewports(
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p, std::vector<Viewport>* out) const
{
    for (auto& q : p)
    {
        Viewport vp;
        if (getViewport(q.first, q.second, &vp))
            out->push_back(vp);
    }
    return !out->empty();
}   // getViewports

// ----------------------------------------------------------------------------
void GEVulkanSSAO::dispatch(VkCommandBuffer cmd,
    const std::vector<Viewport>& viewports, VkPipeline pipeline,
    DescriptorSet set, Area area, int32_t level, int32_t direction_x,
    int32_t direction_y) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &m_sets[set], 0, NULL);

    SSAOPushConstants pc = {};
    // VBAO only has a radius, m_ssao_sigma and m_ssao_k are parameters of the
    // estimator of the OpenGL renderer
    pc.m_params[0] = getGEConfig()->m_ssao_radius;
    for (const Viewport& vp : viewports)
    {
        // The camera of the viewport (ssao_depth.comp, ssao.comp, ssil.comp)
        vp.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        const std::array<int32_t, 4>& r = vp.m_rect;
        for (unsigned i = 0; i < 4; i++)
            pc.m_rect[i] = r[i];
        // Same as the other blurs (viewport height, which has the render
        // scale in it), used by ssao_blur.comp and ssil_blur.comp
        pc.m_params[3] = vp.m_height / REFERENCE_HEIGHT;
        pc.m_misc[0] = level;
        pc.m_misc[1] = direction_x;
        pc.m_misc[2] = direction_y;
        vkCmdPushConstants(cmd, m_pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        std::array<int32_t, 4> area_rect = r;
        if (area == AREA_HALF)
            area_rect = halfRect(r);
        else if (area == AREA_NEXT_LEVEL)
        {
            const int32_t round_up = (1 << level) - 1;
            area_rect = halfRect({{ r[0] >> level, r[1] >> level,
                (r[2] + round_up) >> level, (r[3] + round_up) >> level }});
        }
        vkCmdDispatch(cmd, groups(area_rect[2] - area_rect[0]),
            groups(area_rect[3] - area_rect[1]), 1);
    }
}   // dispatch

// ----------------------------------------------------------------------------
bool GEVulkanSSAO::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    std::vector<Viewport> viewports;
    if (!getViewports(p, &viewports))
        return false;

    // The depth and the normal of the g-buffer are written, and what the last
    // frame read or wrote of the images has to be finished
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

    // 1. Level 0 and 1 of the depth chain
    dispatch(cmd, viewports, m_depth_pipeline, DS_DEPTH, AREA_HALF, 0, 0, 0);
    computeToComputeBarrier(cmd, 0);

    // 2. The next levels, one after another
    dispatch(cmd, viewports, m_down_pipeline, DS_DOWN_1, AREA_NEXT_LEVEL, 1,
        0, 0);
    computeToComputeBarrier(cmd, 0);
    dispatch(cmd, viewports, m_down_pipeline, DS_DOWN_2, AREA_NEXT_LEVEL, 2,
        0, 0);
    computeToComputeBarrier(cmd, 0);

    // 3. Occlusion at half the size
    dispatch(cmd, viewports, m_ao_pipeline, DS_AO, AREA_HALF, 0, 0, 0);
    computeToComputeBarrier(cmd, 0);

    // 4. Horizontal and then vertical blur
    dispatch(cmd, viewports, m_blur_pipeline, DS_BLUR_H, AREA_HALF, 0, 1, 0);
    computeToComputeBarrier(cmd, 0);
    dispatch(cmd, viewports, m_blur_pipeline, DS_BLUR_V, AREA_HALF, 0, 0, 1);
    computeToComputeBarrier(cmd, 0);

    // 5. Full size
    dispatch(cmd, viewports, m_upsample_pipeline, DS_UPSAMPLE, AREA_FULL, 0,
        0, 0);

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
    // The depth and the normal are read by the compute shaders (execution
    // dependency only), so the g-buffer of the next frame has to wait for them
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 0,
        NULL);
    return true;
}   // render

// ----------------------------------------------------------------------------
bool GEVulkanSSAO::renderIndirect(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    std::vector<Viewport> viewports;
    if (!getViewports(p, &viewports))
        return false;

    // The hdr was written by the lighting pass, the normal, the depth chain
    // and the light of the samples are read, and what the last frame read or
    // wrote of the images has to be finished (the same as GEVulkanBloom)
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }

    // 1. Indirect light at half the size
    dispatch(cmd, viewports, m_ssil_pipeline, DS_SSIL, AREA_HALF, 0, 0, 0);
    computeToComputeBarrier(cmd, 0);

    // 2. Horizontal and then vertical blur
    dispatch(cmd, viewports, m_ssil_blur_pipeline, DS_SSIL_BLUR_H, AREA_HALF,
        0, 1, 0);
    computeToComputeBarrier(cmd, 0);
    dispatch(cmd, viewports, m_ssil_blur_pipeline, DS_SSIL_BLUR_V, AREA_HALF,
        0, 0, 1);

    // Sampled by the tonemap pass
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }
    // The normal and the depth chain are read by the compute shaders
    // (execution dependency only), so the g-buffer of the next frame has to
    // wait for them
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 0,
        NULL);
    return true;
}   // renderIndirect

}
