#include "ge_vulkan_depth_of_field.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <stdexcept>

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanDepthOfField::GEVulkanDepthOfField(GEVulkanDriver* vk,
    VkDescriptorSetLayout io_layout,
    const irr::core::dimension2d<irr::u32>& capacity)
                    : GEVulkanPostProcessing(vk, capacity),
                      m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
                      m_data_layout(VK_NULL_HANDLE),
                      m_pipeline_layout(VK_NULL_HANDLE)
{
    m_half.fill(NULL);
    m_sets.fill(VK_NULL_HANDLE);
    m_pipelines.fill(VK_NULL_HANDLE);
    try
    {
        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        const irr::core::dimension2d<irr::u32> half = halfSize(capacity);
        // rgba16f is a format every device supports as storage image
        for (GEVulkanAttachmentTexture*& t : m_half)
        {
            t = new GEVulkanAttachmentTexture(vk, half,
                VK_FORMAT_R16G16B16A16_SFLOAT, usage,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }
        // The images stay in general layout forever
        initializeGeneralImages(m_half.data(), m_half.size());
        createDescriptors();
        createPipelines(io_layout);
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanDepthOfField

// ----------------------------------------------------------------------------
GEVulkanDepthOfField::~GEVulkanDepthOfField()
{
    destroy();
}   // ~GEVulkanDepthOfField

// ----------------------------------------------------------------------------
void GEVulkanDepthOfField::destroy()
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
    if (m_data_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_data_layout, NULL);
    // Descriptor sets are freed with their pool
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_pool, NULL);
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_layout, NULL);
    for (GEVulkanAttachmentTexture*& t : m_half)
    {
        delete t;
        t = NULL;
    }
    m_pipeline_layout = VK_NULL_HANDLE;
    m_data_layout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;
}   // destroy

// ----------------------------------------------------------------------------
void GEVulkanDepthOfField::createDescriptors()
{
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = i == 0 ?
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
            "GEVulkanDepthOfField");
    }

    std::array<VkDescriptorPoolSize, 2> pool_sizes = {};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[0].descriptorCount = DS_COUNT;
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    pool_sizes[1].descriptorCount = DS_COUNT;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = DS_COUNT;
    pool_info.poolSizeCount = pool_sizes.size();
    pool_info.pPoolSizes = pool_sizes.data();
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanDepthOfField");
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
            "GEVulkanDepthOfField");
    }

    const VkImageView h0 = (VkImageView)m_half[0]->getTextureHandler();
    const VkImageView h1 = (VkImageView)m_half[1]->getTextureHandler();
    // Linear and clamped (dof_blur.comp samples between the pixels)
    VkSampler sampler = m_vk->getSampler(GVS_SKYBOX);
    // The binding which a shader doesn't use is written anyway (the other
    // half size image, so it never is one the pass writes)
    auto write = [&](DescriptorSet set, VkImageView in, VkImageView out)
    {
        std::array<VkDescriptorImageInfo, 2> infos = {};
        infos[0].sampler = sampler;
        infos[0].imageView = in;
        infos[0].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        infos[1].imageView = out;
        infos[1].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        std::array<VkWriteDescriptorSet, 2> writes = {};
        for (unsigned i = 0; i < writes.size(); i++)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = m_sets[set];
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = i == 0 ?
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &infos[i];
        }
        vkUpdateDescriptorSets(m_vk->getDevice(), writes.size(),
            writes.data(), 0, NULL);
    };
    write(DS_DOWNSAMPLE, h1, h0);
    write(DS_BLUR_H, h0, h1);
    write(DS_BLUR_V, h1, h0);
    write(DS_COMPOSITE, h0, h1);
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanDepthOfField::createPipelines(VkDescriptorSetLayout io_layout)
{
    m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(PushConstants);
    // The camera is set 1 in utils/camera.glsl
    std::array<VkDescriptorSetLayout, 3> layouts =
        {{ io_layout, m_data_layout, m_layout }};
    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = layouts.size();
    layout_info.pSetLayouts = layouts.data();
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(m_vk->getDevice(), &layout_info, NULL,
        &m_pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for "
            "GEVulkanDepthOfField");
    }
    const char* shaders[K_COUNT] =
    {
        "dof_copy.comp", "dof_downsample.comp", "dof_blur.comp",
        "dof_composite.comp"
    };
    for (unsigned i = 0; i < K_COUNT; i++)
    {
        VkComputePipelineCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = GEVulkanShaderManager::getShader(shaders[i]);
        info.stage.pName = "main";
        info.layout = m_pipeline_layout;
        if (vkCreateComputePipelines(m_vk->getDevice(),
            m_vk->getPipelineCache(), 1, &info, NULL, &m_pipelines[i]) !=
            VK_SUCCESS)
        {
            m_pipelines[i] = VK_NULL_HANDLE;
            throw std::runtime_error(std::string(
                "vkCreateComputePipelines failed for ") + shaders[i]);
        }
    }
}   // createPipelines

// ----------------------------------------------------------------------------
bool GEVulkanDepthOfField::isActive(const std::vector<std::pair<
    GEVulkanDrawCall*, GEVulkanCamera*> >& p) const
{
    // Same as the radius which apply() only copies the image for
    return getGEConfig()->m_dof_max_radius >= 0.5f;
}   // isActive

// ----------------------------------------------------------------------------
void GEVulkanDepthOfField::apply(VkCommandBuffer cmd, VkDescriptorSet io_set,
    const irr::core::dimension2d<irr::u32>& size,
    const std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCamera*> >& p)
{
    // The size of the FBO, which getViewport clamps the viewports to
    m_size = size;
    GEConfig* config = getGEConfig();

    struct Item
    {
        Viewport m_viewport;
        // Half size pixels which are entirely inside of the viewport
        std::array<int32_t, 4> m_half_rect;
    };
    std::vector<Item> items;
    // Nothing to blur: the image is only copied
    if (config->m_dof_max_radius >= 0.5f)
    {
        for (auto& q : p)
        {
            Item item;
            if (!getViewport(q.first, q.second, &item.m_viewport))
                continue;
            item.m_half_rect = innerHalfRect(item.m_viewport.m_rect);
            if (!isEmptyRect(item.m_half_rect))
                items.push_back(item);
        }
    }

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &io_set, 0, NULL);

    // Output = input outside of the viewports, which is the whole image
    // unless one viewport is all of it (no splitscreen)
    const std::array<int32_t, 4> full = {{ 0, 0, (int32_t)size.Width,
        (int32_t)size.Height }};
    if (items.size() != 1 || items[0].m_viewport.m_rect != full)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelines[K_COPY]);
        vkCmdDispatch(cmd, (size.Width + 7) / 8, (size.Height + 7) / 8, 1);
        // The viewports write the pixels the copy wrote
        if (!items.empty())
            computeToComputeBarrier(cmd, 0);
    }
    if (items.empty())
        return;

    PushConstants pc = {};
    pc.m_focus[0] = config->m_dof_focus_distance;
    pc.m_focus[1] = std::max(config->m_dof_focus_range, 0.0f);
    pc.m_focus[2] = std::max(config->m_dof_far_transition, 0.001f);
    pc.m_focus[3] = std::max(config->m_dof_near_transition, 0.0f);

    // Binds the pipeline and the half size images, then dispatches it for
    // every viewport (they don't overlap and only read what is inside of
    // their rect, so there is no barrier between them)
    auto run = [&](Kernel kernel, DescriptorSet set, bool full_size,
                   int32_t direction_x, int32_t direction_y, bool camera)
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipelines[kernel]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipeline_layout, 2, 1, &m_sets[set], 0, NULL);
        for (const Item& item : items)
        {
            if (camera)
            {
                // The camera of the viewport: the same unprojection as the
                // pass which wrote the depth, for any position on the screen
                item.m_viewport.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
                    VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
            }
            for (unsigned i = 0; i < 4; i++)
            {
                pc.m_rect[i] = item.m_viewport.m_rect[i];
                pc.m_half_rect[i] = item.m_half_rect[i];
            }
            // Radius in half size pixels: the same part of the height of the
            // viewport at any resolution, render scale and splitscreen
            pc.m_params[0] = config->m_dof_max_radius *
                (item.m_viewport.m_height / REFERENCE_HEIGHT) * 0.5f;
            pc.m_params[1] = (float)direction_x;
            pc.m_params[2] = (float)direction_y;
            vkCmdPushConstants(cmd, m_pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            const std::array<int32_t, 4>& r = full_size ?
                item.m_viewport.m_rect : item.m_half_rect;
            vkCmdDispatch(cmd, (r[2] - r[0] + 7) / 8, (r[3] - r[1] + 7) / 8,
                1);
        }
    };

    run(K_DOWNSAMPLE, DS_DOWNSAMPLE, false, 0, 0, true);
    computeToComputeBarrier(cmd, 0);
    run(K_BLUR, DS_BLUR_H, false, 1, 0, false);
    computeToComputeBarrier(cmd, 0);
    run(K_BLUR, DS_BLUR_V, false, 0, 1, false);
    computeToComputeBarrier(cmd, 0);
    run(K_COMPOSITE, DS_COMPOSITE, true, 0, 0, true);
}   // apply

}
