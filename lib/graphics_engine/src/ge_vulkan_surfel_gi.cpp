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


#include "../source/Irrlicht/os.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>

namespace GE
{
// The reference grid (d = 96, n = 64, so a cell is 3 units wide and a surfel
// is at most 1.5 units large). A larger surfel covers more of the screen, so
// the pool needs fewer of them and the spawn probabilities are lower (see
// surfel_gi_generate.comp) while the amount of light a pixel receives stays
// the same: surfels are also larger (SURFEL_SIZE is the area of a surfel in
// pixels at the near field)
const float GEVulkanSurfelGI::CELL_DELTA = 3.0f;
const float GEVulkanSurfelGI::CELL_HALF_DELTA = 96.0f;
const float GEVulkanSurfelGI::SURFEL_SIZE = 4.0f;

namespace
{
// Same as the push constants of utils/surfel_gi.glsl
struct SurfelGIPushConstants
{
    // xyz world position of the camera of the first viewport, the center of
    // the inner cube of the grid
    float m_origin[4];
    // x0, y0, x1, y1 (exclusive) in pixels of the g-buffer of the viewport
    int32_t m_rect[4];
    // Frame of the pool, 1 on the first frame of the pool, and the maximum
    // number of rays of a surfel
    uint32_t m_frame;
    uint32_t m_first_frame;
    uint32_t m_max_rays;
    // Size of a cell of the inner cube, half its size, vertical fov of the
    // camera and area of a surfel in pixels
    float m_delta;
    float m_half_d;
    float m_fov;
    float m_surfel_size;
    // Size of the viewport (for the radius of the surfels)
    uint32_t m_full_width;
    uint32_t m_full_height;
};
static_assert(sizeof(SurfelGIPushConstants) == 68,
    "SurfelGIPushConstants is 68 bytes in the shaders");

// ----------------------------------------------------------------------------
// Number of groups of size threads which cover count
uint32_t groups(uint32_t count, uint32_t size)
{
    return (count + size - 1) / size;
}

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanSurfelGI::GEVulkanSurfelGI(GEVulkanDriver* vk,
                                   const irr::core::dimension2d<irr::u32>& size,
                                   GEVulkanAttachmentTexture* depth,
                                   GEVulkanAttachmentTexture* normal)
                    : GEVulkanPostProcessing(vk, size),
                      m_gi(NULL), m_surfel_irradiance(NULL), m_surfel_depth(NULL),
                      m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
                      m_set(VK_NULL_HANDLE), m_data_layout(VK_NULL_HANDLE),
                      m_pipeline_layout(VK_NULL_HANDLE),
                      m_pipelines_created(false), m_frame(0),
                      m_rt_build_counter(0), m_needs_clear(true),
                      m_needs_image_clear(true)
{
    for (Buffer& buffer : m_buffers)
    {
        buffer.m_buffer = VK_NULL_HANDLE;
        buffer.m_memory = VK_NULL_HANDLE;
        buffer.m_size = 0;
    }
    m_pipelines.fill(VK_NULL_HANDLE);
    try
    {
        if (!createBuffers())
        {
            throw std::runtime_error("Failed to create the buffers of "
                "GEVulkanSurfelGI");
        }
        // Half the size of the render target, hdr values
        m_gi = new GEVulkanAttachmentTexture(vk, halfSize(size),
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
            VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
        // The 6x6 tile of every surfel, the ray guiding map (the tiles are
        // cleared by surfel_gi_integrate.comp when a surfel is new)
        m_surfel_irradiance = new GEVulkanAttachmentTexture(vk,
            irr::core::dimension2d<irr::u32>(IRRADIANCE_COLUMNS * IRRADIANCE_TILE,
                IRRADIANCE_ROWS * IRRADIANCE_TILE), VK_FORMAT_R16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        // Matching depth-moment tiles, initialized by the integration shader
        m_surfel_depth = new GEVulkanAttachmentTexture(vk,
            irr::core::dimension2d<irr::u32>(IRRADIANCE_COLUMNS * IRRADIANCE_TILE,
                IRRADIANCE_ROWS * IRRADIANCE_TILE), VK_FORMAT_R32G32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT);
        std::array<GEVulkanAttachmentTexture*, 3> general_images =
        {
            m_gi, m_surfel_irradiance, m_surfel_depth
        };
        initializeGeneralImages(general_images.data(), general_images.size());
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
bool GEVulkanSurfelGI::createBuffers()
{
    const VkDeviceSize sizes[BI_COUNT] =
    {
        sizeof(SurfelCounter),
        sizeof(Surfel) * MAX_SURFELS,
        sizeof(uint32_t) * MAX_SURFELS,
        sizeof(uint32_t) * 2 * CELL_COUNT,
        sizeof(uint32_t) * CELL_SURFELS,
        sizeof(uint32_t) * RAY_BUDGET,
        sizeof(float) * 4 * RAY_BUDGET,
        sizeof(SurfelLight) * MAX_SURFELS,
        sizeof(float) * 2 * RAY_BUDGET,
        // Three VkDispatchIndirectCommands: tracing, then the cooperative
        // integration, then the irradiance sharing
        sizeof(uint32_t) * 9,
        sizeof(uint32_t) * MAX_SURFELS
    };
    // Only surfels need clearing; the other buffers are written before reading
    const bool clearable[BI_COUNT] =
    {
        false, true, false, false, false, false, false, false, false, false,
        false
    };
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    alloc_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    for (unsigned i = 0; i < BI_COUNT; i++)
    {
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (clearable[i])
            usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        // Tracing and integration groups are read by vkCmdDispatchIndirect
        if (i == BI_DISPATCH)
            usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (!m_vk->createBuffer(sizes[i], usage, alloc_info,
            m_buffers[i].m_buffer, m_buffers[i].m_memory))
            return false;
        m_buffers[i].m_size = sizes[i];
    }
    return true;
}   // createBuffers

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::destroy()
{
    m_vk->waitIdle();
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
    delete m_gi;
    m_gi = NULL;
    delete m_surfel_irradiance;
    m_surfel_irradiance = NULL;
    delete m_surfel_depth;
    m_surfel_depth = NULL;
    for (Buffer& buffer : m_buffers)
    {
        if (buffer.m_buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_vk->getVmaAllocator(), buffer.m_buffer,
                buffer.m_memory);
        }
        buffer.m_buffer = VK_NULL_HANDLE;
        buffer.m_memory = VK_NULL_HANDLE;
        buffer.m_size = 0;
    }
    m_pipeline_layout = VK_NULL_HANDLE;
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
void GEVulkanSurfelGI::createDescriptors(GEVulkanAttachmentTexture* depth,
                                         GEVulkanAttachmentTexture* normal)
{
    VkDevice device = m_vk->getDevice();
    // 0 normal, 1 depth, 2 irradiance, 3..9 pool/cells/rays, 10/11 irradiance
    // tiles, 12 compact surfels, 13 ray PDF/distance, 14 trace/integration
    // groups, 15/16 depth moments, 17 active surfel IDs
    std::array<VkDescriptorSetLayoutBinding, 18> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        if (i < 2 || i == 10 || i == 15)
            bindings[i].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        else if (i == 2 || i == 11 || i == 16)
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        else
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
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
    pool_sizes[0].descriptorCount = 4;
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
    if (vkAllocateDescriptorSets(device, &alloc_info, &m_set) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanSurfelGI");
    }

    std::array<VkDescriptorImageInfo, 7> image_infos = {};
    image_infos[0].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[0].imageView = (VkImageView)normal->getTextureHandler();
    image_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_infos[1].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[1].imageView = (VkImageView)depth->getTextureHandler();
    image_infos[1].imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    image_infos[2].imageView = (VkImageView)m_gi->getTextureHandler();
    image_infos[2].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    // The maps are read with texelFetch (the values of the texels), the
    // sampler is only there because they're combined image samplers
    image_infos[3].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[3].imageView =
        (VkImageView)m_surfel_irradiance->getTextureHandler();
    image_infos[3].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[4].imageView =
        (VkImageView)m_surfel_irradiance->getTextureHandler();
    image_infos[4].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[5].sampler = m_vk->getSampler(GVS_NEAREST);
    image_infos[5].imageView = (VkImageView)m_surfel_depth->getTextureHandler();
    image_infos[5].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    image_infos[6].imageView = (VkImageView)m_surfel_depth->getTextureHandler();
    image_infos[6].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkDescriptorBufferInfo, BI_COUNT> buffer_infos = {};
    for (unsigned i = 0; i < BI_COUNT; i++)
    {
        buffer_infos[i].buffer = m_buffers[i].m_buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = m_buffers[i].m_size;
    }
    // Same binding order as utils/surfel_gi.glsl (-1 for image bindings)
    const int buffer_order[18] =
    {
        -1, -1, -1,
        BI_SURFEL, BI_COUNTER, BI_DEAD, BI_CELL, BI_CELL_SURFEL, BI_RAY_ID,
        BI_RAY_RESULT,
        -1, -1,
        BI_SURFEL_LIGHT, BI_RAY_DATA, BI_DISPATCH,
        -1, -1, BI_ACTIVE
    };

    std::array<VkWriteDescriptorSet, 18> writes = {};
    for (unsigned i = 0; i < writes.size(); i++)
    {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        if (i < 3)
        {
            writes[i].descriptorType = i < 2 ?
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER :
                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &image_infos[i];
        }
        else if (i == 10)
        {
            writes[i].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &image_infos[3];
        }
        else if (i == 11)
        {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &image_infos[4];
        }
        else if (i == 15)
        {
            writes[i].descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[i].pImageInfo = &image_infos[5];
        }
        else if (i == 16)
        {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[i].pImageInfo = &image_infos[6];
        }
        else
        {
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo = &buffer_infos[buffer_order[i]];
        }
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, NULL);
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createPipeline(const char* shader,
                                      VkPipeline* pipeline,
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
    if (pipeline_info.stage.module == VK_NULL_HANDLE ||
        vkCreateComputePipelines(m_vk->getDevice(), m_vk->getPipelineCache(),
        1, &pipeline_info, NULL, pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkCreateComputePipelines "
            "failed for GEVulkanSurfelGI ") + shader);
    }
}   // createPipeline

// ----------------------------------------------------------------------------
void GEVulkanSurfelGI::createPipelines()
{
    GEVulkanRTScene* rt_scene = m_vk->getRTScene();
    if (!rt_scene)
        throw std::runtime_error("No ray traced scene for GEVulkanSurfelGI");
    m_data_layout = GEVulkanDrawCall::createDataLayout(m_vk);
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(SurfelGIPushConstants);
    // Same set order as solid_rt.frag (the scene, the data of the draw call,
    // this pass, the mesh textures of the hits and the environment)
    std::array<VkDescriptorSetLayout, 5> layouts =
    {{
        rt_scene->getDescriptorSetLayout(),
        m_data_layout,
        m_layout,
        *m_vk->getMeshTextureDescriptor()->getDescriptorSetLayout(),
        m_vk->getSkyBoxRenderer()->getEnvDescriptorSetLayout()
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
            "GEVulkanSurfelGI");
    }
    createPipeline("surfel_gi_prepare.comp", &m_pipelines[PI_PREPARE]);
    createPipeline("surfel_gi_update.comp", &m_pipelines[PI_UPDATE]);
    createPipeline("surfel_gi_cell_info.comp", &m_pipelines[PI_CELL_INFO]);
    createPipeline("surfel_gi_cell_to_surfel.comp",
        &m_pipelines[PI_CELL_TO_SURFEL]);

    // The same specialization constants as the ray traced reflection (a hit is
    // shaded by the same code, utils/rt_reflection.glsl): the ambient of the
    // environment and the sun shadow of a hit
    struct TraceConstants
    {
        VkBool32 m_ibl;
        float m_specular_levels_minus_one;
        uint32_t m_rt_shadow_type;
        uint32_t m_shadow_size;
        uint32_t m_shadow_type;
    };
    TraceConstants constants = {};
    constants.m_ibl = getGEConfig()->m_ibl &&
        GEVulkanFeatures::supportsComputeInMainQueue();
    float ts = GEVulkanEnvironmentMap::getSpecularEnvironmentMapSize().Width;
    constants.m_specular_levels_minus_one = std::floor(std::log2(ts));
    constants.m_rt_shadow_type = (uint32_t)getRTShadowType();
    // The hit of a ray of the surfels is lit like the one of a reflection
    // (utils/rt_reflection.glsl): the sun has a shadow if it has a shadow map
    // or it's ray traced
    constants.m_shadow_size = getShadowSize();
    constants.m_shadow_type = (uint32_t)getShadowType();
    std::array<VkSpecializationMapEntry, 5> entries = {};
    entries[0].constantID = 0;
    entries[0].offset = offsetof(TraceConstants, m_ibl);
    entries[0].size = sizeof(VkBool32);
    entries[1].constantID = 1;
    entries[1].offset = offsetof(TraceConstants, m_specular_levels_minus_one);
    entries[1].size = sizeof(float);
    entries[2].constantID = 13;
    entries[2].offset = offsetof(TraceConstants, m_rt_shadow_type);
    entries[2].size = sizeof(uint32_t);
    entries[3].constantID = 5;
    entries[3].offset = offsetof(TraceConstants, m_shadow_size);
    entries[3].size = sizeof(uint32_t);
    entries[4].constantID = 6;
    entries[4].offset = offsetof(TraceConstants, m_shadow_type);
    entries[4].size = sizeof(uint32_t);
    VkSpecializationInfo specialization = {};
    specialization.mapEntryCount = entries.size();
    specialization.pMapEntries = entries.data();
    specialization.dataSize = sizeof(TraceConstants);
    specialization.pData = &constants;
    createPipeline("surfel_gi_trace.comp", &m_pipelines[PI_TRACE],
        &specialization);

    createPipeline("surfel_gi_integrate.comp", &m_pipelines[PI_INTEGRATE]);
    createPipeline("surfel_gi_share.comp", &m_pipelines[PI_SHARE]);
    createPipeline("surfel_gi_generate.comp", &m_pipelines[PI_GENERATE]);
}   // createPipelines

// ----------------------------------------------------------------------------
bool GEVulkanSurfelGI::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCamera*> >& p)
{
    std::vector<Viewport> viewports;
    GEVulkanCamera* first_camera = NULL;
    for (auto& q : p)
    {
        Viewport vp;
        if (!getViewport(q.first, q.second, &vp))
            continue;
        viewports.push_back(vp);
        if (!first_camera)
            first_camera = q.second;
    }
    if (viewports.empty() || !first_camera)
        return false;

    // The layout of the environment descriptor set is created by the skybox
    // renderer, which doesn't exist yet when the FBO creates this pass (during
    // the creation of the swapchain)
    if (!m_pipelines_created)
    {
        try
        {
            createPipelines();
        }
        catch (const std::exception& e)
        {
            GE::logMessage(GE::GLL_ERROR,
                "Failed to create the pipelines of the surfel GI: %s",
                e.what());
        }
        m_pipelines_created = true;
    }

    // The depth and the normal of the g-buffer are written, and what the last
    // frame read or wrote of the image and the buffers has to be finished
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

    GEVulkanRTScene* rt_scene = m_vk->getRTScene();
    // Nothing to trace in the menus, before the scene is built or if the
    // pipelines can't be created, the lighting pass samples the image, so it
    // has to be cleared once. Generate is the last pipeline created.
    if (m_pipelines[PI_GENERATE] == VK_NULL_HANDLE || !rt_scene ||
        !rt_scene->isReady())
    {
        if (m_needs_image_clear)
        {
            VkMemoryBarrier to_transfer = {};
            to_transfer.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            to_transfer.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT;
            to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &to_transfer, 0, NULL, 0,
                NULL);
            VkClearColorValue color = {};
            VkImageSubresourceRange range = {};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = 1;
            range.layerCount = 1;
            vkCmdClearColorImage(cmd, m_gi->getImage(),
                VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
            VkMemoryBarrier to_shader = {};
            to_shader.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            to_shader.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            to_shader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &to_shader, 0, NULL,
                0, NULL);
            m_needs_image_clear = false;
        }
        return false;
    }

    // The surfels are attached to the triangles of the ray traced scene: when
    // it's built again (another track, another scene) the triangles of the
    // pool are gone, so the pool is reset (the prepare pass zeroes it and
    // fills the stack of the free surfels again)
    if (m_rt_build_counter != rt_scene->getBuildCounter())
    {
        m_rt_build_counter = rt_scene->getBuildCounter();
        m_needs_clear = true;
        m_needs_image_clear = true;
    }

    // The pool is initialized by the first frame which runs the passes: the
    // surfels are zeroed (none of them is alive) and surfel_gi_prepare.comp
    // fills the stack of the free surfels and the counters
    if (m_needs_clear)
    {
        vkCmdFillBuffer(cmd, m_buffers[BI_SURFEL].m_buffer, 0, VK_WHOLE_SIZE,
            0);
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }

    m_frame++;
    if (m_frame == 0)
        m_frame = 1;

    SurfelGIPushConstants pc = {};
    const irr::core::vector3df origin = first_camera->getAbsolutePosition();
    pc.m_origin[0] = origin.X;
    pc.m_origin[1] = origin.Y;
    pc.m_origin[2] = origin.Z;
    pc.m_frame = m_frame;
    pc.m_first_frame = m_needs_clear ? 1 : 0;
    pc.m_max_rays = MAX_RAYS;
    pc.m_delta = CELL_DELTA;
    pc.m_half_d = CELL_HALF_DELTA;
    // The vertical fov of the projection of the camera (the aspect of the
    // projection is not used)
    const irr::core::matrix4& projection =
        first_camera->getViewFrustum()->getTransform(
        irr::video::ETS_PROJECTION);
    float fov = 2.0f * std::atan(1.0f / std::max(projection[5], 1.0e-4f));
    if (!(fov > 0.0f && fov < 3.14159265f))
        fov = 1.0f;
    pc.m_fov = fov;
    pc.m_surfel_size = SURFEL_SIZE;
    pc.m_full_width = (uint32_t)(viewports[0].m_rect[2] - viewports[0].m_rect[0]);
    pc.m_full_height =
        (uint32_t)(viewports[0].m_rect[3] - viewports[0].m_rect[1]);

    const unsigned buffer_idx = (unsigned)m_vk->getCurrentBufferIdx();
    const VkDescriptorSet rt_set = *rt_scene->getDescriptorSet(buffer_idx);
    const VkDescriptorSet* mesh_set =
        m_vk->getMeshTextureDescriptor()->getDescriptorSet();
    const VkDescriptorSet* env_set =
        viewports[0].m_draw_call->getEnvDescriptorSet(m_vk);
    auto push_constants = [&]()
    {
        vkCmdPushConstants(cmd, m_pipeline_layout,
            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    };

    // 1. The cells and the counters of the frame
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_PREPARE]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    push_constants();
    vkCmdDispatch(cmd, groups(std::max(CELL_COUNT, MAX_SURFELS), 64), 1, 1);
    computeToComputeBarrier(cmd, 0);

    // 2. The surfels follow their triangles, update their radius and ask for
    // the rays (the camera and the global light of the first viewport)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_UPDATE]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &rt_set, 0, NULL);
    viewports[0].m_draw_call->bindDataDescriptorSet(m_vk, cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 3, 1, mesh_set, 0, NULL);
    push_constants();
    vkCmdDispatch(cmd, groups(MAX_SURFELS, 64), 1, 1);
    computeToComputeBarrier(cmd, 0);

    // 3. The offsets of the lists of the cells, and then the lists (a surfel
    // is in the 27 cells around it)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_CELL_INFO]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    push_constants();
    vkCmdDispatch(cmd, groups(CELL_COUNT, 64), 1, 1);
    computeToComputeBarrier(cmd, 0);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_CELL_TO_SURFEL]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    push_constants();
    vkCmdDispatch(cmd, groups(MAX_SURFELS, 64), 1, 1);
    computeToComputeBarrier(cmd, 0);

    // 4. Trace only the rays of the frame. Cell info wrote both indirect
    // commands; this barrier covers tracing and cooperative integration.
    {
        VkMemoryBarrier to_indirect = {};
        to_indirect.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        to_indirect.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        to_indirect.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &to_indirect, 0, NULL, 0,
            NULL);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_TRACE]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &rt_set, 0, NULL);
    viewports[0].m_draw_call->bindDataDescriptorSet(m_vk, cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 3, 1, mesh_set, 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 4, 1, env_set, 0, NULL);
    push_constants();
    vkCmdDispatchIndirect(cmd, m_buffers[BI_DISPATCH].m_buffer, 0);
    computeToComputeBarrier(cmd, 0);

    // 5. Integrate samples and depth moments, one 32-thread group per active
    // surfel (linear group ID is x + y * 256)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_INTEGRATE]);
    // The global light is read for the skytop color of the clamp of the
    // samples (surfelFireflyClamp)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &rt_set, 0, NULL);
    viewports[0].m_draw_call->bindDataDescriptorSet(m_vk, cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    push_constants();
    vkCmdDispatchIndirect(cmd, m_buffers[BI_DISPATCH].m_buffer,
        3 * sizeof(uint32_t));
    computeToComputeBarrier(cmd, 0);

    // 6. Share from immutable SurfelLight into the MSME mean of the surfels
    // which have rays (one thread per entry of u_active)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_SHARE]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &rt_set, 0, NULL);
    viewports[0].m_draw_call->bindDataDescriptorSet(m_vk, cmd,
        VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    push_constants();
    vkCmdDispatchIndirect(cmd, m_buffers[BI_DISPATCH].m_buffer,
        6 * sizeof(uint32_t));
    computeToComputeBarrier(cmd, 0);

    // 7. The irradiation of the pixels of every viewport, and the spawn and
    // the removal of the surfels
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipelines[PI_GENERATE]);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, 1, &rt_set, 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 2, 1, &m_set, 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 3, 1, mesh_set, 0, NULL);
    // The environment seeds a surfel which no surfel covers (surfelSpawn)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 4, 1, env_set, 0, NULL);
    for (const Viewport& vp : viewports)
    {
        memcpy(pc.m_rect, vp.m_rect.data(), sizeof(pc.m_rect));
        vp.m_draw_call->bindDataDescriptorSet(m_vk, cmd,
            VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline_layout, 1);
        push_constants();
        const std::array<int32_t, 4> half_rect = halfRect(vp.m_rect);
        vkCmdDispatch(cmd,
            groups((uint32_t)std::max(half_rect[2] - half_rect[0], 0), 16),
            groups((uint32_t)std::max(half_rect[3] - half_rect[1], 0), 16), 1);
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
    // The depth and the normal are read by the compute shaders (execution
    // dependency only), so the g-buffer of the next frame has to wait for them
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 0,
        NULL);
    m_needs_clear = false;
    m_needs_image_clear = false;
    return true;
}   // render

}
