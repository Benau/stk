#include "ge_vulkan_anti_aliasing.hpp"

#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
// The size of the work groups of the kernels (see the shaders): the edges
// kernel outputs 14x14 quads (28x28 pixels) per group
const int32_t EDGES_GROUP_PIXELS = 28;
// ----------------------------------------------------------------------------
const char* g_shaders[] =
{
    "cmaa2_copy.comp",
    "cmaa2_edges.comp",
    "cmaa2_args.comp",
    "cmaa2_candidates.comp",
    "cmaa2_apply.comp"
};
}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanAntiAliasing::GEVulkanAntiAliasing(GEVulkanDriver* vk,
    VkDescriptorSetLayout io_layout,
    const irr::core::dimension2d<irr::u32>& capacity)
                    : GEVulkanPostProcessing(vk, capacity),
                      m_layout(VK_NULL_HANDLE), m_pool(VK_NULL_HANDLE),
                      m_set(VK_NULL_HANDLE), m_pipeline_layout(VK_NULL_HANDLE),
                      m_initialized(false)
{
    m_buffers.fill(VK_NULL_HANDLE);
    m_allocations.fill(VK_NULL_HANDLE);
    m_pipelines.fill(VK_NULL_HANDLE);
    memset(&m_constants, 0, sizeof(m_constants));
    try
    {
        createBuffers(capacity);
        createDescriptors();
        createPipelines(io_layout);
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanAntiAliasing

// ----------------------------------------------------------------------------
GEVulkanAntiAliasing::~GEVulkanAntiAliasing()
{
    destroy();
}   // ~GEVulkanAntiAliasing

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::destroy()
{
    VkDevice device = m_vk->getDevice();
    for (VkPipeline pipeline : m_pipelines)
    {
        if (pipeline != VK_NULL_HANDLE)
            vkDestroyPipeline(device, pipeline, NULL);
    }
    if (m_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    // The set is freed with its pool
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_pool, NULL);
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_layout, NULL);
    for (unsigned i = 0; i < B_COUNT; i++)
    {
        if (m_buffers[i] != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_vk->getVmaAllocator(), m_buffers[i],
                m_allocations[i]);
        }
    }
}   // destroy

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::createBuffers(
    const irr::core::dimension2d<irr::u32>& capacity)
{
    const uint64_t w = std::max(1u, capacity.Width);
    const uint64_t h = std::max(1u, capacity.Height);
    const uint64_t pixels = w * h;
    // The sizes of the reference implementation: it starts ignoring edges in
    // extreme cases (full screen of dense noise) if they are not enough, the
    // shaders check them
    const uint32_t edges_stride = (uint32_t)((w + 1) / 2);
    const uint32_t heads_stride = (uint32_t)((w + 1) / 2);
    const uint64_t heads_rows = (h + 1) / 2;
    const uint32_t max_candidates = (uint32_t)std::max<uint64_t>(pixels / 4, 4);
    const uint32_t max_locations =
        (uint32_t)std::max<uint64_t>((pixels + 3) / 6, 4);
    const uint32_t max_items = (uint32_t)std::max<uint64_t>(pixels / 2, 4);

    m_constants.m_buf[0] = edges_stride;
    m_constants.m_buf[1] = heads_stride;
    m_constants.m_buf[2] = max_candidates;
    m_constants.m_buf[3] = max_locations;
    m_constants.m_max_items = max_items;

    std::array<VkDeviceSize, B_COUNT> sizes;
    sizes[B_EDGES] = (VkDeviceSize)edges_stride * h * 4;
    sizes[B_HEADS] = (VkDeviceSize)heads_stride * heads_rows * 4;
    sizes[B_CANDIDATES] = (VkDeviceSize)max_candidates * 4;
    sizes[B_LOCATIONS] = (VkDeviceSize)max_locations * 4;
    sizes[B_ITEMS] = (VkDeviceSize)max_items * 8;
    sizes[B_CONTROL] = 16 * 4;
    sizes[B_INDIRECT] = 4 * 4;

    for (unsigned i = 0; i < B_COUNT; i++)
    {
        VmaAllocationCreateInfo info = {};
        info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (i == B_INDIRECT)
            usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        if (!m_vk->createBuffer(sizes[i], usage, info, m_buffers[i],
            m_allocations[i]))
        {
            m_buffers[i] = VK_NULL_HANDLE;
            throw std::runtime_error("createBuffer failed for "
                "GEVulkanAntiAliasing");
        }
    }
}   // createBuffers

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::createDescriptors()
{
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayoutBinding, B_COUNT> bindings = {};
    for (unsigned i = 0; i < B_COUNT; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
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
            "GEVulkanAntiAliasing");
    }

    VkDescriptorPoolSize pool_size = {};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = B_COUNT;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    pool_info.maxSets = 1;
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed for "
            "GEVulkanAntiAliasing");
    }

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &m_layout;
    if (vkAllocateDescriptorSets(device, &alloc_info, &m_set) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed for "
            "GEVulkanAntiAliasing");
    }

    std::array<VkDescriptorBufferInfo, B_COUNT> infos = {};
    std::array<VkWriteDescriptorSet, B_COUNT> writes = {};
    for (unsigned i = 0; i < B_COUNT; i++)
    {
        infos[i].buffer = m_buffers[i];
        infos[i].offset = 0;
        infos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(device, writes.size(), writes.data(), 0, NULL);
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::createPipelines(VkDescriptorSetLayout io_layout)
{
    VkDevice device = m_vk->getDevice();
    std::array<VkDescriptorSetLayout, 2> layouts = {{ io_layout, m_layout }};
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(PushConstants);
    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = layouts.size();
    layout_info.pSetLayouts = layouts.data();
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(device, &layout_info, NULL,
        &m_pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for "
            "GEVulkanAntiAliasing");
    }

    for (unsigned i = 0; i < K_COUNT; i++)
    {
        VkComputePipelineCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        info.stage.module = GEVulkanShaderManager::getShader(g_shaders[i]);
        info.stage.pName = "main";
        info.layout = m_pipeline_layout;
        if (vkCreateComputePipelines(device, m_vk->getPipelineCache(), 1,
            &info, NULL, &m_pipelines[i]) != VK_SUCCESS)
        {
            m_pipelines[i] = VK_NULL_HANDLE;
            throw std::runtime_error(std::string("vkCreateComputePipelines "
                "failed for ") + g_shaders[i]);
        }
    }
}   // createPipelines

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::initBuffers(VkCommandBuffer cmd)
{
    // No edges, no lists (the heads of the lists are only set for the quads
    // which have edges, but they are never read for any other) and the counters
    // are zero. The counters are zero again after every use
    vkCmdFillBuffer(cmd, m_buffers[B_EDGES], 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, m_buffers[B_HEADS], 0, VK_WHOLE_SIZE, 0xFFFFFFFFu);
    vkCmdFillBuffer(cmd, m_buffers[B_CONTROL], 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, m_buffers[B_INDIRECT], 0, VK_WHOLE_SIZE, 0);
    VkMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &b, 0, NULL, 0, NULL);
    m_initialized = true;
}   // initBuffers

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::barrier(VkCommandBuffer cmd)
{
    // Every kernel reads what the last one wrote (buffers, the output image,
    // and the indirect arguments)
    VkMemoryBarrier b = {};
    b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &b, 0, NULL, 0, NULL);
}   // barrier

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::push(VkCommandBuffer cmd,
                                const std::array<int32_t, 4>& rect)
{
    PushConstants pc = m_constants;
    for (unsigned i = 0; i < 4; i++)
        pc.m_rect[i] = rect[i];
    vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(pc), &pc);
}   // push

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::bind(VkCommandBuffer cmd, Kernel kernel)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelines[kernel]);
}   // bind

// ----------------------------------------------------------------------------
void GEVulkanAntiAliasing::apply(VkCommandBuffer cmd, VkDescriptorSet io_set,
    const irr::core::dimension2d<irr::u32>& size,
    const std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCamera*> >& p)
{
    // The size of the FBO, which getViewport clamps the viewports to
    m_size = size;
    if (!m_initialized)
        initBuffers(cmd);

    const std::array<VkDescriptorSet, 2> sets = {{ io_set, m_set }};
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeline_layout, 0, sets.size(), sets.data(), 0, NULL);

    // 0. Output = input for the whole image
    const std::array<int32_t, 4> full = {{ 0, 0, (int32_t)size.Width,
        (int32_t)size.Height }};
    push(cmd, full);
    bind(cmd, K_COPY);
    vkCmdDispatch(cmd, (size.Width + 7) / 8, (size.Height + 7) / 8, 1);
    barrier(cmd);

    // The viewports don't overlap, but they share the working buffers, so
    // one after the other
    for (auto& q : p)
    {
        Viewport vp;
        if (!getViewport(q.first, q.second, &vp))
            continue;
        push(cmd, vp.m_rect);

        // 1. Edges. The quads start at the even pixel before the viewport
        const int32_t x0 = vp.m_rect[0] & ~1;
        const int32_t y0 = vp.m_rect[1] & ~1;
        bind(cmd, K_EDGES);
        vkCmdDispatch(cmd,
            (vp.m_rect[2] - x0 + EDGES_GROUP_PIXELS - 1) / EDGES_GROUP_PIXELS,
            (vp.m_rect[3] - y0 + EDGES_GROUP_PIXELS - 1) / EDGES_GROUP_PIXELS,
            1);
        barrier(cmd);

        // 2. The shapes
        bind(cmd, K_ARGS);
        vkCmdDispatch(cmd, 2, 1, 1);
        barrier(cmd);
        bind(cmd, K_CANDIDATES);
        vkCmdDispatchIndirect(cmd, m_buffers[B_INDIRECT], 0);
        barrier(cmd);

        // 3. The colors of the pixels (and the counters are zero again)
        bind(cmd, K_ARGS);
        vkCmdDispatch(cmd, 1, 2, 1);
        barrier(cmd);
        bind(cmd, K_APPLY);
        vkCmdDispatchIndirect(cmd, m_buffers[B_INDIRECT], 0);
        barrier(cmd);
    }
}   // apply

}
