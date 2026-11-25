#include "ge_vulkan_static_draw_batches.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_dynamic_buffer.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_light_handler.hpp"
#include "ge_vulkan_static_draw_source.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include <algorithm>
#include <cstring>

namespace GE
{
namespace
{
const unsigned COMMAND_SIZE = sizeof(VkDrawIndexedIndirectCommand);

// ----------------------------------------------------------------------------
void writeBuffer(VkDescriptorSet set, unsigned binding, VkDescriptorType type,
                 VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range)
{
    VkDescriptorBufferInfo info = { buffer, offset, range };
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &info;
    vkUpdateDescriptorSets(getVKDriver()->getDevice(), 1, &write, 0, NULL);
}   // writeBuffer

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanStaticDrawBatches::GEVulkanStaticDrawBatches(
    const std::shared_ptr<GEVulkanStaticDrawSource>& source,
    VkDescriptorSetLayout draw_layout)
                         : GEVulkanStaticDrawBatches(source, draw_layout, true)
{
}   // GEVulkanStaticDrawBatches

// ----------------------------------------------------------------------------
GEVulkanStaticDrawBatches::GEVulkanStaticDrawBatches(
    const std::shared_ptr<GEVulkanStaticDrawSource>& source,
    VkDescriptorSetLayout draw_layout, bool with_stats)
                         : m_source(source), m_buffer(VK_NULL_HANDLE),
                           m_memory(NULL), m_objects_offset(0),
                           m_stats_buffer(VK_NULL_HANDLE),
                           m_stats_memory(NULL), m_stats_mapped(NULL),
                           m_stats_stride(0), m_group_count(0),
                           m_poly_count(0), m_pool(VK_NULL_HANDLE)
{
    if (!m_source->isUsable())
        return;
    if (!createBuffer() || (with_stats && !createStats()) ||
        !createDescriptors(draw_layout))
    {
        logMessage(GLL_WARNING,
            "Static GPU draws of a view are unavailable, using the CPU");
        m_compute_sets.clear();
    }
}   // GEVulkanStaticDrawBatches

// ----------------------------------------------------------------------------
GEVulkanStaticDrawBatches::~GEVulkanStaticDrawBatches()
{
    GEVulkanDriver* vk = getVKDriver();
    if (m_buffer != VK_NULL_HANDLE || m_stats_buffer != VK_NULL_HANDLE)
        vk->waitIdle();
    vkDestroyDescriptorPool(vk->getDevice(), m_pool, NULL);
    vmaDestroyBuffer(vk->getVmaAllocator(), m_stats_buffer, m_stats_memory);
    vmaDestroyBuffer(vk->getVmaAllocator(), m_buffer, m_memory);
}   // ~GEVulkanStaticDrawBatches

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawBatches::isUsable() const
{
    const Variant variant = getVariant();
    return m_source->isUsable() && !m_compute_sets.empty() &&
        (variant == GEVulkanStaticDrawSource::SV_VIEW ||
        m_source->isShadowUsable(variant));
}   // isUsable

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawBatches::findRanges(const std::string& shader,
                                           const Range*& ranges,
                                           unsigned& count) const
{
    if (getVariant() == GEVulkanStaticDrawSource::SV_VIEW)
    {
        auto it = m_source->getRanges().find(shader);
        if (it == m_source->getRanges().end())
            return false;
        ranges = &it->second;
        count = 1;
        return true;
    }
    auto it = m_source->getShadowRanges().find(shader);
    if (it == m_source->getShadowRanges().end())
        return false;
    ranges = it->second.data();
    count = it->second.size();
    return true;
}   // findRanges

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawBatches::onCPU(unsigned entry) const
{
    return !isUsable() || m_source->onCPU(entry);
}   // onCPU

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawBatches::createBuffer()
{
    GEVulkanDriver* vk = getVKDriver();
    const VkDeviceSize commands_size = m_source->getCommandCount() *
        COMMAND_SIZE;
    m_objects_offset = commands_size + getPadding(commands_size,
        vk->getPhysicalDeviceProperties().limits
        .minStorageBufferOffsetAlignment);
    VmaAllocationCreateInfo info = {};
    info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    return vk->createBuffer(m_objects_offset + m_source->getSourceCount() *
        sizeof(ObjectData), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        info, m_buffer, m_memory);
}   // createBuffer

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawBatches::createStats()
{
    GEVulkanDriver* vk = getVKDriver();
    const unsigned group_size = GEVulkanStaticDrawSource::getGroupSize();
    m_group_count = (m_source->getSourceCount() + group_size - 1) / group_size;
    const VkDeviceSize size = m_group_count * sizeof(uint32_t);
    m_stats_stride = size + getPadding(size, vk->getPhysicalDeviceProperties()
        .limits.minStorageBufferOffsetAlignment);
    const VkDeviceSize total = m_stats_stride * (vk->getMaxFrameInFlight() + 1);

    // Host cached memory if there is any (or the coherent one), the shader
    // only does plain stores and the CPU reads it back with random access
    VmaAllocationCreateInfo info = {};
    info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (!vk->createBuffer(total, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, info,
        m_stats_buffer, m_stats_memory))
        return false;
    VmaAllocationInfo alloc_info;
    vmaGetAllocationInfo(vk->getVmaAllocator(), m_stats_memory, &alloc_info);
    if (alloc_info.pMappedData == NULL)
        return false;
    // The first frames read the slots before the GPU writes them
    memset(alloc_info.pMappedData, 0, total);
    vmaFlushAllocation(vk->getVmaAllocator(), m_stats_memory, 0,
        VK_WHOLE_SIZE);
    m_stats_mapped = (const uint32_t*)alloc_info.pMappedData;
    return true;
}   // createStats

// ----------------------------------------------------------------------------
void GEVulkanStaticDrawBatches::readPolyCount(unsigned idx)
{
    // The GPU wrote this slot when it was last used (the frame in flight count
    // plus one frames ago), and the fence of that frame has been waited for
    // before the command buffers are built. The values are not cleared, they
    // are all overwritten by the dispatch which is recorded after this
    const VkDeviceSize offset = idx * m_stats_stride;
    vmaInvalidateAllocation(getVKDriver()->getVmaAllocator(), m_stats_memory,
        offset, m_stats_stride);
    const uint32_t* groups = m_stats_mapped + offset / sizeof(uint32_t);
    unsigned total = 0;
    for (unsigned i = 0; i < m_group_count; i++)
        total += groups[i];
    m_poly_count = total;
}   // readPolyCount

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawBatches::createDescriptors(
                                              VkDescriptorSetLayout draw_layout)
{
    GEVulkanDriver* vk = getVKDriver();
    const unsigned count = vk->getMaxFrameInFlight() + 1;
    // Compute sets: camera, 4 buffers. Draw sets: camera, light, 3 objects
    VkDescriptorPoolSize sizes[] =
    {
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 3 * count },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * count },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 3 * count }
    };
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 2 * count;
    pool_info.poolSizeCount = 3;
    pool_info.pPoolSizes = sizes;
    if (vkCreateDescriptorPool(vk->getDevice(), &pool_info, NULL, &m_pool) !=
        VK_SUCCESS)
        return false;

    std::vector<VkDescriptorSetLayout> layouts(count,
        m_source->getComputeSetLayout());
    layouts.resize(2 * count, draw_layout);
    std::vector<VkDescriptorSet> sets(layouts.size());
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_pool;
    alloc_info.descriptorSetCount = layouts.size();
    alloc_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(vk->getDevice(), &alloc_info, sets.data()) !=
        VK_SUCCESS)
        return false;
    m_compute_sets.assign(sets.begin(), sets.begin() + count);
    m_draw_sets.assign(sets.begin() + count, sets.end());
    m_camera_observers.resize(count);
    return true;
}   // createDescriptors

// ----------------------------------------------------------------------------
void GEVulkanStaticDrawBatches::updateDescriptors(GEVulkanCamera* cam,
                                                  bool lighting)
{
    const unsigned idx = getVKDriver()->getCurrentBufferIdx();
    // The camera uniform buffer is recreated when it grows
    if (!m_camera_observers[idx].expired())
        return;
    m_camera_observers[idx] = cam->getCameraUBO()->getBufferObserver();

    VkBuffer camera = GEVulkanDynamicBuffer::supportsHostTransfer() ?
        cam->getCameraUBO()->getHostBuffer()[idx] :
        cam->getCameraUBO()->getLocalBuffer()[idx];
    const VkDescriptorType ubo = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    const VkDescriptorType sbo = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    const VkDescriptorType sbo_dynamic =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    const VkDeviceSize commands_size = m_source->getCommandCount() *
        COMMAND_SIZE;
    const VkDeviceSize objects_size = m_source->getSourceCount() *
        sizeof(ObjectData);

    VkDescriptorSet set = m_compute_sets[idx];
    writeBuffer(set, 0, ubo, camera, 0, GEVulkanCamera::getCameraUBOSize());
    writeBuffer(set, 1, sbo, m_source->getBuffer(), 0,
        m_source->getSourceCount() * m_source->getSourceSize());
    writeBuffer(set, 2, sbo, m_buffer, 0, commands_size);
    writeBuffer(set, 3, sbo, m_buffer, m_objects_offset, objects_size);
    // The shadow views have no statistics, their shader doesn't use it
    if (m_stats_buffer != VK_NULL_HANDLE)
    {
        writeBuffer(set, 4, sbo, m_stats_buffer, idx * m_stats_stride,
            m_group_count * sizeof(uint32_t));
    }

    // The same bindings as GEVulkanDrawCall::updateDataDescriptorSets. The
    // draws don't use the skinning and the materials of the draw ids, but
    // their descriptors must be valid
    set = m_draw_sets[idx];
    writeBuffer(set, 0, ubo, camera, 0, GEVulkanCamera::getCameraUBOSize());
    writeBuffer(set, 1, sbo_dynamic, m_buffer, m_objects_offset, objects_size);
    writeBuffer(set, 2, sbo_dynamic, m_buffer, m_objects_offset, objects_size);
    writeBuffer(set, 3, ubo, camera, 0, lighting ?
        GEVulkanLightHandler::getMaxSize() :
        GEVulkanCamera::getCameraUBOSize());
    if (GEVulkanFeatures::supportsBindMeshTexturesAtOnce())
    {
        writeBuffer(set, 4, sbo_dynamic, m_buffer, m_objects_offset,
            objects_size);
    }
}   // updateDescriptors

// ----------------------------------------------------------------------------
void GEVulkanStaticDrawBatches::dispatch(VkCommandBuffer cmd,
                                         GEVulkanDrawCall& dc,
                                         GEVulkanCamera* cam)
{
    updateDescriptors(cam, dc.m_light_handler != NULL);
    const unsigned idx = getVKDriver()->getCurrentBufferIdx();
    if (m_stats_buffer != VK_NULL_HANDLE)
        readPolyCount(idx);
    const VkPipelineStageFlags readers = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
        VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT;

    // The previous frame may still draw with the destination of this view
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, readers | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);

    // Reset the instance counts of the commands
    VkBufferCopy copy = {};
    copy.srcOffset = m_source->getCommandsOffset();
    copy.size = m_source->getCommandCount() * COMMAND_SIZE;
    vkCmdCopyBuffer(cmd, m_source->getBuffer(), m_buffer, 1, &copy);
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
        NULL);

    // The levels of detail use the camera of the scene (not the one of the
    // shadow) as the CPU does
    const irr::core::vector3df& position = cam->getAbsolutePosition();
    const float lod_camera[4] =
        { position.X, position.Y, position.Z, getSkipSmall() };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_source->getPipeline(getVariant()));
    vkCmdPushConstants(cmd, m_source->getPipelineLayout(),
        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(lod_camera), lod_camera);
    const uint32_t camera_offset = dc.m_camera_ubo_offset;
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        m_source->getPipelineLayout(), 0, 1, &m_compute_sets[idx], 1,
        &camera_offset);
    const unsigned group_size = GEVulkanStaticDrawSource::getGroupSize();
    vkCmdDispatch(cmd, (m_source->getSourceCount() + group_size - 1) /
        group_size, 1, 1);

    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT |
        VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, readers, 0,
        1, &barrier, 0, NULL, 0, NULL);
}   // dispatch

// ----------------------------------------------------------------------------
void GEVulkanStaticDrawBatches::render(VkCommandBuffer cmd,
                                       GEVulkanDrawCall& dc,
                                       const std::string& shader,
                                       GEVulkanPipelineType pt,
                                       VkPipelineLayout layout,
                                       std::vector<uint32_t> offsets)
{
    const Range* ranges = NULL;
    unsigned range_count = 0;
    if (!findRanges(shader, ranges, range_count))
        return;
    GEVulkanDriver* vk = getVKDriver();
    // The descriptors of the skinning and the materials are not used, and
    // the objects start at the beginning of their descriptor
    for (unsigned i : { 1u, 2u, 4u })
    {
        if (i < offsets.size())
            offsets[i] = 0;
    }
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, 1,
        &m_draw_sets[vk->getCurrentBufferIdx()], offsets.size(),
        offsets.data());

    const bool bind_textures = !dc.m_texture_descriptor->isSingleDescriptor();
    for (unsigned r = 0; r < range_count; r++)
    {
        const unsigned first = ranges[r].m_first;
        const unsigned count = ranges[r].m_count;
        if (!bind_textures && GEVulkanFeatures::supportsMultiDrawIndirect())
        {
            // The count of the commands is the count of the draws (the
            // instance count of the culled ones is 0)
            const unsigned max_count = vk->getPhysicalDeviceProperties()
                .limits.maxDrawIndirectCount;
            for (unsigned done = 0; done < count;)
            {
                const unsigned draw_count = std::min(count - done, max_count);
                vkCmdDrawIndexedIndirect(cmd, m_buffer,
                    (first + done) * COMMAND_SIZE, draw_count, COMMAND_SIZE);
                done += draw_count;
            }
            continue;
        }
        // One draw for each command, the textures are bound when they change
        int current_material = -1;
        for (unsigned i = first; i < first + count; i++)
        {
            if (bind_textures && m_source->getMaterial(i) != current_material)
            {
                current_material = m_source->getMaterial(i);
                dc.bindSingleMaterial(cmd, shader, current_material, pt);
            }
            vkCmdDrawIndexedIndirect(cmd, m_buffer, i * COMMAND_SIZE, 1,
                COMMAND_SIZE);
        }
    }
}   // render

}
