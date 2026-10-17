#include "ge_vulkan_rt_scene.hpp"

#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_render_info.hpp"
#include "ge_spm_buffer.hpp"
#include "ge_vulkan_animated_mesh_scene_node.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_dynamic_buffer.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_texture_descriptor.hpp"
#include "mini_glm.hpp"

#include "IMeshSceneNode.h"
#include "ISceneNode.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <string>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
struct RTSource
{
    GESPMBuffer* m_buffer;
    irr::scene::ISceneNode* m_node;
    unsigned m_material_index;
    uint32_t m_shading_type;
};   // RTSource

// ----------------------------------------------------------------------------
// Only the materials which can be shaded by the ray traced shaders (with the
// default vertex shader), alphatest, transparent, etc. are not included, so
// the rays pass through them
bool getShadingType(const GEMaterial* material, uint32_t* type)
{
    if (!material || material->isTransparent() ||
        material->m_vertex_shader != "spm.vert")
        return false;
    if (material->m_fragment_shader == "solid.frag" ||
        material->m_fragment_shader == "normalmap.frag")
    {
        *type = GRTST_SOLID;
        return true;
    }
    return false;
}   // getShadingType

// ----------------------------------------------------------------------------
void collectNode(irr::scene::ISceneNode* node, std::vector<RTSource>& sources)
{
    if (node->isVisible())
    {
        irr::scene::IMesh* mesh = NULL;
        if (node->getType() == irr::scene::ESNT_ANIMATED_MESH)
            mesh = static_cast<GEVulkanAnimatedMeshSceneNode*>(node)->getMesh();
        else if (node->getType() == irr::scene::ESNT_MESH)
            mesh = static_cast<irr::scene::IMeshSceneNode*>(node)->getMesh();
        if (mesh && node->isTrulyVisible())
        {
            for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
            {
                irr::scene::IMeshBuffer* b = mesh->getMeshBuffer(i);
                if (b->getVertexType() != irr::video::EVT_SKINNED_MESH ||
                    i >= node->getMaterialCount())
                    continue;
                GESPMBuffer* buffer = static_cast<GESPMBuffer*>(b);
                if (buffer->hasSkinning() || buffer->isDynamic() ||
                    buffer->getVertexCount() == 0 ||
                    buffer->getIndexCount() < 3)
                    continue;
                irr::video::SMaterial& m = node->getMaterial(i);
                const std::shared_ptr<GERenderInfo>& ri = m.getRenderInfo();
                if (!ri || ri->getRayTracingType() != GRTT_STATIC)
                    continue;
                uint32_t type = 0;
                if (!getShadingType(GEMaterialManager::getMaterial(
                    node->getMaterialType(i)), &type))
                    continue;
                sources.push_back({ buffer, node, i, type });
            }
        }
    }
    const irr::core::array<irr::scene::ISceneNode*>& children =
        node->getChildren();
    for (irr::u32 i = 0; i < children.size(); i++)
        collectNode(children[i], sources);
}   // collectNode

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanRTScene::GEVulkanRTScene(GEVulkanDriver* vk)
{
    m_vk = vk;
    m_descriptor_layout = VK_NULL_HANDLE;
    m_descriptor_pool = VK_NULL_HANDLE;
    m_data_buffer = VK_NULL_HANDLE;
    m_data_memory = VK_NULL_HANDLE;
    m_blas_buffer = VK_NULL_HANDLE;
    m_blas_memory = VK_NULL_HANDLE;
    m_blas = VK_NULL_HANDLE;
    m_tlas_buffer = VK_NULL_HANDLE;
    m_tlas_memory = VK_NULL_HANDLE;
    m_tlas = VK_NULL_HANDLE;
    m_geometry_count = 0;
    m_triangle_count = 0;

    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    for (auto& b : bindings)
    {
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = bindings.size();
    layout_info.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(m_vk->getDevice(), &layout_info, NULL,
        &m_descriptor_layout) != VK_SUCCESS)
    {
        throw std::runtime_error(
            "vkCreateDescriptorSetLayout failed for GEVulkanRTScene");
    }
    std::array<VkDescriptorPoolSize, 2> sizes = {};
    sizes[0].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    const unsigned set_count = GEVulkanDriver::getMaxFrameInFlight() + 1;
    sizes[0].descriptorCount = set_count;
    sizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[1].descriptorCount = set_count * 2;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = sizes.size();
    pool_info.pPoolSizes = sizes.data();
    pool_info.maxSets = set_count;
    if (vkCreateDescriptorPool(m_vk->getDevice(), &pool_info, NULL,
        &m_descriptor_pool) != VK_SUCCESS)
    {
        throw std::runtime_error(
            "vkCreateDescriptorPool failed for GEVulkanRTScene");
    }
    std::vector<VkDescriptorSetLayout> layouts(set_count, m_descriptor_layout);
    m_descriptor_sets.resize(set_count, VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_descriptor_pool;
    alloc_info.descriptorSetCount = set_count;
    alloc_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(m_vk->getDevice(), &alloc_info,
        m_descriptor_sets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error(
            "vkAllocateDescriptorSets failed for GEVulkanRTScene");
    }
}   // GEVulkanRTScene

// ----------------------------------------------------------------------------
GEVulkanRTScene::~GEVulkanRTScene()
{
    clear();
    m_vk->waitIdle();
    vkDestroyDescriptorPool(m_vk->getDevice(), m_descriptor_pool, NULL);
    vkDestroyDescriptorSetLayout(m_vk->getDevice(), m_descriptor_layout, NULL);
}   // ~GEVulkanRTScene

// ----------------------------------------------------------------------------
// Nothing uses the sets while the scene is built (clear waits for the device)
void GEVulkanRTScene::updateDescriptorSet()
{
    VkWriteDescriptorSetAccelerationStructureKHR as_write = {};
    as_write.sType =
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    as_write.accelerationStructureCount = 1;
    as_write.pAccelerationStructures = &m_tlas;
    VkDescriptorBufferInfo data_info = { m_data_buffer, 0, VK_WHOLE_SIZE };
    for (unsigned set = 0; set < m_descriptor_sets.size(); set++)
    {
        // Each set reads the host buffer of the same index
        VkDescriptorBufferInfo geometry_info =
            { m_geometry_data->getHostBuffer()[set], 0, VK_WHOLE_SIZE };
        std::array<VkWriteDescriptorSet, 3> writes = {};
        for (unsigned i = 0; i < writes.size(); i++)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = m_descriptor_sets[set];
            writes[i].dstBinding = i;
            writes[i].descriptorCount = 1;
        }
        writes[0].descriptorType =
            VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        writes[0].pNext = &as_write;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &data_info;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[2].pBufferInfo = &geometry_info;
        vkUpdateDescriptorSets(m_vk->getDevice(), writes.size(),
            writes.data(), 0, NULL);
    }
}   // updateDescriptorSet

// ----------------------------------------------------------------------------
void GEVulkanRTScene::clear()
{
    if (m_tlas == VK_NULL_HANDLE && m_blas == VK_NULL_HANDLE &&
        m_data_buffer == VK_NULL_HANDLE && !m_geometry_data &&
        m_geometry_sources.empty() && m_texture_ids.empty())
        return;
    m_vk->waitIdle();
    VmaAllocator allocator = m_vk->getVmaAllocator();
    if (m_tlas != VK_NULL_HANDLE)
        vkDestroyAccelerationStructureKHR(m_vk->getDevice(), m_tlas, NULL);
    if (m_tlas_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, m_tlas_buffer, m_tlas_memory);
    if (m_blas != VK_NULL_HANDLE)
        vkDestroyAccelerationStructureKHR(m_vk->getDevice(), m_blas, NULL);
    if (m_blas_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, m_blas_buffer, m_blas_memory);
    if (m_data_buffer != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, m_data_buffer, m_data_memory);
    // Destroys the buffers (it waits for the device too)
    m_geometry_data.reset();
    for (GeometrySource& source : m_geometry_sources)
        source.m_node->drop();
    m_geometry_sources.clear();
    m_geometry_infos.clear();
    m_tlas = m_blas = VK_NULL_HANDLE;
    m_tlas_buffer = m_blas_buffer = m_data_buffer = VK_NULL_HANDLE;
    m_tlas_memory = m_blas_memory = m_data_memory = VK_NULL_HANDLE;
    m_texture_ids.clear();
    m_geometry_count = 0;
    m_triangle_count = 0;
}   // clear

// ----------------------------------------------------------------------------
VkDeviceAddress GEVulkanRTScene::getAddress(VkBuffer buffer) const
{
    VkBufferDeviceAddressInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(m_vk->getDevice(), &info);
}   // getAddress

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::createDeviceBuffer(const void* data, VkDeviceSize size,
                                         VkBufferUsageFlags usage,
                                         VkBuffer& buffer,
                                         VmaAllocation& memory)
{
    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_memory = VK_NULL_HANDLE;
    VmaAllocationCreateInfo staging_info = {};
    staging_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    staging_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    staging_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!m_vk->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        staging_info, staging_buffer, staging_memory))
        return false;

    void* mapped = NULL;
    if (vmaMapMemory(m_vk->getVmaAllocator(), staging_memory, &mapped) !=
        VK_SUCCESS)
    {
        vmaDestroyBuffer(m_vk->getVmaAllocator(), staging_buffer,
            staging_memory);
        return false;
    }
    memcpy(mapped, data, (size_t)size);
    vmaUnmapMemory(m_vk->getVmaAllocator(), staging_memory);
    vmaFlushAllocation(m_vk->getVmaAllocator(), staging_memory, 0, size);

    VmaAllocationCreateInfo local_info = {};
    local_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    local_info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    bool ok = m_vk->createBuffer(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        local_info, buffer, memory);
    if (ok)
        m_vk->copyBuffer(staging_buffer, buffer, size);
    vmaDestroyBuffer(m_vk->getVmaAllocator(), staging_buffer, staging_memory);
    return ok;
}   // createDeviceBuffer

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::createAccelerationStructure(
    VkAccelerationStructureTypeKHR type, VkDeviceSize size, VkBuffer& buffer,
    VmaAllocation& memory, VkAccelerationStructureKHR& as)
{
    VmaAllocationCreateInfo info = {};
    info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    if (!m_vk->createBuffer(size,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, info, buffer, memory))
        return false;
    VkAccelerationStructureCreateInfoKHR create_info = {};
    create_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    create_info.buffer = buffer;
    create_info.size = size;
    create_info.type = type;
    return vkCreateAccelerationStructureKHR(m_vk->getDevice(), &create_info,
        NULL, &as) == VK_SUCCESS;
}   // createAccelerationStructure

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::createScratchBuffer(VkDeviceSize size, VkBuffer& buffer,
                                          VmaAllocation& memory,
                                          VkDeviceAddress& aligned_address)
{
    const VkDeviceSize alignment =
        GEVulkanFeatures::getAccelerationStructureScratchAlignment();
    VmaAllocationCreateInfo info = {};
    info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    info.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    if (!m_vk->createBuffer(size + alignment,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, info, buffer, memory))
        return false;
    VkDeviceAddress address = getAddress(buffer);
    aligned_address = (address + alignment - 1) & ~(alignment - 1);
    return true;
}   // createScratchBuffer

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::buildAccelerationStructures(
    const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
    const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& ranges)
{
    VkDevice device = m_vk->getDevice();
    VmaAllocator allocator = m_vk->getVmaAllocator();

    // BLAS with all geometries
    VkAccelerationStructureBuildGeometryInfoKHR blas_info = {};
    blas_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    blas_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blas_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    blas_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    blas_info.geometryCount = (uint32_t)geometries.size();
    blas_info.pGeometries = geometries.data();
    std::vector<uint32_t> primitive_counts;
    primitive_counts.reserve(ranges.size());
    for (const auto& range : ranges)
        primitive_counts.push_back(range.primitiveCount);
    VkAccelerationStructureBuildSizesInfoKHR blas_sizes = {};
    blas_sizes.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blas_info,
        primitive_counts.data(), &blas_sizes);
    if (!createAccelerationStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        blas_sizes.accelerationStructureSize, m_blas_buffer, m_blas_memory,
        m_blas))
        return false;
    blas_info.dstAccelerationStructure = m_blas;

    VkBuffer blas_scratch = VK_NULL_HANDLE;
    VmaAllocation blas_scratch_memory = VK_NULL_HANDLE;
    if (!createScratchBuffer(blas_sizes.buildScratchSize, blas_scratch,
        blas_scratch_memory, blas_info.scratchData.deviceAddress))
        return false;

    // TLAS with a single identity instance of the BLAS, the triangles are in
    // world space already
    VkAccelerationStructureDeviceAddressInfoKHR address_info = {};
    address_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = m_blas;
    VkAccelerationStructureInstanceKHR instance = {};
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags =
        VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR |
        VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    instance.accelerationStructureReference =
        vkGetAccelerationStructureDeviceAddressKHR(device, &address_info);

    VkBuffer instance_buffer = VK_NULL_HANDLE;
    VmaAllocation instance_memory = VK_NULL_HANDLE;
    VmaAllocationCreateInfo instance_alloc = {};
    instance_alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    instance_alloc.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    instance_alloc.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!m_vk->createBuffer(sizeof(instance),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, instance_alloc,
        instance_buffer, instance_memory))
    {
        vmaDestroyBuffer(allocator, blas_scratch, blas_scratch_memory);
        return false;
    }
    void* mapped = NULL;
    if (vmaMapMemory(allocator, instance_memory, &mapped) != VK_SUCCESS)
    {
        vmaDestroyBuffer(allocator, instance_buffer, instance_memory);
        vmaDestroyBuffer(allocator, blas_scratch, blas_scratch_memory);
        return false;
    }
    memcpy(mapped, &instance, sizeof(instance));
    vmaUnmapMemory(allocator, instance_memory);
    vmaFlushAllocation(allocator, instance_memory, 0, sizeof(instance));

    VkAccelerationStructureGeometryKHR tlas_geometry = {};
    tlas_geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlas_geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlas_geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlas_geometry.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlas_geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    tlas_geometry.geometry.instances.data.deviceAddress =
        getAddress(instance_buffer);
    VkAccelerationStructureBuildGeometryInfoKHR tlas_info = {};
    tlas_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlas_info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlas_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tlas_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlas_info.geometryCount = 1;
    tlas_info.pGeometries = &tlas_geometry;
    const uint32_t instance_count = 1;
    VkAccelerationStructureBuildSizesInfoKHR tlas_sizes = {};
    tlas_sizes.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(device,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlas_info,
        &instance_count, &tlas_sizes);

    VkBuffer tlas_scratch = VK_NULL_HANDLE;
    VmaAllocation tlas_scratch_memory = VK_NULL_HANDLE;
    bool ok = createAccelerationStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        tlas_sizes.accelerationStructureSize, m_tlas_buffer, m_tlas_memory,
        m_tlas) && createScratchBuffer(tlas_sizes.buildScratchSize,
        tlas_scratch, tlas_scratch_memory, tlas_info.scratchData.deviceAddress);
    if (ok)
    {
        tlas_info.dstAccelerationStructure = m_tlas;

        VkCommandBuffer cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
        const VkAccelerationStructureBuildRangeInfoKHR* blas_ranges =
            ranges.data();
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &blas_info, &blas_ranges);

        // The TLAS build reads the BLAS
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
            &barrier, 0, NULL, 0, NULL);

        VkAccelerationStructureBuildRangeInfoKHR tlas_range = {};
        tlas_range.primitiveCount = instance_count;
        const VkAccelerationStructureBuildRangeInfoKHR* tlas_range_ptr =
            &tlas_range;
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_info,
            &tlas_range_ptr);
        // The submit waits for the fence
        GEVulkanCommandLoader::endSingleTimeCommands(cmd);
    }
    vmaDestroyBuffer(allocator, instance_buffer, instance_memory);
    vmaDestroyBuffer(allocator, blas_scratch, blas_scratch_memory);
    if (tlas_scratch != VK_NULL_HANDLE)
        vmaDestroyBuffer(allocator, tlas_scratch, tlas_scratch_memory);
    return ok;
}   // buildAccelerationStructures

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::createGeometryData(
                                        const std::vector<GERTGeometry>& infos)
{
    m_geometry_infos = infos;
    const size_t size = infos.size() * sizeof(GERTGeometry);
    m_geometry_data.reset(new GEVulkanDynamicBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, size,
        GEVulkanDriver::getMaxFrameInFlight() + 1, 0));
    // Every buffer has the built data, so any of them can be used right after
    // the build (and always without m_dynamic_ray_tracing)
    for (unsigned i = 0; i < m_geometry_data->getHostBuffer().size(); i++)
    {
        if (m_geometry_data->getHostBuffer()[i] == VK_NULL_HANDLE ||
            m_geometry_data->getMappedAddr()[i] == NULL)
            return false;
        memcpy(m_geometry_data->getMappedAddr()[i], infos.data(), size);
        vmaFlushAllocation(m_vk->getVmaAllocator(),
            m_geometry_data->getHostMemory()[i], 0, size);
    }
    return true;
}   // createGeometryData

// ----------------------------------------------------------------------------
void GEVulkanRTScene::update()
{
    if (!isReady() || !m_geometry_data)
        return;
    // The buffer of this frame is not used by the gpu (frames in flight + 1
    // of them), and the submit makes the host write visible. The mapped memory
    // is only written, never read
    const unsigned buffer_idx = m_vk->getCurrentBufferIdx();
    GERTGeometry* mapped_addr =
        (GERTGeometry*)m_geometry_data->getMappedAddr()[buffer_idx];
    for (size_t i = 0; i < m_geometry_sources.size(); i++)
    {
        GERTGeometry* obj = mapped_addr + i;
        *obj = m_geometry_infos[i];
        const GeometrySource& source = m_geometry_sources[i];
        if (source.m_material_index >= source.m_node->getMaterialCount())
            continue;
        irr::video::SMaterial& m =
            source.m_node->getMaterial(source.m_material_index);
        // Same as ObjectData::init (the const overload doesn't create a
        // texture matrix)
        const irr::video::SMaterial& cm = m;
        const irr::core::matrix4& texture_matrix = cm.getTextureMatrix(0);
        obj->m_uv_shift[0] = texture_matrix[8];
        obj->m_uv_shift[1] = texture_matrix[9];
        const std::shared_ptr<GERenderInfo>& ri = m.getRenderInfo();
        if (!ri)
            continue;
        obj->m_hue = ri->getHue() > 0.0f ? ri->getHue() : 0.0f;
        // Same as build (the ray traced pass needs PBR)
        obj->m_vertex_color =
            srgb255ToLinearFromSColor(ri->getVertexColor()).color;
    }
    vmaFlushAllocation(m_vk->getVmaAllocator(),
        m_geometry_data->getHostMemory()[buffer_idx], 0,
        m_geometry_sources.size() * sizeof(GERTGeometry));
}   // update

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::build(irr::scene::ISceneNode* root)
{
    clear();
    if (!GEVulkanFeatures::supportsRayQuery() ||
        !GEVulkanFeatures::supportsComputeInMainQueue() || !root)
        return false;

    std::vector<RTSource> sources;
    collectNode(root, sources);
    if (sources.empty())
        return false;

    const uint32_t max_geometry =
        GEVulkanFeatures::getAccelerationStructureMaxGeometryCount();
    if (max_geometry != 0 && sources.size() > max_geometry)
    {
        GE::logMessage(GLL_WARNING, "GEVulkanRTScene: "
            "too many geometries, only the first ones are used");
        sources.resize(max_geometry);
    }

    // Bake the world space vertices (decoded here, so no packed format is
    // needed in the shaders) and the indices, relative to the first vertex of
    // each geometry
    std::vector<GERTVertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<GERTGeometry> infos;
    std::vector<std::pair<uint32_t, uint32_t> > counts;
    GEVulkanTextureDescriptor* td = m_vk->getMeshTextureDescriptor();
    for (RTSource& s : sources)
    {
        GESPMBuffer* buffer = s.m_buffer;
        irr::video::SMaterial& m = s.m_node->getMaterial(s.m_material_index);
        const std::shared_ptr<GERenderInfo>& ri = m.getRenderInfo();
        s.m_node->updateAbsolutePosition();
        const irr::core::matrix4& abs = s.m_node->getAbsoluteTransformation();
        irr::core::matrix4 normal_matrix;
        if (abs.getInverse(normal_matrix))
            normal_matrix = normal_matrix.getTransposed();
        else
            normal_matrix.makeIdentity();

        GERTGeometry info = {};
        info.m_vertex_offset = (uint32_t)vertices.size();
        info.m_index_offset = (uint32_t)indices.size();
        // Slots are shared with the node (same material gives the same slot)
        // and kept alive by us
        std::shared_ptr<int>& slot = td->getTextureID(m,
            GEMaterialManager::getMaterial(s.m_node->getMaterialType(
            s.m_material_index)));
        m_texture_ids.push_back(slot);
        info.m_material_id = (uint32_t)*slot;
        info.m_shading_type = s.m_shading_type;
        // Same as ObjectData::init, update() keeps them current, the const
        // overload doesn't create a texture matrix
        const irr::video::SMaterial& cm = m;
        const irr::core::matrix4& texture_matrix = cm.getTextureMatrix(0);
        info.m_uv_shift[0] = texture_matrix[8];
        info.m_uv_shift[1] = texture_matrix[9];
        info.m_hue = ri->getHue() > 0.0f ? ri->getHue() : 0.0f;
        info.m_vertex_color =
            srgb255ToLinearFromSColor(ri->getVertexColor()).color;
        s.m_node->grab();
        m_geometry_sources.push_back({ s.m_node, s.m_material_index });

        const irr::video::S3DVertexSkinnedMesh* src =
            static_cast<const irr::video::S3DVertexSkinnedMesh*>(
            buffer->getVertices());
        for (unsigned i = 0; i < buffer->getVertexCount(); i++)
        {
            irr::core::vector3df pos = src[i].m_position;
            abs.transformVect(pos);
            irr::core::vector3df normal =
                MiniGLM::decompressVector3(src[i].m_normal);
            normal_matrix.rotateVect(normal);
            normal.normalize();
            GERTVertex v;
            v.m_position[0] = pos.X;
            v.m_position[1] = pos.Y;
            v.m_position[2] = pos.Z;
            v.m_color = src[i].m_color.color;
            v.m_normal[0] = normal.X;
            v.m_normal[1] = normal.Y;
            v.m_normal[2] = normal.Z;
            v.m_uv = (uint32_t)(uint16_t)src[i].m_all_uvs[0] |
                ((uint32_t)(uint16_t)src[i].m_all_uvs[1] << 16);
            vertices.push_back(v);
        }
        const irr::u16* idx = buffer->getIndices();
        const uint32_t index_count = buffer->getIndexCount() / 3 * 3;
        for (uint32_t i = 0; i < index_count; i++)
            indices.push_back(idx[i]);
        infos.push_back(info);
        counts.push_back({ buffer->getVertexCount(), index_count });
    }

    // The indices go after the vertices in the data buffer
    const VkDeviceSize vertex_bytes = vertices.size() * sizeof(GERTVertex);
    const VkDeviceSize index_bytes = indices.size() * sizeof(uint32_t);
    const uint32_t index_base = (uint32_t)(vertex_bytes / sizeof(uint32_t));
    for (GERTGeometry& info : infos)
        info.m_index_offset += index_base;
    std::vector<uint8_t> data((size_t)(vertex_bytes + index_bytes));
    memcpy(data.data(), vertices.data(), (size_t)vertex_bytes);
    memcpy(data.data() + vertex_bytes, indices.data(), (size_t)index_bytes);

    if (!createDeviceBuffer(data.data(), data.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        m_data_buffer, m_data_memory) ||
        !createGeometryData(infos))
    {
        GE::logMessage(GLL_ERROR,
            "GEVulkanRTScene: creating buffers failed");
        clear();
        return false;
    }

    const VkDeviceAddress base = getAddress(m_data_buffer);
    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    geometries.reserve(infos.size());
    ranges.reserve(infos.size());
    uint32_t triangles = 0;
    for (unsigned i = 0; i < infos.size(); i++)
    {
        VkAccelerationStructureGeometryKHR g = {};
        g.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        g.geometry.triangles.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        g.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        g.geometry.triangles.vertexData.deviceAddress = base +
            (VkDeviceAddress)infos[i].m_vertex_offset * sizeof(GERTVertex);
        g.geometry.triangles.vertexStride = sizeof(GERTVertex);
        g.geometry.triangles.maxVertex = counts[i].first - 1;
        g.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
        g.geometry.triangles.indexData.deviceAddress = base +
            (VkDeviceAddress)infos[i].m_index_offset * sizeof(uint32_t);
        geometries.push_back(g);
        VkAccelerationStructureBuildRangeInfoKHR range = {};
        range.primitiveCount = counts[i].second / 3;
        ranges.push_back(range);
        triangles += range.primitiveCount;
    }

    if (!buildAccelerationStructures(geometries, ranges))
    {
        GE::logMessage(GLL_ERROR, "GEVulkanRTScene: building failed");
        clear();
        return false;
    }
    m_geometry_count = (uint32_t)infos.size();
    m_triangle_count = triangles;
    updateDescriptorSet();
    GE::logMessage(GLL_INFO, "GEVulkanRTScene: built %d geometries",
        m_geometry_count);
    return true;
}   // build

}
