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
#include "ge_vulkan_mesh_cache.hpp"
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

// ----------------------------------------------------------------------------
// A node with GRTT_DYNAMIC meshes buffers, the same checks as collectNode but
// the mesh is not baked, it's a BLAS of its own shared by the nodes using it
struct DynamicSource
{
    irr::scene::ISceneNode* m_node;
    irr::scene::IMesh* m_mesh;
    // Bits of the mesh buffers which are geometries
    uint64_t m_buffer_mask;
    std::vector<unsigned> m_buffers;
    std::vector<uint32_t> m_shading_types;
};   // DynamicSource

// ----------------------------------------------------------------------------
void collectDynamicNode(irr::scene::ISceneNode* node,
                        std::vector<DynamicSource>& sources)
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
            DynamicSource source;
            source.m_node = node;
            source.m_mesh = mesh;
            source.m_buffer_mask = 0;
            for (unsigned i = 0; i < mesh->getMeshBufferCount() && i < 64; i++)
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
                const std::shared_ptr<GERenderInfo>& ri =
                    node->getMaterial(i).getRenderInfo();
                if (!ri || ri->getRayTracingType() != GRTT_DYNAMIC)
                    continue;
                uint32_t type = 0;
                if (!getShadingType(GEMaterialManager::getMaterial(
                    node->getMaterialType(i)), &type))
                    continue;
                source.m_buffer_mask |= (uint64_t)1 << i;
                source.m_buffers.push_back(i);
                source.m_shading_types.push_back(type);
            }
            if (!source.m_buffers.empty())
                sources.push_back(source);
        }
    }
    const irr::core::array<irr::scene::ISceneNode*>& children =
        node->getChildren();
    for (irr::u32 i = 0; i < children.size(); i++)
        collectDynamicNode(children[i], sources);
}   // collectDynamicNode

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
    m_instance_capacity = MAX_INSTANCES;
    m_tlas_scratch_size = 0;
    m_cache_version = 0;
    m_frame_counter = 0;
    m_warned_dynamic_limit = false;
    m_geometry_count = 0;
    m_triangle_count = 0;

    std::array<VkDescriptorSetLayoutBinding, 4> bindings = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    // The mesh cache of the driver
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
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
    sizes[1].descriptorCount = set_count * 3;
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
    VkDescriptorBufferInfo data_info = { m_data_buffer, 0, VK_WHOLE_SIZE };
    GEVulkanMeshCache* mesh_cache = m_vk->getVulkanMeshCache();
    VkDescriptorBufferInfo cache_info =
        { getMeshCacheDescriptorBuffer(mesh_cache), 0, VK_WHOLE_SIZE };
    for (unsigned set = 0; set < m_descriptor_sets.size(); set++)
    {
        // Each set reads the TLAS and the host buffer of the same index
        VkWriteDescriptorSetAccelerationStructureKHR as_write = {};
        as_write.sType =
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
        as_write.accelerationStructureCount = 1;
        as_write.pAccelerationStructures = &m_top_levels[set].m_tlas;
        VkDescriptorBufferInfo geometry_info =
            { m_geometry_data->getHostBuffer()[set], 0, VK_WHOLE_SIZE };
        std::array<VkWriteDescriptorSet, 4> writes = {};
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
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[3].pBufferInfo = &cache_info;
        vkUpdateDescriptorSets(m_vk->getDevice(), writes.size(),
            writes.data(), 0, NULL);
        m_top_levels[set].m_cache_version =
            mesh_cache ? mesh_cache->getUpdateCount() : 0;
    }
}   // updateDescriptorSet

// ----------------------------------------------------------------------------
VkBuffer GEVulkanRTScene::getMeshCacheDescriptorBuffer(
                                        GEVulkanMeshCache* mesh_cache) const
{
    // A descriptor has to be valid even if the shader doesn't read it
    if (mesh_cache && mesh_cache->getBuffer() != VK_NULL_HANDLE)
        return mesh_cache->getBuffer();
    return m_data_buffer;
}   // getMeshCacheDescriptorBuffer

// ----------------------------------------------------------------------------
// The descriptor set of this index is not used by the gpu
void GEVulkanRTScene::writeMeshCacheDescriptor(unsigned buffer_idx,
                                               GEVulkanMeshCache* mesh_cache)
{
    VkDescriptorBufferInfo cache_info =
        { getMeshCacheDescriptorBuffer(mesh_cache), 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_descriptor_sets[buffer_idx];
    write.dstBinding = 3;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &cache_info;
    vkUpdateDescriptorSets(m_vk->getDevice(), 1, &write, 0, NULL);
    m_top_levels[buffer_idx].m_cache_version =
        mesh_cache ? mesh_cache->getUpdateCount() : 0;
}   // writeMeshCacheDescriptor

// ----------------------------------------------------------------------------
void GEVulkanRTScene::clear()
{
    if (m_top_levels.empty() && m_mesh_blas.empty() &&
        m_blas == VK_NULL_HANDLE &&
        m_data_buffer == VK_NULL_HANDLE && !m_geometry_data &&
        m_geometry_sources.empty() && m_texture_ids.empty())
        return;
    m_vk->waitIdle();
    VmaAllocator allocator = m_vk->getVmaAllocator();
    // The device is idle, so the BLAS of the meshes can go too
    m_pending_blas.clear();
    while (!m_mesh_blas.empty())
        destroyMeshBLAS(m_mesh_blas.begin()->second.get());
    m_cache_version = 0;
    for (TopLevel& t : m_top_levels)
    {
        if (t.m_scratch_buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator, t.m_scratch_buffer, t.m_scratch_memory);
        if (t.m_tlas != VK_NULL_HANDLE)
            vkDestroyAccelerationStructureKHR(m_vk->getDevice(), t.m_tlas, NULL);
        if (t.m_tlas_buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(allocator, t.m_tlas_buffer, t.m_tlas_memory);
        if (t.m_instance_buffer != VK_NULL_HANDLE)
        {
            if (t.m_instance_mapped)
                vmaUnmapMemory(allocator, t.m_instance_memory);
            vmaDestroyBuffer(allocator, t.m_instance_buffer,
                t.m_instance_memory);
        }
    }
    m_top_levels.clear();
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
    m_blas = VK_NULL_HANDLE;
    m_blas_buffer = m_data_buffer = VK_NULL_HANDLE;
    m_blas_memory = m_data_memory = VK_NULL_HANDLE;
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
VkAccelerationStructureInstanceKHR GEVulkanRTScene::getStaticInstance() const
{
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
        vkGetAccelerationStructureDeviceAddressKHR(m_vk->getDevice(),
        &address_info);
    return instance;
}   // getStaticInstance

// ----------------------------------------------------------------------------
void GEVulkanRTScene::setupTopLevelBuild(const TopLevel& top_level,
    VkAccelerationStructureGeometryKHR& geometry,
    VkAccelerationStructureBuildGeometryInfoKHR& info) const
{
    geometry = {};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geometry.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress =
        getAddress(top_level.m_instance_buffer);
    info = {};
    info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    // It's built again every time the instances change
    info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount = 1;
    info.pGeometries = &geometry;
    info.dstAccelerationStructure = top_level.m_tlas;
}   // setupTopLevelBuild

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

    // One TLAS (and instance buffer) for each buffer index, all with the same
    // single instance of the BLAS for now
    VkAccelerationStructureInstanceKHR instance = getStaticInstance();
    const unsigned top_level_count = GEVulkanDriver::getMaxFrameInFlight() + 1;
    m_top_levels.resize(top_level_count);
    VmaAllocationCreateInfo instance_alloc = {};
    instance_alloc.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    instance_alloc.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    instance_alloc.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkDeviceSize instance_bytes =
        (VkDeviceSize)m_instance_capacity * sizeof(instance);
    bool ok = true;
    for (TopLevel& t : m_top_levels)
    {
        if (!m_vk->createBuffer(instance_bytes,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, instance_alloc,
            t.m_instance_buffer, t.m_instance_memory) ||
            vmaMapMemory(allocator, t.m_instance_memory,
            &t.m_instance_mapped) != VK_SUCCESS)
        {
            t.m_instance_mapped = NULL;
            ok = false;
            break;
        }
        memcpy(t.m_instance_mapped, &instance, sizeof(instance));
        vmaFlushAllocation(allocator, t.m_instance_memory, 0, instance_bytes);
    }

    VkAccelerationStructureGeometryKHR tlas_geometry = {};
    VkAccelerationStructureBuildGeometryInfoKHR tlas_info = {};
    VkAccelerationStructureBuildSizesInfoKHR tlas_sizes = {};
    if (ok)
    {
        setupTopLevelBuild(m_top_levels[0], tlas_geometry, tlas_info);
        tlas_sizes.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        vkGetAccelerationStructureBuildSizesKHR(device,
            VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlas_info,
            &m_instance_capacity, &tlas_sizes);
        m_tlas_scratch_size = tlas_sizes.buildScratchSize;
    }

    VkBuffer tlas_scratch = VK_NULL_HANDLE;
    VmaAllocation tlas_scratch_memory = VK_NULL_HANDLE;
    if (ok)
    {
        for (TopLevel& t : m_top_levels)
        {
            if (!createAccelerationStructure(
                VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                tlas_sizes.accelerationStructureSize, t.m_tlas_buffer,
                t.m_tlas_memory, t.m_tlas))
            {
                ok = false;
                break;
            }
        }
    }
    VkDeviceAddress tlas_scratch_address = 0;
    ok = ok && createScratchBuffer(tlas_sizes.buildScratchSize, tlas_scratch,
        tlas_scratch_memory, tlas_scratch_address);
    if (ok)
    {
        VkCommandBuffer cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
        const VkAccelerationStructureBuildRangeInfoKHR* blas_ranges =
            ranges.data();
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &blas_info, &blas_ranges);

        // The TLAS builds read the BLAS, and they share the scratch buffer
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask =
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        // Only the static instance
        VkAccelerationStructureBuildRangeInfoKHR tlas_range = {};
        tlas_range.primitiveCount = 1;
        const VkAccelerationStructureBuildRangeInfoKHR* tlas_range_ptr =
            &tlas_range;
        for (TopLevel& t : m_top_levels)
        {
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
                &barrier, 0, NULL, 0, NULL);
            setupTopLevelBuild(t, tlas_geometry, tlas_info);
            tlas_info.scratchData.deviceAddress = tlas_scratch_address;
            vkCmdBuildAccelerationStructuresKHR(cmd, 1, &tlas_info,
                &tlas_range_ptr);
        }
        // The submit waits for the fence
        GEVulkanCommandLoader::endSingleTimeCommands(cmd);
    }
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
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        size + MAX_DYNAMIC_GEOMETRIES * sizeof(GERTGeometry),
        GEVulkanDriver::getMaxFrameInFlight() + 1, 0));
    // Every buffer has the built data (the static geometries), so any of them
    // can be used right after the build (and always without
    // m_dynamic_ray_tracing)
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
GEVulkanRTScene::MeshBLAS* GEVulkanRTScene::getMeshBLAS(
                            irr::scene::IMesh* mesh, uint64_t buffer_mask,
                            const std::vector<unsigned>& buffer_indices,
                            const std::vector<uint32_t>& shading_types,
                            GEVulkanMeshCache* mesh_cache)
{
    MeshKey key = { mesh, buffer_mask };
    auto it = m_mesh_blas.find(key);
    if (it != m_mesh_blas.end())
        return it->second.get();

    // A mesh which can't be created is also kept (without m_blas), so it's
    // not tried every frame, it's destroyed when it's not used anymore
    std::unique_ptr<MeshBLAS> blas(new MeshBLAS());
    MeshBLAS* ret = blas.get();
    ret->m_last_used_frame = m_frame_counter;
    m_mesh_blas[key] = std::move(blas);

    // The vertices (32 bytes, position first) and the 16 bit indices of the
    // mesh buffers are used as they are in the mesh cache buffer
    const VkDeviceAddress cache_address = getAddress(mesh_cache->getBuffer());
    const size_t static_pitch = sizeof(GERTVertex);
    for (unsigned g = 0; g < buffer_indices.size(); g++)
    {
        GESPMBuffer* buffer = static_cast<GESPMBuffer*>(
            mesh->getMeshBuffer(buffer_indices[g]));
        const uint32_t index_count = buffer->getIndexCount() / 3 * 3;
        GERTGeometry info = {};
        info.m_vertex_offset = (uint32_t)buffer->getVBOOffset();
        info.m_index_offset = (uint32_t)(mesh_cache->getIBOOffset() /
            sizeof(uint16_t) + buffer->getIBOOffset());
        info.m_shading_type = shading_types[g];
        info.m_flags = GRTGF_CACHE_VERTEX | GRTGF_CACHE_INDEX;
        ret->m_infos.push_back(info);

        VkAccelerationStructureGeometryKHR geometry = {};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
        geometry.geometry.triangles.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress = cache_address +
            (VkDeviceAddress)info.m_vertex_offset * static_pitch;
        geometry.geometry.triangles.vertexStride = static_pitch;
        geometry.geometry.triangles.maxVertex = buffer->getVertexCount() - 1;
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT16;
        geometry.geometry.triangles.indexData.deviceAddress = cache_address +
            (VkDeviceAddress)info.m_index_offset * sizeof(uint16_t);
        ret->m_as_geometries.push_back(geometry);
        VkAccelerationStructureBuildRangeInfoKHR range = {};
        range.primitiveCount = index_count / 3;
        ret->m_as_ranges.push_back(range);
    }

    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = (uint32_t)ret->m_as_geometries.size();
    build_info.pGeometries = ret->m_as_geometries.data();
    std::vector<uint32_t> primitive_counts;
    for (const auto& range : ret->m_as_ranges)
        primitive_counts.push_back(range.primitiveCount);
    VkAccelerationStructureBuildSizesInfoKHR sizes = {};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(m_vk->getDevice(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info,
        primitive_counts.data(), &sizes);
    if (!createAccelerationStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        sizes.accelerationStructureSize, ret->m_blas_buffer,
        ret->m_blas_memory, ret->m_blas))
    {
        GE::logMessage(GLL_ERROR, "GEVulkanRTScene: creating the BLAS of a "
            "dynamic mesh failed");
        // Not built, so it must not be used
        if (ret->m_blas != VK_NULL_HANDLE)
        {
            vkDestroyAccelerationStructureKHR(m_vk->getDevice(), ret->m_blas,
                NULL);
        }
        ret->m_blas = VK_NULL_HANDLE;
        return ret;
    }
    ret->m_build_scratch_size = sizes.buildScratchSize;
    VkAccelerationStructureDeviceAddressInfoKHR address_info = {};
    address_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = ret->m_blas;
    ret->m_address = vkGetAccelerationStructureDeviceAddressKHR(
        m_vk->getDevice(), &address_info);
    m_pending_blas.push_back(ret);
    return ret;
}   // getMeshBLAS

// ----------------------------------------------------------------------------
void GEVulkanRTScene::destroyMeshBLAS(MeshBLAS* blas)
{
    for (auto it = m_mesh_blas.begin(); it != m_mesh_blas.end(); it++)
    {
        if (it->second.get() != blas)
            continue;
        for (auto p = m_pending_blas.begin(); p != m_pending_blas.end(); p++)
        {
            if (*p == blas)
            {
                m_pending_blas.erase(p);
                break;
            }
        }
        if (blas->m_blas != VK_NULL_HANDLE)
            vkDestroyAccelerationStructureKHR(m_vk->getDevice(), blas->m_blas,
                NULL);
        if (blas->m_blas_buffer != VK_NULL_HANDLE)
            vmaDestroyBuffer(m_vk->getVmaAllocator(), blas->m_blas_buffer,
                blas->m_blas_memory);
        m_mesh_blas.erase(it);
        return;
    }
}   // destroyMeshBLAS

// ----------------------------------------------------------------------------
// Not used by the instances of the frames which can be still on the gpu
// (the TLAS of a buffer index and the shaders use the BLAS and its data)
void GEVulkanRTScene::destroyUnusedMeshBLAS()
{
    const uint64_t grace = GEVulkanDriver::getMaxFrameInFlight() + 2;
    std::vector<MeshBLAS*> unused;
    for (auto& p : m_mesh_blas)
    {
        if (m_frame_counter - p.second->m_last_used_frame > grace)
            unused.push_back(p.second.get());
    }
    for (MeshBLAS* blas : unused)
        destroyMeshBLAS(blas);
}   // destroyUnusedMeshBLAS

// ----------------------------------------------------------------------------
void GEVulkanRTScene::update(irr::scene::ISceneNode* root)
{
    if (!isReady() || !m_geometry_data)
        return;
    m_frame_counter++;
    // The buffers of this frame are not used by the gpu (frames in flight + 1
    // of them), and the submit makes the host write visible. The mapped memory
    // is only written, never read
    const unsigned buffer_idx = m_vk->getCurrentBufferIdx();
    TopLevel& t = m_top_levels[buffer_idx];
    GERTGeometry* mapped_addr =
        (GERTGeometry*)m_geometry_data->getMappedAddr()[buffer_idx];
    VmaAllocator allocator = m_vk->getVmaAllocator();
    const size_t static_count = m_geometry_infos.size();

    GEVulkanMeshCache* mesh_cache = m_vk->getVulkanMeshCache();
    const uint64_t cache_version =
        mesh_cache ? mesh_cache->getUpdateCount() : 0;
    if (cache_version != m_cache_version)
    {
        // The buffer and the offsets of all meshes are new (and a mesh could
        // be gone, so a mesh address can be used by another one). The mesh
        // cache waited for the device before it destroyed its buffer, and the
        // frames of the TLAS with them are done
        m_pending_blas.clear();
        while (!m_mesh_blas.empty())
            destroyMeshBLAS(m_mesh_blas.begin()->second.get());
        m_cache_version = cache_version;
    }
    if (t.m_cache_version != cache_version)
        writeMeshCacheDescriptor(buffer_idx, mesh_cache);
    // The BLAS of the dynamic nodes are made of the mesh cache
    const bool dynamic = getGEConfig()->m_dynamic_ray_tracing &&
        GEVulkanFeatures::supportsBaseVertexRendering() && mesh_cache &&
        mesh_cache->getBuffer() != VK_NULL_HANDLE;

    if (!dynamic)
    {
        // Back to what the build wrote, once for each buffer index
        if (!t.m_geometry_static)
        {
            memcpy(mapped_addr, m_geometry_infos.data(),
                static_count * sizeof(GERTGeometry));
            vmaFlushAllocation(allocator,
                m_geometry_data->getHostMemory()[buffer_idx], 0,
                static_count * sizeof(GERTGeometry));
            t.m_geometry_static = true;
        }
        if (!t.m_instance_static)
        {
            VkAccelerationStructureInstanceKHR instance = getStaticInstance();
            memcpy(t.m_instance_mapped, &instance, sizeof(instance));
            vmaFlushAllocation(allocator, t.m_instance_memory, 0,
                sizeof(instance));
            t.m_instance_count = 1;
            t.m_instance_static = true;
            t.m_dirty = true;
            t.m_slots.clear();
        }
        destroyUnusedMeshBLAS();
        return;
    }

    // The static geometries, with what can change
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
        obj->m_hue = m.isColorizable() && ri->getHue() > 0.0f ?
            ri->getHue() : 0.0f;
        // Same as build (the ray traced pass needs PBR)
        obj->m_vertex_color =
            srgb255ToLinearFromSColor(ri->getVertexColor()).color;
    }
    t.m_geometry_static = false;

    // The dynamic nodes, found every frame (a node can be added, removed or
    // use another mesh)
    std::vector<DynamicSource> sources;
    if (root)
        collectDynamicNode(root, sources);
    t.m_slots.clear();
    VkAccelerationStructureInstanceKHR* instances =
        (VkAccelerationStructureInstanceKHR*)t.m_instance_mapped;
    GEVulkanTextureDescriptor* td = m_vk->getMeshTextureDescriptor();
    uint32_t instance_count = 1;
    size_t geometry_count = static_count;
    for (DynamicSource& s : sources)
    {
        MeshBLAS* blas = getMeshBLAS(s.m_mesh, s.m_buffer_mask, s.m_buffers,
            s.m_shading_types, mesh_cache);
        if (blas->m_blas == VK_NULL_HANDLE)
            continue;
        if (instance_count >= m_instance_capacity ||
            geometry_count - static_count + blas->m_infos.size() >
            MAX_DYNAMIC_GEOMETRIES)
        {
            if (!m_warned_dynamic_limit)
            {
                m_warned_dynamic_limit = true;
                GE::logMessage(GLL_WARNING, "GEVulkanRTScene: too many "
                    "dynamic instances, the other ones are not traced");
            }
            break;
        }
        blas->m_last_used_frame = m_frame_counter;

        // Row major 3x4 of the column major 4x4 of the node
        irr::scene::ISceneNode* node = s.m_node;
        const irr::core::matrix4& abs = node->getAbsoluteTransformation();
        VkAccelerationStructureInstanceKHR instance = {};
        for (unsigned r = 0; r < 3; r++)
        {
            for (unsigned c = 0; c < 4; c++)
                instance.transform.matrix[r][c] = abs[c * 4 + r];
        }
        instance.instanceCustomIndex = (uint32_t)geometry_count;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags =
            VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR |
            VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        instance.accelerationStructureReference = blas->m_address;
        instances[instance_count++] = instance;

        for (unsigned g = 0; g < blas->m_infos.size(); g++)
        {
            GERTGeometry info = blas->m_infos[g];
            const unsigned material_index = s.m_buffers[g];
            irr::video::SMaterial& m = node->getMaterial(material_index);
            std::shared_ptr<int> slot = td->getTextureID(m,
                GEMaterialManager::getMaterial(
                node->getMaterialType(material_index)));
            t.m_slots.push_back(slot);
            info.m_material_id = (uint32_t)*slot;
            const irr::video::SMaterial& cm = m;
            const irr::core::matrix4& texture_matrix = cm.getTextureMatrix(0);
            info.m_uv_shift[0] = texture_matrix[8];
            info.m_uv_shift[1] = texture_matrix[9];
            const std::shared_ptr<GERenderInfo>& ri = m.getRenderInfo();
            info.m_hue = ri->getHue() > 0.0f ? ri->getHue() : 0.0f;
            info.m_vertex_color =
                srgb255ToLinearFromSColor(ri->getVertexColor()).color;
            mapped_addr[geometry_count++] = info;
        }
    }
    vmaFlushAllocation(allocator, m_geometry_data->getHostMemory()[buffer_idx],
        0, geometry_count * sizeof(GERTGeometry));

    // Without dynamic instances the TLAS of the build (or the last restore)
    // is what it should be
    if (instance_count > 1 || !t.m_instance_static)
    {
        instances[0] = getStaticInstance();
        vmaFlushAllocation(allocator, t.m_instance_memory, 0,
            instance_count * sizeof(VkAccelerationStructureInstanceKHR));
        t.m_dirty = true;
    }
    t.m_instance_count = instance_count;
    t.m_instance_static = instance_count == 1;
    destroyUnusedMeshBLAS();
}   // update

// ----------------------------------------------------------------------------
void GEVulkanRTScene::buildTopLevel(VkCommandBuffer cmd)
{
    if (!isReady())
        return;
    TopLevel& t = m_top_levels[m_vk->getCurrentBufferIdx()];
    if (!t.m_dirty)
        return;
    t.m_dirty = false;

    const VkDeviceSize alignment =
        GEVulkanFeatures::getAccelerationStructureScratchAlignment();
    auto align = [alignment](VkDeviceSize size)
        { return (size + alignment - 1) / alignment * alignment; };
    // The TLAS first, then the BLAS of the new meshes
    VkDeviceSize needed = align(m_tlas_scratch_size);
    for (MeshBLAS* blas : m_pending_blas)
        needed += align(blas->m_build_scratch_size);
    if (t.m_scratch_size < needed)
    {
        // The previous frame which used this index is finished
        if (t.m_scratch_buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_vk->getVmaAllocator(), t.m_scratch_buffer,
                t.m_scratch_memory);
        }
        t.m_scratch_buffer = VK_NULL_HANDLE;
        t.m_scratch_size = 0;
        if (!createScratchBuffer(needed, t.m_scratch_buffer,
            t.m_scratch_memory, t.m_scratch_address))
        {
            GE::logMessage(GLL_ERROR,
                "GEVulkanRTScene: creating the scratch buffer failed");
            return;
        }
        t.m_scratch_size = needed;
    }

    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    if (!m_pending_blas.empty())
    {
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> ranges;
        VkDeviceAddress scratch = t.m_scratch_address +
            align(m_tlas_scratch_size);
        for (MeshBLAS* blas : m_pending_blas)
        {
            VkAccelerationStructureBuildGeometryInfoKHR info = {};
            info.sType =
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            info.flags =
                VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
            info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            info.geometryCount = (uint32_t)blas->m_as_geometries.size();
            info.pGeometries = blas->m_as_geometries.data();
            info.dstAccelerationStructure = blas->m_blas;
            info.scratchData.deviceAddress = scratch;
            scratch += align(blas->m_build_scratch_size);
            infos.push_back(info);
            ranges.push_back(blas->m_as_ranges.data());
        }
        vkCmdBuildAccelerationStructuresKHR(cmd, (uint32_t)infos.size(),
            infos.data(), ranges.data());
        m_pending_blas.clear();
        // The TLAS build reads them (and uses the same scratch buffer)
        barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
            &barrier, 0, NULL, 0, NULL);
    }

    VkAccelerationStructureGeometryKHR geometry = {};
    VkAccelerationStructureBuildGeometryInfoKHR info = {};
    setupTopLevelBuild(t, geometry, info);
    info.scratchData.deviceAddress = t.m_scratch_address;
    VkAccelerationStructureBuildRangeInfoKHR range = {};
    range.primitiveCount = t.m_instance_count;
    const VkAccelerationStructureBuildRangeInfoKHR* range_ptr = &range;
    vkCmdBuildAccelerationStructuresKHR(cmd, 1, &info, &range_ptr);

    // The ray queries of the fragment shaders
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
        NULL);
}   // buildTopLevel

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
        info.m_hue = cm.isColorizable() && ri->getHue() > 0.0f ?
            ri->getHue() : 0.0f;
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
