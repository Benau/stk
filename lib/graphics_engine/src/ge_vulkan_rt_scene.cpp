#include "ge_vulkan_rt_scene.hpp"

#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_render_info.hpp"
#include "ge_spm.hpp"
#include "ge_spm_buffer.hpp"
#include "ge_vulkan_animated_mesh_scene_node.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_dynamic_buffer.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_mesh_cache.hpp"
#include "ge_vulkan_shader_manager.hpp"
#include "ge_vulkan_texture_descriptor.hpp"
#include "mini_glm.hpp"

#include "IMeshSceneNode.h"
#include "ISceneNode.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
// A2B10G10R10 snorm, x is the lowest bits (the format of the normals and
// tangents of the mesh cache), w are the 2 bits at the top
uint32_t packSnorm1010102(float x, float y, float z, uint32_t w)
{
    auto to_10 = [](float v) -> uint32_t
    {
        v = std::min(1.0f, std::max(-1.0f, v));
        return (uint32_t)(int32_t)std::lround(v * 511.0f) & 0x3FFu;
    };
    return to_10(x) | (to_10(y) << 10) | (to_10(z) << 20) | ((w & 3u) << 30);
}   // packSnorm1010102

// ----------------------------------------------------------------------------
// The push constants of grass_rt.comp
struct GrassPushConstants
{
    // The wind direction of grass.vert
    float m_wind[3];
    uint32_t m_source_word;
    uint32_t m_dst_word;
    uint32_t m_vertex_count;
};   // GrassPushConstants

// ----------------------------------------------------------------------------
struct RTSource
{
    GESPMBuffer* m_buffer;
    irr::scene::ISceneNode* m_node;
    unsigned m_material_index;
    uint32_t m_shading_type;
};   // RTSource

// ----------------------------------------------------------------------------
// The geometries which discard by the alpha of their texture are not opaque,
// so the ray query gives a candidate to confirm (solid_rt.frag)
VkGeometryFlagsKHR getGeometryFlags(uint32_t shading_type)
{
    return (shading_type == GRTST_ALPHATEST || shading_type == GRTST_UNLIT ||
        shading_type == GRTST_GRASS) ? 0 : VK_GEOMETRY_OPAQUE_BIT_KHR;
}   // getGeometryFlags

// ----------------------------------------------------------------------------
// The shading type of a material by its shader (the name of GEMaterialManager),
// only the ones which can be shaded by the ray traced shaders are here:
// transparent, ghost, etc. are not included, so the rays pass through them. The
// grass is only used by the dynamic nodes (GEVulkanRTScene::update)
bool getShadingType(irr::video::E_MATERIAL_TYPE mt, uint32_t* type)
{
    static std::vector<int> types;
    if (types.empty())
    {
        types.resize(irr::video::EMT_MATERIAL_COUNT, -1);
        types[GEMaterialManager::getIrrMaterialType("solid")] = GRTST_SOLID;
        types[GEMaterialManager::getIrrMaterialType("normalmap")] = GRTST_NORMALMAP;
        types[GEMaterialManager::getIrrMaterialType("decal")] = GRTST_DECAL;
        types[GEMaterialManager::getIrrMaterialType("splatting")] = GRTST_SPLATTING;
        types[GEMaterialManager::getIrrMaterialType("alphatest")] = GRTST_ALPHATEST;
        types[GEMaterialManager::getIrrMaterialType("unlit")] = GRTST_UNLIT;
        types[GEMaterialManager::getIrrMaterialType("grass")] = GRTST_GRASS;
    }
    int ret = types[mt];
    if (ret == -1)
        return false;
    // Not in the scene, so the rays pass through them
    if ((ret == GRTST_ALPHATEST || ret == GRTST_UNLIT) &&
        !getGEConfig()->m_ray_tracing.m_alphatest)
        return false;
    if (ret == GRTST_GRASS && !getGEConfig()->m_ray_tracing.m_grass)
        return false;
    *type = ret;
    return true;
}   // getShadingType

// ----------------------------------------------------------------------------
// visible is if the node and all its parents are visible, it's passed down by
// the parent (isChildVisibleForGE) instead of isTrulyVisible, so a parent (a
// LODNode) can show its most detailed level without changing the scene
// The meshes with the grass material (it has a vertex shader which moves the
// vertices) are not in the static BLAS, they are instances (collectDynamicNode)
void collectNode(irr::scene::ISceneNode* node, std::vector<RTSource>& sources,
                 bool visible)
{
    if (!visible)
        return;
    {
        irr::scene::IMesh* mesh = NULL;
        bool skinning_node = node->getType() == irr::scene::ESNT_ANIMATED_MESH;
        if (skinning_node)
            mesh = static_cast<GEVulkanAnimatedMeshSceneNode*>(node)->getMesh();
        else if (node->getType() == irr::scene::ESNT_MESH)
            mesh = static_cast<irr::scene::IMeshSceneNode*>(node)->getMesh();
        if (mesh)
        {
            for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
            {
                irr::scene::IMeshBuffer* b = mesh->getMeshBuffer(i);
                if (b->getVertexType() != irr::video::EVT_SKINNED_MESH ||
                    i >= node->getMaterialCount())
                    continue;
                GESPMBuffer* buffer = static_cast<GESPMBuffer*>(b);
                if ((skinning_node && buffer->hasSkinning()) ||
                    buffer->isDynamic() ||
                    buffer->getVertexCount() == 0 ||
                    buffer->getIndexCount() < 3)
                    continue;
                irr::video::SMaterial& m = node->getMaterial(i);
                const std::shared_ptr<GERenderInfo>& ri = m.getRenderInfo();
                if (!ri || ri->getRayTracingType() != GRTT_STATIC)
                    continue;
                static irr::video::E_MATERIAL_TYPE grass_type =
                    GEMaterialManager::getIrrMaterialType("grass");
                if (node->getMaterialType(i) == grass_type)
                    continue;
                uint32_t type = 0;
                if (!getShadingType(node->getMaterialType(i), &type))
                    continue;
                sources.push_back({ buffer, node, i, type });
            }
        }
    }
    const irr::core::array<irr::scene::ISceneNode*>& children =
        node->getChildren();
    for (irr::u32 i = 0; i < children.size(); i++)
        collectNode(children[i], sources,
            node->isChildVisibleForGE(children[i]));
}   // collectNode

// ----------------------------------------------------------------------------
// A geometry (mesh buffer with skinning) of an animated node with GRTT_DYNAMIC
struct SkinSource
{
    irr::scene::ISceneNode* m_node;
    GESPMBuffer* m_buffer;
    unsigned m_material_index;
    uint32_t m_shading_type;
};   // SkinSource

// ----------------------------------------------------------------------------
// A node with GRTT_DYNAMIC meshes buffers, the same checks as collectNode but
// the mesh is not baked, it's a BLAS of its own shared by the nodes using it.
// The skinned geometries (SkinSource) are collected in the same walk, a buffer
// with skinning is never in both
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
// The mesh buffers with the grass material (any GRTT_ but GRTT_NONE, the static
// ones are not in the static BLAS) are in grass, with their own mask
void collectDynamicNode(irr::scene::ISceneNode* node,
                        std::vector<DynamicSource>& sources,
                        std::vector<DynamicSource>& grass,
                        std::vector<SkinSource>& skins, bool visible)
{
    if (!visible)
        return;
    {
        irr::scene::IMesh* mesh = NULL;
        bool skinning_node = node->getType() == irr::scene::ESNT_ANIMATED_MESH;
        if (skinning_node)
            mesh = static_cast<GEVulkanAnimatedMeshSceneNode*>(node)->getMesh();
        else if (node->getType() == irr::scene::ESNT_MESH)
            mesh = static_cast<irr::scene::IMeshSceneNode*>(node)->getMesh();
        if (mesh)
        {
            static irr::video::E_MATERIAL_TYPE grass_type =
                GEMaterialManager::getIrrMaterialType("grass");
            DynamicSource source;
            source.m_node = node;
            source.m_mesh = mesh;
            source.m_buffer_mask = 0;
            DynamicSource grass_source = source;
            for (unsigned i = 0; i < mesh->getMeshBufferCount() && i < 64; i++)
            {
                irr::scene::IMeshBuffer* b = mesh->getMeshBuffer(i);
                if (b->getVertexType() != irr::video::EVT_SKINNED_MESH ||
                    i >= node->getMaterialCount())
                    continue;
                GESPMBuffer* buffer = static_cast<GESPMBuffer*>(b);
                if ((skinning_node && buffer->hasSkinning()) ||
                    buffer->isDynamic() ||
                    buffer->getVertexCount() == 0 ||
                    buffer->getIndexCount() < 3)
                    continue;
                const std::shared_ptr<GERenderInfo>& ri =
                    node->getMaterial(i).getRenderInfo();
                if (!ri || ri->getRayTracingType() == GRTT_NONE)
                    continue;
                const bool is_grass = node->getMaterialType(i) == grass_type;
                if (!is_grass && ri->getRayTracingType() != GRTT_DYNAMIC)
                    continue;
                uint32_t type = 0;
                if (!getShadingType(node->getMaterialType(i), &type))
                    continue;
                DynamicSource& target = is_grass ? grass_source : source;
                target.m_buffer_mask |= (uint64_t)1 << i;
                target.m_buffers.push_back(i);
                target.m_shading_types.push_back(type);
            }
            if (!source.m_buffers.empty())
                sources.push_back(source);
            if (!grass_source.m_buffers.empty())
                grass.push_back(grass_source);
        }
    }
    if (node->getType() == irr::scene::ESNT_ANIMATED_MESH)
    {
        irr::scene::IMesh* mesh =
            static_cast<GEVulkanAnimatedMeshSceneNode*>(node)->getMesh();
        if (mesh && !static_cast<GESPM*>(mesh)->isStatic())
        {
            for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
            {
                irr::scene::IMeshBuffer* b = mesh->getMeshBuffer(i);
                if (b->getVertexType() != irr::video::EVT_SKINNED_MESH ||
                    i >= node->getMaterialCount())
                    continue;
                GESPMBuffer* buffer = static_cast<GESPMBuffer*>(b);
                if (!buffer->hasSkinning())
                    continue;
                const std::shared_ptr<GERenderInfo>& ri =
                    node->getMaterial(i).getRenderInfo();
                if (!ri || ri->getRayTracingType() != GRTT_DYNAMIC)
                    continue;
                if (buffer->isDynamic() || buffer->getVertexCount() == 0 ||
                    buffer->getIndexCount() < 3)
                    continue;
                uint32_t type = 0;
                if (!getShadingType(node->getMaterialType(i), &type))
                    continue;
                skins.push_back({ node, buffer, i, type });
            }
        }
    }
    const irr::core::array<irr::scene::ISceneNode*>& children =
        node->getChildren();
    for (irr::u32 i = 0; i < children.size(); i++)
        collectDynamicNode(children[i], sources, grass, skins,
            node->isChildVisibleForGE(children[i]));
}   // collectDynamicNode

// ----------------------------------------------------------------------------
// What changes of a geometry of a dynamic node (the material id of the mesh
// textures, uv shift, hue and custom vertex color), the slot is kept alive
void setDynamicMaterial(GERTGeometry& info, irr::scene::ISceneNode* node,
                        unsigned material_index, GEVulkanTextureDescriptor* td,
                        std::vector<std::shared_ptr<int> >& slots)
{
    irr::video::SMaterial& m = node->getMaterial(material_index);
    std::shared_ptr<int> slot = td->getTextureID(m,
        GEMaterialManager::getMaterial(node->getMaterialType(material_index)));
    slots.push_back(slot);
    info.m_material_id = (uint32_t)*slot;
    const irr::video::SMaterial& cm = m;
    const irr::core::matrix4& texture_matrix = cm.getTextureMatrix(0);
    info.m_uv_shift[0] = texture_matrix[8];
    info.m_uv_shift[1] = texture_matrix[9];
    const std::shared_ptr<GERenderInfo>& ri = m.getRenderInfo();
    info.m_hue = cm.isColorizable() && ri->getHue() > 0.0f ?
        ri->getHue() : 0.0f;
    info.m_vertex_color = srgb255ToLinearFromSColor(ri->getVertexColor()).color;
}   // setDynamicMaterial

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
    m_built_alphatest = false;
    m_grass_next_id = 0;
    m_grass_pipeline = VK_NULL_HANDLE;
    m_grass_pipeline_failed = false;
    m_skin_layout = VK_NULL_HANDLE;
    m_skin_pool = VK_NULL_HANDLE;
    m_skin_pipeline_layout = VK_NULL_HANDLE;
    m_skin_pipeline = VK_NULL_HANDLE;
    m_skin_pipeline_failed = false;
    m_skin_region_offset = 0;
    m_skin_block = VK_NULL_HANDLE;
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
    if (m_grass_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(m_vk->getDevice(), m_grass_pipeline, NULL);
    if (m_skin_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(m_vk->getDevice(), m_skin_pipeline, NULL);
    if (m_skin_pipeline_layout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(m_vk->getDevice(), m_skin_pipeline_layout,
            NULL);
    }
    if (m_skin_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(m_vk->getDevice(), m_skin_pool, NULL);
    if (m_skin_layout != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(m_vk->getDevice(), m_skin_layout, NULL);
    }
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
    if (m_top_levels.empty() && m_mesh_blas.empty() && !m_skin_set &&
        m_grass_blas.empty() &&
        m_retired_skin_sets.empty() &&
        m_blas == VK_NULL_HANDLE &&
        m_data_buffer == VK_NULL_HANDLE && !m_geometry_data &&
        m_geometry_sources.empty() && m_texture_ids.empty())
        return;
    m_vk->waitIdle();
    VmaAllocator allocator = m_vk->getVmaAllocator();
    // The device is idle, so the BLAS of the meshes can go too
    while (!m_grass_blas.empty())
        destroyGrassBLAS(m_grass_blas.begin()->second.get());
    m_pending_blas.clear();
    if (m_skin_set)
        destroySkinSet(m_skin_set.get());
    m_skin_set.reset();
    for (auto& set : m_retired_skin_sets)
        destroySkinSet(set.get());
    m_retired_skin_sets.clear();
    m_skin_failed_buffers.clear();
    // After the sets, which free their output vertices
    if (m_skin_block != VK_NULL_HANDLE)
        vmaDestroyVirtualBlock(m_skin_block);
    m_skin_block = VK_NULL_HANDLE;
    m_skin_region_offset = 0;
    m_skinning_data.reset();
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
                                         VmaAllocation& memory,
                                         VkDeviceSize buffer_size)
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
    bool ok = m_vk->createBuffer(buffer_size > size ? buffer_size : size,
        usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, local_info, buffer, memory);
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
        VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
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
    // GERayTracingConfig::m_dynamic_objects)
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
        geometry.flags = getGeometryFlags(info.m_shading_type);
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
    std::vector<GrassBLAS*> unused_grass;
    for (auto& p : m_grass_blas)
    {
        if (m_frame_counter - p.second->m_last_used_frame > grace)
            unused_grass.push_back(p.second.get());
    }
    for (GrassBLAS* blas : unused_grass)
        destroyGrassBLAS(blas);
    // The skinned BLAS, a replaced one is in the frames in flight too
    if (m_skin_set && m_frame_counter - m_skin_set->m_last_used_frame > grace)
    {
        destroySkinSet(m_skin_set.get());
        m_skin_set.reset();
    }
    for (auto it = m_retired_skin_sets.begin();
        it != m_retired_skin_sets.end();)
    {
        if (m_frame_counter - (*it)->m_last_used_frame > grace)
        {
            destroySkinSet(it->get());
            it = m_retired_skin_sets.erase(it);
        }
        else
            it++;
    }
}   // destroyUnusedMeshBLAS

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::createGrassPipeline()
{
    if (m_grass_pipeline != VK_NULL_HANDLE)
        return true;
    if (m_grass_pipeline_failed)
        return false;
    m_grass_pipeline_failed = true;
    // The descriptor sets and the pipeline layout of the skin pipeline are
    // used (the mesh cache and the data buffer are the same bindings, binding
    // 0 is not read by grass_rt.comp but the set has to be valid, and its
    // push constants are bigger than the ones of grass_rt.comp)
    if (!createSkinPipeline() || !createSkinningData())
        return false;
    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module =
        GEVulkanShaderManager::getShader("grass_rt.comp");
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_skin_pipeline_layout;
    if (pipeline_info.stage.module == VK_NULL_HANDLE ||
        vkCreateComputePipelines(m_vk->getDevice(), m_vk->getPipelineCache(),
        1, &pipeline_info, NULL, &m_grass_pipeline) != VK_SUCCESS)
    {
        m_grass_pipeline = VK_NULL_HANDLE;
        return false;
    }
    m_grass_pipeline_failed = false;
    return true;
}   // createGrassPipeline

// ----------------------------------------------------------------------------
GEVulkanRTScene::GrassBLAS* GEVulkanRTScene::getGrassBLAS(
                            irr::scene::IMesh* mesh, uint64_t buffer_mask,
                            const std::vector<unsigned>& buffer_indices,
                            GEVulkanMeshCache* mesh_cache)
{
    MeshKey key = { mesh, buffer_mask };
    auto it = m_grass_blas.find(key);
    if (it != m_grass_blas.end())
        return it->second.get();

    // One which can't be created is also kept (without m_blas), so it's not
    // tried every frame, it's destroyed when it's not used anymore
    std::unique_ptr<GrassBLAS> blas(new GrassBLAS());
    GrassBLAS* ret = blas.get();
    ret->m_id = ++m_grass_next_id;
    ret->m_last_used_frame = m_frame_counter;
    m_grass_blas[key] = std::move(blas);
    if (!createGrassPipeline() || m_skin_block == VK_NULL_HANDLE)
        return ret;

    // Only the positions (3 floats without padding) are new, the rest of the
    // vertex and the indices are used from the mesh cache
    const size_t position_pitch = sizeof(float) * 3;
    VkDeviceSize out_bytes = 0;
    for (unsigned buffer_index : buffer_indices)
    {
        out_bytes += (VkDeviceSize)mesh->getMeshBuffer(buffer_index)->
            getVertexCount() * position_pitch;
    }
    VmaVirtualAllocationCreateInfo alloc_info = {};
    alloc_info.size = out_bytes;
    alloc_info.alignment = sizeof(GERTVertex);
    VkDeviceSize offset = 0;
    if (vmaVirtualAllocate(m_skin_block, &alloc_info, &ret->m_allocation,
        &offset) != VK_SUCCESS)
    {
        ret->m_allocation = VK_NULL_HANDLE;
        GE::logMessage(GLL_WARNING, "GEVulkanRTScene: no space for the "
            "grass meshes (SKIN_DATA_BYTES)");
        return ret;
    }
    // From the start of the data buffer
    const VkDeviceSize base_offset = m_skin_region_offset + offset;
    const VkDeviceAddress data_address = getAddress(m_data_buffer);
    const VkDeviceAddress cache_address = getAddress(mesh_cache->getBuffer());
    VkDeviceSize out_prefix = 0;
    for (unsigned g = 0; g < buffer_indices.size(); g++)
    {
        GESPMBuffer* buffer = static_cast<GESPMBuffer*>(
            mesh->getMeshBuffer(buffer_indices[g]));
        const uint32_t vertex_count = buffer->getVertexCount();
        const uint32_t index_count = buffer->getIndexCount() / 3 * 3;
        GERTGeometry info = {};
        info.m_vertex_offset = (uint32_t)buffer->getVBOOffset();
        info.m_index_offset = (uint32_t)(mesh_cache->getIBOOffset() /
            sizeof(uint16_t) + buffer->getIBOOffset());
        info.m_shading_type = GRTST_GRASS;
        info.m_flags = GRTGF_CACHE_VERTEX | GRTGF_CACHE_INDEX;
        ret->m_infos.push_back(info);
        ret->m_source_words.push_back((uint32_t)(buffer->getVBOOffset() *
            sizeof(GERTVertex) / sizeof(uint32_t)));
        ret->m_dst_words.push_back((uint32_t)((base_offset + out_prefix) /
            sizeof(uint32_t)));
        ret->m_vertex_counts.push_back(vertex_count);

        VkAccelerationStructureGeometryKHR geometry = {};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = getGeometryFlags(info.m_shading_type);
        geometry.geometry.triangles.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress = data_address +
            base_offset + out_prefix;
        geometry.geometry.triangles.vertexStride = position_pitch;
        geometry.geometry.triangles.maxVertex = vertex_count - 1;
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT16;
        geometry.geometry.triangles.indexData.deviceAddress = cache_address +
            (VkDeviceAddress)info.m_index_offset * sizeof(uint16_t);
        ret->m_as_geometries.push_back(geometry);
        VkAccelerationStructureBuildRangeInfoKHR range = {};
        range.primitiveCount = index_count / 3;
        ret->m_as_ranges.push_back(range);
        out_prefix += (VkDeviceSize)vertex_count * position_pitch;
    }

    // Updated every frame, so the build is made for it
    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
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
        GE::logMessage(GLL_ERROR,
            "GEVulkanRTScene: creating the BLAS of a grass mesh failed");
        // Not built, so it must not be used
        if (ret->m_blas != VK_NULL_HANDLE)
        {
            vkDestroyAccelerationStructureKHR(m_vk->getDevice(), ret->m_blas,
                NULL);
        }
        ret->m_blas = VK_NULL_HANDLE;
        return ret;
    }
    ret->m_scratch_size = std::max(sizes.buildScratchSize,
        sizes.updateScratchSize);
    VkAccelerationStructureDeviceAddressInfoKHR address_info = {};
    address_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = ret->m_blas;
    ret->m_address = vkGetAccelerationStructureDeviceAddressKHR(
        m_vk->getDevice(), &address_info);
    return ret;
}   // getGrassBLAS

// ----------------------------------------------------------------------------
void GEVulkanRTScene::destroyGrassBLAS(GrassBLAS* blas)
{
    for (auto it = m_grass_blas.begin(); it != m_grass_blas.end(); it++)
    {
        if (it->second.get() != blas)
            continue;
        if (blas->m_blas != VK_NULL_HANDLE)
        {
            vkDestroyAccelerationStructureKHR(m_vk->getDevice(), blas->m_blas,
                NULL);
        }
        if (blas->m_blas_buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(m_vk->getVmaAllocator(), blas->m_blas_buffer,
                blas->m_blas_memory);
        }
        if (blas->m_allocation != VK_NULL_HANDLE && m_skin_block)
            vmaVirtualFree(m_skin_block, blas->m_allocation);
        m_grass_blas.erase(it);
        return;
    }
}   // destroyGrassBLAS

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::createSkinPipeline()
{
    if (m_skin_pipeline != VK_NULL_HANDLE)
        return true;
    if (m_skin_pipeline_failed)
        return false;
    m_skin_pipeline_failed = true;
    VkDevice device = m_vk->getDevice();
    // The joint matrices, the mesh cache and the data buffer (output)
    std::array<VkDescriptorSetLayoutBinding, 3> bindings = {};
    for (unsigned i = 0; i < bindings.size(); i++)
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
    if (vkCreateDescriptorSetLayout(device, &layout_info, NULL,
        &m_skin_layout) != VK_SUCCESS)
        return false;

    const unsigned set_count = GEVulkanDriver::getMaxFrameInFlight() + 1;
    VkDescriptorPoolSize size = {};
    size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    size.descriptorCount = set_count * 3;
    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &size;
    pool_info.maxSets = set_count;
    if (vkCreateDescriptorPool(device, &pool_info, NULL, &m_skin_pool) !=
        VK_SUCCESS)
        return false;
    std::vector<VkDescriptorSetLayout> layouts(set_count, m_skin_layout);
    m_skin_descriptor_sets.resize(set_count, VK_NULL_HANDLE);
    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_skin_pool;
    alloc_info.descriptorSetCount = set_count;
    alloc_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(device, &alloc_info,
        m_skin_descriptor_sets.data()) != VK_SUCCESS)
        return false;

    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    // mat4 and 5 words (see skin_rt.comp)
    push_constant.size = sizeof(float) * 16 + sizeof(uint32_t) * 5;
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &m_skin_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
        &m_skin_pipeline_layout) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module = GEVulkanShaderManager::getShader("skin_rt.comp");
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_skin_pipeline_layout;
    if (pipeline_info.stage.module == VK_NULL_HANDLE ||
        vkCreateComputePipelines(device, m_vk->getPipelineCache(), 1,
        &pipeline_info, NULL, &m_skin_pipeline) != VK_SUCCESS)
    {
        m_skin_pipeline = VK_NULL_HANDLE;
        return false;
    }
    m_skin_pipeline_failed = false;
    return true;
}   // createSkinPipeline

// ----------------------------------------------------------------------------
bool GEVulkanRTScene::createSkinningData()
{
    if (m_skinning_data)
        return true;
    const size_t size = MAX_SKINNING_MATRICES * sizeof(irr::core::matrix4);
    std::unique_ptr<GEVulkanDynamicBuffer> data(new GEVulkanDynamicBuffer(
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, size,
        GEVulkanDriver::getMaxFrameInFlight() + 1, 0));
    irr::core::matrix4 identity;
    for (unsigned i = 0; i < data->getHostBuffer().size(); i++)
    {
        if (data->getHostBuffer()[i] == VK_NULL_HANDLE ||
            data->getMappedAddr()[i] == NULL)
            return false;
        // Only the first matrix is never written again
        memcpy(data->getMappedAddr()[i], identity.pointer(), sizeof(identity));
        vmaFlushAllocation(m_vk->getVmaAllocator(), data->getHostMemory()[i],
            0, sizeof(identity));
        VkDescriptorBufferInfo info =
            { data->getHostBuffer()[i], 0, VK_WHOLE_SIZE };
        VkWriteDescriptorSet write = {};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = m_skin_descriptor_sets[i];
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &info;
        vkUpdateDescriptorSets(m_vk->getDevice(), 1, &write, 0, NULL);
    }
    m_skinning_data = std::move(data);
    return true;
}   // createSkinningData

// ----------------------------------------------------------------------------
std::unique_ptr<GEVulkanRTScene::SkinSet> GEVulkanRTScene::createSkinSet(
                                const std::vector<GESPMBuffer*>& buffers,
                                const std::vector<uint32_t>& shading_types,
                                GEVulkanMeshCache* mesh_cache)
{
    std::unique_ptr<SkinSet> ret;
    if (!createSkinPipeline() || !createSkinningData() ||
        m_skin_block == VK_NULL_HANDLE)
        return ret;

    std::unique_ptr<SkinSet> set(new SkinSet());
    set->m_buffers = buffers;
    set->m_last_used_frame = m_frame_counter;

    // Only the output vertices are new, the bind pose and the indices are used
    // from the mesh cache
    VkDeviceSize out_bytes = 0;
    for (GESPMBuffer* b : buffers)
        out_bytes += (VkDeviceSize)b->getVertexCount() * sizeof(GERTVertex);
    VmaVirtualAllocationCreateInfo alloc_info = {};
    alloc_info.size = out_bytes;
    alloc_info.alignment = sizeof(GERTVertex);
    VkDeviceSize offset = 0;
    if (vmaVirtualAllocate(m_skin_block, &alloc_info, &set->m_allocation,
        &offset) != VK_SUCCESS)
    {
        set->m_allocation = VK_NULL_HANDLE;
        GE::logMessage(GLL_WARNING, "GEVulkanRTScene: no space for the "
            "skinned meshes (SKIN_DATA_BYTES)");
        destroySkinSet(set.get());
        return ret;
    }
    // From the start of the data buffer
    const VkDeviceSize base_offset = m_skin_region_offset + offset;
    const VkDeviceAddress data_address = getAddress(m_data_buffer);
    const VkDeviceAddress cache_address = getAddress(mesh_cache->getBuffer());
    const size_t bone_pitch = sizeof(int16_t) * 8;
    VkDeviceSize out_prefix = 0;
    for (unsigned g = 0; g < buffers.size(); g++)
    {
        GESPMBuffer* b = buffers[g];
        const uint32_t vertex_count = b->getVertexCount();
        const uint32_t index_count = b->getIndexCount() / 3 * 3;

        GERTGeometry info = {};
        info.m_vertex_offset = (uint32_t)((base_offset + out_prefix) /
            sizeof(GERTVertex));
        info.m_index_offset = (uint32_t)(mesh_cache->getIBOOffset() /
            sizeof(uint16_t) + b->getIBOOffset());
        info.m_shading_type = shading_types[g];
        info.m_flags = GRTGF_CACHE_INDEX;
        set->m_infos.push_back(info);
        // The static part of a vertex is 8 words in the mesh cache, the joints
        // and weights (4 words) are after the indices, indexed by the vertex
        // index of the mesh buffer like the vertex binding (see
        // GEVulkanMeshCache::updateCache)
        set->m_source_words.push_back((uint32_t)(b->getVBOOffset() *
            sizeof(GERTVertex) / sizeof(uint32_t)));
        set->m_skin_words.push_back((uint32_t)((mesh_cache->
            getSkinningVBOOffset() + b->getVBOOffset() * bone_pitch) /
            sizeof(uint32_t)));
        set->m_dst_words.push_back((uint32_t)((base_offset + out_prefix) /
            sizeof(uint32_t)));
        set->m_vertex_counts.push_back(vertex_count);

        VkAccelerationStructureGeometryKHR geometry = {};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.flags = getGeometryFlags(info.m_shading_type);
        geometry.geometry.triangles.sType =
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometry.geometry.triangles.vertexData.deviceAddress = data_address +
            (VkDeviceAddress)info.m_vertex_offset * sizeof(GERTVertex);
        geometry.geometry.triangles.vertexStride = sizeof(GERTVertex);
        geometry.geometry.triangles.maxVertex = vertex_count - 1;
        geometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT16;
        geometry.geometry.triangles.indexData.deviceAddress = cache_address +
            (VkDeviceAddress)info.m_index_offset * sizeof(uint16_t);
        set->m_as_geometries.push_back(geometry);
        VkAccelerationStructureBuildRangeInfoKHR range = {};
        range.primitiveCount = index_count / 3;
        set->m_as_ranges.push_back(range);
        out_prefix += (VkDeviceSize)vertex_count * sizeof(GERTVertex);
    }

    // Updated every frame, so the build is made for it
    VkAccelerationStructureBuildGeometryInfoKHR build_info = {};
    build_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    build_info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    build_info.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    build_info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    build_info.geometryCount = (uint32_t)set->m_as_geometries.size();
    build_info.pGeometries = set->m_as_geometries.data();
    std::vector<uint32_t> primitive_counts;
    for (const auto& range : set->m_as_ranges)
        primitive_counts.push_back(range.primitiveCount);
    VkAccelerationStructureBuildSizesInfoKHR sizes = {};
    sizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(m_vk->getDevice(),
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &build_info,
        primitive_counts.data(), &sizes);
    if (!createAccelerationStructure(
        VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        sizes.accelerationStructureSize, set->m_blas_buffer,
        set->m_blas_memory, set->m_blas))
    {
        GE::logMessage(GLL_ERROR,
            "GEVulkanRTScene: creating the skinned BLAS failed");
        if (set->m_blas != VK_NULL_HANDLE)
        {
            vkDestroyAccelerationStructureKHR(m_vk->getDevice(), set->m_blas,
                NULL);
        }
        set->m_blas = VK_NULL_HANDLE;
        destroySkinSet(set.get());
        return ret;
    }
    set->m_scratch_size = std::max(sizes.buildScratchSize,
        sizes.updateScratchSize);
    VkAccelerationStructureDeviceAddressInfoKHR address_info = {};
    address_info.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    address_info.accelerationStructure = set->m_blas;
    set->m_address = vkGetAccelerationStructureDeviceAddressKHR(
        m_vk->getDevice(), &address_info);
    return set;
}   // createSkinSet

// ----------------------------------------------------------------------------
// The device must not use it
void GEVulkanRTScene::destroySkinSet(SkinSet* set)
{
    if (set->m_blas != VK_NULL_HANDLE)
        vkDestroyAccelerationStructureKHR(m_vk->getDevice(), set->m_blas, NULL);
    if (set->m_blas_buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_vk->getVmaAllocator(), set->m_blas_buffer,
            set->m_blas_memory);
    }
    if (set->m_allocation != VK_NULL_HANDLE && m_skin_block)
        vmaVirtualFree(m_skin_block, set->m_allocation);
    set->m_buffers.clear();
    set->m_blas = VK_NULL_HANDLE;
    set->m_blas_buffer = VK_NULL_HANDLE;
    set->m_allocation = VK_NULL_HANDLE;
}   // destroySkinSet

// ----------------------------------------------------------------------------
void GEVulkanRTScene::update(irr::scene::ISceneNode* root,
                         const std::vector<irr::core::vector3df>& cameras)
{
    if (!isReady() || !m_geometry_data)
        return;
    m_frame_counter++;
    // The buffers of this frame are not used by the gpu (frames in flight + 1
    // of them), and the submit makes the host write visible. The mapped memory
    // is only written, never read
    const unsigned buffer_idx = m_vk->getCurrentBufferIdx();
    TopLevel& t = m_top_levels[buffer_idx];
    t.m_grass_updates.clear();
    GERTGeometry* mapped_addr =
        (GERTGeometry*)m_geometry_data->getMappedAddr()[buffer_idx];
    VmaAllocator allocator = m_vk->getVmaAllocator();
    const size_t static_count = m_geometry_infos.size();

    GEVulkanMeshCache* mesh_cache = m_vk->getVulkanMeshCache();
    const uint64_t cache_version =
        mesh_cache ? mesh_cache->getUpdateCount() : 0;
    if (cache_version != m_cache_version)
    {
        m_warned_dynamic_limit = false;
        // The buffer and the offsets of all meshes are new (and a mesh could
        // be gone, so a mesh address can be used by another one). The mesh
        // cache waited for the device before it destroyed its buffer, and the
        // frames of the TLAS with them are done
        m_pending_blas.clear();
        while (!m_mesh_blas.empty())
            destroyMeshBLAS(m_mesh_blas.begin()->second.get());
        while (!m_grass_blas.empty())
            destroyGrassBLAS(m_grass_blas.begin()->second.get());
        if (m_skin_set)
            destroySkinSet(m_skin_set.get());
        m_skin_set.reset();
        for (auto& set : m_retired_skin_sets)
            destroySkinSet(set.get());
        m_retired_skin_sets.clear();
        m_skin_failed_buffers.clear();
        m_cache_version = cache_version;
    }
    if (t.m_cache_version != cache_version)
        writeMeshCacheDescriptor(buffer_idx, mesh_cache);
    // The BLAS of the dynamic nodes are made of the mesh cache
    const bool dynamic = getGEConfig()->m_ray_tracing.m_dynamic_objects &&
        GEVulkanFeatures::supportsBaseVertexRendering() && mesh_cache &&
        mesh_cache->getBuffer() != VK_NULL_HANDLE;

    if (!dynamic)
    {
        t.m_skin_active = false;
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
        // The static BLAS only (the grass is found with the dynamic nodes)
        VkAccelerationStructureInstanceKHR* static_instances =
            (VkAccelerationStructureInstanceKHR*)t.m_instance_mapped;
        if (!t.m_instance_static)
        {
            static_instances[0] = getStaticInstance();
            vmaFlushAllocation(allocator, t.m_instance_memory, 0,
                sizeof(VkAccelerationStructureInstanceKHR));
            t.m_dirty = true;
            t.m_slots.clear();
        }
        t.m_instance_count = 1;
        t.m_instance_static = true;
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
    std::vector<DynamicSource> grass_sources;
    std::vector<SkinSource> skins;
    if (root)
    {
        collectDynamicNode(root, sources, grass_sources, skins,
            root->isTrulyVisible());
    }
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
            VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = blas->m_address;
        instances[instance_count++] = instance;

        for (unsigned g = 0; g < blas->m_infos.size(); g++)
        {
            GERTGeometry info = blas->m_infos[g];
            setDynamicMaterial(info, node, s.m_buffers[g], td, t.m_slots);
            mapped_addr[geometry_count++] = info;
        }
    }

    // The grass, every node is an instance of the BLAS of its mesh (the wind is
    // applied to its vertices in the model space once for all the nodes, see
    // grass_rt.comp). The geometries of the nodes with the same mesh and the
    // same material are shared, only a mesh which has a node is updated
    std::map<std::vector<uint32_t>, uint32_t> grass_geometries;
    for (DynamicSource& s : grass_sources)
    {
        GrassBLAS* blas = getGrassBLAS(s.m_mesh, s.m_buffer_mask, s.m_buffers,
            mesh_cache);
        if (blas->m_blas == VK_NULL_HANDLE)
            continue;
        if (instance_count >= m_instance_capacity)
        {
            if (!m_warned_dynamic_limit)
            {
                m_warned_dynamic_limit = true;
                GE::logMessage(GLL_WARNING, "GEVulkanRTScene: too many "
                    "grass instances, the other ones are not traced");
            }
            break;
        }
        irr::scene::ISceneNode* node = s.m_node;
        std::vector<GERTGeometry> infos;
        std::vector<uint32_t> geometry_key;
        geometry_key.push_back(blas->m_id);
        for (unsigned g = 0; g < blas->m_infos.size(); g++)
        {
            GERTGeometry info = blas->m_infos[g];
            setDynamicMaterial(info, node, s.m_buffers[g], td, t.m_slots);
            infos.push_back(info);
            uint32_t hue_bits, shift_x_bits, shift_y_bits;
            memcpy(&hue_bits, &info.m_hue, sizeof(uint32_t));
            memcpy(&shift_x_bits, &info.m_uv_shift[0], sizeof(uint32_t));
            memcpy(&shift_y_bits, &info.m_uv_shift[1], sizeof(uint32_t));
            geometry_key.push_back(info.m_material_id);
            geometry_key.push_back(hue_bits);
            geometry_key.push_back(shift_x_bits);
            geometry_key.push_back(shift_y_bits);
        }
        uint32_t first_geometry = 0;
        auto found = grass_geometries.find(geometry_key);
        if (found != grass_geometries.end())
            first_geometry = found->second;
        else
        {
            if (geometry_count - static_count + infos.size() >
                MAX_DYNAMIC_GEOMETRIES)
            {
                if (!m_warned_dynamic_limit)
                {
                    m_warned_dynamic_limit = true;
                    GE::logMessage(GLL_WARNING, "GEVulkanRTScene: too many "
                        "grass geometries, the other ones are not traced");
                }
                break;
            }
            first_geometry = (uint32_t)geometry_count;
            for (const GERTGeometry& info : infos)
                mapped_addr[geometry_count++] = info;
            grass_geometries[geometry_key] = first_geometry;
        }
        blas->m_last_used_frame = m_frame_counter;
        if (blas->m_listed_frame != m_frame_counter)
        {
            blas->m_listed_frame = m_frame_counter;
            t.m_grass_updates.push_back(blas);
        }

        // Row major 3x4 of the column major 4x4 of the node
        const irr::core::matrix4& abs = node->getAbsoluteTransformation();
        VkAccelerationStructureInstanceKHR instance = {};
        for (unsigned r = 0; r < 3; r++)
        {
            for (unsigned c = 0; c < 4; c++)
                instance.transform.matrix[r][c] = abs[c * 4 + r];
        }
        instance.instanceCustomIndex = first_geometry;
        instance.mask = 0xFF;
        instance.instanceShaderBindingTableRecordOffset = 0;
        instance.flags =
            VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = blas->m_address;
        instances[instance_count++] = instance;
    }

    // The skinned nodes, all in the BLAS of the current set
    t.m_skin_active = false;
    t.m_skin_dispatches.clear();
    if (!skins.empty())
    {
        std::vector<GESPMBuffer*> buffers;
        std::vector<uint32_t> shading_types;
        for (const SkinSource& s : skins)
        {
            buffers.push_back(s.m_buffer);
            shading_types.push_back(s.m_shading_type);
        }
        if ((!m_skin_set || m_skin_set->m_buffers != buffers) &&
            m_skin_failed_buffers != buffers)
        {
            std::unique_ptr<SkinSet> set =
                createSkinSet(buffers, shading_types, mesh_cache);
            if (set)
            {
                if (m_skin_set)
                {
                    // The frames in flight can use it
                    m_retired_skin_sets.push_back(std::move(m_skin_set));
                }
                m_skin_set = std::move(set);
                m_skin_failed_buffers.clear();
            }
            else
                m_skin_failed_buffers = buffers;
        }
        if (m_skin_set && m_skin_set->m_buffers == buffers &&
            instance_count < m_instance_capacity &&
            geometry_count - static_count + buffers.size() <=
            MAX_DYNAMIC_GEOMETRIES)
        {
            // The joint matrices of every node (once for its geometries), in
            // the buffer of this frame, the first one is the identity
            irr::core::matrix4* matrices = (irr::core::matrix4*)
                m_skinning_data->getMappedAddr()[buffer_idx];
            unsigned matrix_count = 1;
            bool matrices_ok = true;
            std::vector<TopLevel::SkinDispatch> dispatches;
            // The vertices are baked relative to this origin, with the
            // translation of the instance as the origin, otherwise a node far
            // from the world origin loses the precision of its 32 bit
            // vertices (the small details are distorted). All the skinned
            // nodes share a BLAS, so the origin is the position of the node
            // nearest to the first camera, which is the one that should have
            // the best precision (the order of the skins is not changed)
            irr::core::vector3df skin_origin;
            if (!skins.empty())
            {
                float nearest = -1.0f;
                for (const SkinSource& s : skins)
                {
                    const irr::core::vector3df pos =
                        s.m_node->getAbsoluteTransformation().getTranslation();
                    const float dist = cameras.empty() ? 0.0f :
                        (float)pos.getDistanceFromSQ(cameras[0]);
                    if (nearest < 0.0f || dist < nearest)
                    {
                        nearest = dist;
                        skin_origin = pos;
                    }
                }
            }
            irr::scene::ISceneNode* last_node = NULL;
            int last_offset = 0;
            for (const SkinSource& s : skins)
            {
                if (s.m_node != last_node)
                {
                    GEVulkanAnimatedMeshSceneNode* node =
                        static_cast<GEVulkanAnimatedMeshSceneNode*>(s.m_node);
                    const unsigned joint_count = static_cast<GESPM*>(
                        node->getMesh())->getJointCount();
                    if (matrix_count + joint_count > MAX_SKINNING_MATRICES)
                    {
                        matrices_ok = false;
                        break;
                    }
                    last_node = s.m_node;
                    last_offset = (int)matrix_count;
                    node->getSkinningMatrices(matrices + matrix_count);
                    matrix_count += joint_count;
                }
                TopLevel::SkinDispatch dispatch;
                memcpy(dispatch.m_world, s.m_node->getAbsoluteTransformation()
                    .pointer(), sizeof(dispatch.m_world));
                // Column major, the translation is the last column
                dispatch.m_world[12] -= skin_origin.X;
                dispatch.m_world[13] -= skin_origin.Y;
                dispatch.m_world[14] -= skin_origin.Z;
                dispatch.m_skinning_offset = last_offset;
                dispatches.push_back(dispatch);
            }
            if (matrix_count > 1)
            {
                vmaFlushAllocation(allocator,
                    m_skinning_data->getHostMemory()[buffer_idx], 0,
                    matrix_count * sizeof(irr::core::matrix4));
            }
            if (!matrices_ok)
            {
                GE::logMessage(GLL_WARNING, "GEVulkanRTScene: too many "
                    "joints (MAX_SKINNING_MATRICES), skinned nodes are not "
                    "traced");
            }
            else
            {
                m_skin_set->m_last_used_frame = m_frame_counter;
                // World space with the origin of the skin removed (see
                // skin_origin), the instance only moves it back. The shaders
                // use the rotation part (identity) for the normals
                VkAccelerationStructureInstanceKHR instance = {};
                instance.transform.matrix[0][0] = 1.0f;
                instance.transform.matrix[1][1] = 1.0f;
                instance.transform.matrix[2][2] = 1.0f;
                instance.transform.matrix[0][3] = skin_origin.X;
                instance.transform.matrix[1][3] = skin_origin.Y;
                instance.transform.matrix[2][3] = skin_origin.Z;
                instance.instanceCustomIndex = (uint32_t)geometry_count;
                instance.mask = 0xFF;
                instance.instanceShaderBindingTableRecordOffset = 0;
                instance.flags =
                    VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
                instance.accelerationStructureReference = m_skin_set->m_address;
                instances[instance_count++] = instance;
                for (unsigned g = 0; g < skins.size(); g++)
                {
                    GERTGeometry info = m_skin_set->m_infos[g];
                    setDynamicMaterial(info, skins[g].m_node,
                        skins[g].m_material_index, td, t.m_slots);
                    mapped_addr[geometry_count++] = info;
                }
                t.m_skin_dispatches = dispatches;
                // Updated every frame, so the TLAS is built every frame too
                t.m_skin_active = true;
            }
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
    SkinSet* skin_set = t.m_skin_active && m_skin_set &&
        m_skin_pipeline != VK_NULL_HANDLE ? m_skin_set.get() : NULL;
    const bool grass = !t.m_grass_updates.empty() &&
        m_grass_pipeline != VK_NULL_HANDLE;

    const VkDeviceSize alignment =
        GEVulkanFeatures::getAccelerationStructureScratchAlignment();
    auto align = [alignment](VkDeviceSize size)
        { return (size + alignment - 1) / alignment * alignment; };
    // The TLAS first, then the BLAS of the new meshes, then the skinned one
    VkDeviceSize needed = align(m_tlas_scratch_size);
    for (MeshBLAS* blas : m_pending_blas)
        needed += align(blas->m_build_scratch_size);
    if (skin_set)
        needed += align(skin_set->m_scratch_size);
    if (grass)
    {
        for (GrassBLAS* blas : t.m_grass_updates)
            needed += align(blas->m_scratch_size);
    }
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
    if (skin_set || grass)
    {
        // Skin the bind pose to the output vertices of the BLAS. The queue
        // executes the previous frame before this one, but the ray queries
        // of its fragment shaders (using the BLAS and the vertices) can be
        // still running
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT |
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        barrier.dstAccessMask = barrier.srcAccessMask;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
            &barrier, 0, NULL, 0, NULL);

        // Binding 0 (the joint matrices of this index) never changes
        VkDescriptorSet set =
            m_skin_descriptor_sets[m_vk->getCurrentBufferIdx()];
        VkDescriptorBufferInfo cache_info = { getMeshCacheDescriptorBuffer(
            m_vk->getVulkanMeshCache()), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo data_info = { m_data_buffer, 0, VK_WHOLE_SIZE };
        std::array<VkWriteDescriptorSet, 2> writes = {};
        for (unsigned i = 0; i < writes.size(); i++)
        {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[i].dstSet = set;
            writes[i].dstBinding = i + 1;
            writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        }
        writes[0].pBufferInfo = &cache_info;
        writes[1].pBufferInfo = &data_info;
        // This set is not used by the previous frames in flight
        vkUpdateDescriptorSets(m_vk->getDevice(), writes.size(),
            writes.data(), 0, NULL);
        if (skin_set)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_skin_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                m_skin_pipeline_layout, 0, 1, &set, 0, NULL);
            for (unsigned g = 0; g < t.m_skin_dispatches.size(); g++)
            {
                struct
                {
                    float m_world[16];
                    uint32_t m_source_word;
                    uint32_t m_skin_word;
                    uint32_t m_dst_word;
                    uint32_t m_vertex_count;
                    int32_t m_skinning_offset;
                } pc;
                static_assert(sizeof(pc) == sizeof(float) * 16 + 5 * 4,
                    "same as the push constants of skin_rt.comp");
                memcpy(pc.m_world, t.m_skin_dispatches[g].m_world,
                    sizeof(pc.m_world));
                pc.m_source_word = skin_set->m_source_words[g];
                pc.m_skin_word = skin_set->m_skin_words[g];
                pc.m_dst_word = skin_set->m_dst_words[g];
                pc.m_vertex_count = skin_set->m_vertex_counts[g];
                pc.m_skinning_offset = t.m_skin_dispatches[g].m_skinning_offset;
                vkCmdPushConstants(cmd, m_skin_pipeline_layout,
                    VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                vkCmdDispatch(cmd, (pc.m_vertex_count + 63) / 64, 1, 1);
            }
        }
        if (grass)
        {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                m_grass_pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                m_skin_pipeline_layout, 0, 1, &set, 0, NULL);
            float wind[3];
            GEMaterialManager::getWindDirection(wind);
            for (GrassBLAS* blas : t.m_grass_updates)
            {
                for (unsigned g = 0; g < blas->m_vertex_counts.size(); g++)
                {
                    GrassPushConstants pc;
                    memcpy(pc.m_wind, wind, sizeof(wind));
                    pc.m_source_word = blas->m_source_words[g];
                    pc.m_dst_word = blas->m_dst_words[g];
                    pc.m_vertex_count = blas->m_vertex_counts[g];
                    vkCmdPushConstants(cmd, m_skin_pipeline_layout,
                        VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                    vkCmdDispatch(cmd, (pc.m_vertex_count + 63) / 64, 1, 1);
                }
            }
        }
        // The builds read the output vertices
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
            VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1,
            &barrier, 0, NULL, 0, NULL);
    }

    if (!m_pending_blas.empty() || skin_set || grass)
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
        m_pending_blas.clear();
        if (skin_set)
        {
            // The first time it's built, then it's updated (refit), with
            // the same flags
            VkAccelerationStructureBuildGeometryInfoKHR info = {};
            info.sType =
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
            info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            info.flags = VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
                VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
            info.mode = skin_set->m_built ?
                VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR :
                VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
            info.srcAccelerationStructure =
                skin_set->m_built ? skin_set->m_blas : VK_NULL_HANDLE;
            info.dstAccelerationStructure = skin_set->m_blas;
            info.geometryCount = (uint32_t)skin_set->m_as_geometries.size();
            info.pGeometries = skin_set->m_as_geometries.data();
            info.scratchData.deviceAddress = scratch;
            infos.push_back(info);
            ranges.push_back(skin_set->m_as_ranges.data());
            skin_set->m_built = true;
            scratch += align(skin_set->m_scratch_size);
        }
        if (grass)
        {
            // The first time they are built, then updated (refit), with the
            // same flags
            for (GrassBLAS* blas : t.m_grass_updates)
            {
                VkAccelerationStructureBuildGeometryInfoKHR info = {};
                info.sType =
                    VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
                info.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
                info.flags =
                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
                    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
                info.mode = blas->m_built ?
                    VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR :
                    VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
                info.srcAccelerationStructure =
                    blas->m_built ? blas->m_blas : VK_NULL_HANDLE;
                info.dstAccelerationStructure = blas->m_blas;
                info.geometryCount = (uint32_t)blas->m_as_geometries.size();
                info.pGeometries = blas->m_as_geometries.data();
                info.scratchData.deviceAddress = scratch;
                scratch += align(blas->m_scratch_size);
                infos.push_back(info);
                ranges.push_back(blas->m_as_ranges.data());
                blas->m_built = true;
            }
        }
        vkCmdBuildAccelerationStructuresKHR(cmd, (uint32_t)infos.size(),
            infos.data(), ranges.data());
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

    // The ray queries of the fragment shaders, which read the skinned
    // vertices (the output of the compute shader) too
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR |
        VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR |
        VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
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

    // The shading types the geometries are collected with
    m_built_alphatest = getGEConfig()->m_ray_tracing.m_alphatest;
    std::vector<RTSource> sources;
    collectNode(root, sources, root->isTrulyVisible());
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
    const size_t static_count = sources.size();

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
            // The tangent is a direction of the surface, so it's not
            // transformed by the normal matrix, the sign of the bitangent
            // (2 bits at the top) is the one of the vertex
            irr::core::vector3df tangent =
                MiniGLM::decompressVector3(src[i].m_tangent);
            abs.rotateVect(tangent);
            tangent.normalize();
            GERTVertex v;
            v.m_position[0] = pos.X;
            v.m_position[1] = pos.Y;
            v.m_position[2] = pos.Z;
            v.m_normal = packSnorm1010102(normal.X, normal.Y, normal.Z, 0);
            v.m_color = src[i].m_color.color;
            v.m_uv = (uint32_t)(uint16_t)src[i].m_all_uvs[0] |
                ((uint32_t)(uint16_t)src[i].m_all_uvs[1] << 16);
            v.m_uv_two = (uint32_t)(uint16_t)src[i].m_all_uvs[2] |
                ((uint32_t)(uint16_t)src[i].m_all_uvs[3] << 16);
            v.m_tangent = packSnorm1010102(tangent.X, tangent.Y, tangent.Z,
                src[i].m_tangent >> 30);
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

    // The output vertices of the skinned nodes are after the static data,
    // aligned to the vertex (offsets of GERTGeometry are in units of
    // GERTVertex), only if the mesh cache is used
    VkDeviceSize data_buffer_size = 0;
    VkDeviceSize region_end = (data.size() + sizeof(GERTVertex) - 1) /
        sizeof(GERTVertex) * sizeof(GERTVertex);
    if (GEVulkanFeatures::supportsBaseVertexRendering())
    {
        m_skin_region_offset = region_end;
        region_end += SKIN_DATA_BYTES;
        data_buffer_size = region_end;
        VmaVirtualBlockCreateInfo block_info = {};
        block_info.size = SKIN_DATA_BYTES;
        if (vmaCreateVirtualBlock(&block_info, &m_skin_block) != VK_SUCCESS)
            m_skin_block = VK_NULL_HANDLE;
    }
    if (!createDeviceBuffer(data.data(), data.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        m_data_buffer, m_data_memory, data_buffer_size) ||
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
    geometries.reserve(static_count);
    ranges.reserve(static_count);
    uint32_t triangles = 0;
    for (unsigned i = 0; i < static_count; i++)
    {
        VkAccelerationStructureGeometryKHR g = {};
        g.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        g.flags = getGeometryFlags(infos[i].m_shading_type);
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
    m_geometry_count = (uint32_t)static_count;
    m_triangle_count = triangles;
    updateDescriptorSet();
    GE::logMessage(GLL_INFO, "GEVulkanRTScene: built %d geometries",
        m_geometry_count);
    return true;
}   // build

}
