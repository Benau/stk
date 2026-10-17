#ifndef HEADER_GE_VULKAN_RT_SCENE_HPP
#define HEADER_GE_VULKAN_RT_SCENE_HPP

#include "ge_vma.hpp"
#include "vulkan_wrapper.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace irr
{
    namespace scene { class ISceneNode; }
}

namespace GE
{
class GEVulkanDriver;

// Shading of a ray hit, shared with RT_SHADING_* in the ray traced shaders
enum GERTShadingType : uint32_t
{
    GRTST_SOLID = 0,
};

// Vertex of the ray traced scene (world space, 32 bytes), the position is
// read by the acceleration structure build so it must be the first 3 floats
struct GERTVertex
{
    float m_position[3];
    // SColor of the vertex (ARGB)
    uint32_t m_color;
    float m_normal[3];
    // 2 half floats, same layout as packHalf2x16
    uint32_t m_uv;
};

// Per geometry data of the acceleration structure, indexed by the geometry
// index of a ray hit (std430, 32 bytes), offsets are in units of
// GERTVertex and uint32_t of the data buffer
struct GERTGeometry
{
    uint32_t m_vertex_offset;
    uint32_t m_index_offset;
    // Slot of the mesh texture descriptor, layers are at
    // m_material_id * TOTAL_MESH_TEXTURE_LAYER
    uint32_t m_material_id;
    // GERTShadingType
    uint32_t m_shading_type;
    float m_uv_shift[2];
    float m_hue;
    // SColor of the custom vertex color of the render info
    uint32_t m_vertex_color;
};

// All the static nodes (those with GERTT_STATIC in the render info of the
// materials) are joined into a single bottom level acceleration structure of
// world space triangles, which is shared by all cameras. The scene should be
// built after the nodes are added and cleared before the scene is cleared
class GEVulkanRTScene
{
private:
    GEVulkanDriver* m_vk;

    // Vertices (GERTVertex) followed by indices (uint32_t) of all geometries
    VkBuffer m_data_buffer;

    VmaAllocation m_data_memory;

    // GERTGeometry array
    VkBuffer m_geometry_buffer;

    VmaAllocation m_geometry_memory;

    // Set 0 of the ray traced pass: the TLAS, the data buffer and the geometry
    // buffer (see solid_rt.frag), updated after the scene is built
    VkDescriptorSetLayout m_descriptor_layout;

    VkDescriptorPool m_descriptor_pool;

    VkDescriptorSet m_descriptor_set;

    VkBuffer m_blas_buffer;

    VmaAllocation m_blas_memory;

    VkAccelerationStructureKHR m_blas;

    VkBuffer m_tlas_buffer;

    VmaAllocation m_tlas_memory;

    VkAccelerationStructureKHR m_tlas;

    // Keep the slots of the mesh texture descriptor alive
    std::vector<std::shared_ptr<int> > m_texture_ids;

    uint32_t m_geometry_count;

    uint32_t m_triangle_count;

    // ------------------------------------------------------------------------
    bool createDeviceBuffer(const void* data, VkDeviceSize size,
                            VkBufferUsageFlags usage, VkBuffer& buffer,
                            VmaAllocation& memory);
    // ------------------------------------------------------------------------
    VkDeviceAddress getAddress(VkBuffer buffer) const;
    // ------------------------------------------------------------------------
    bool createAccelerationStructure(VkAccelerationStructureTypeKHR type,
        VkDeviceSize size, VkBuffer& buffer, VmaAllocation& memory,
        VkAccelerationStructureKHR& as);
    // ------------------------------------------------------------------------
    bool createScratchBuffer(VkDeviceSize size, VkBuffer& buffer,
        VmaAllocation& memory, VkDeviceAddress& aligned_address);
    // ------------------------------------------------------------------------
    void updateDescriptorSet();
    // ------------------------------------------------------------------------
    bool buildAccelerationStructures(
        const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
        const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& ranges);
public:
    // ------------------------------------------------------------------------
    GEVulkanRTScene(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    ~GEVulkanRTScene();
    // ------------------------------------------------------------------------
    // Collect the static nodes under root and build the scene, it waits for
    // the device so call it when loading, returns true if something was built
    bool build(irr::scene::ISceneNode* root);
    // ------------------------------------------------------------------------
    // Wait for the device and destroy everything
    void clear();
    // ------------------------------------------------------------------------
    bool isReady() const                   { return m_tlas != VK_NULL_HANDLE; }
    // ------------------------------------------------------------------------
    VkAccelerationStructureKHR getTLAS() const               { return m_tlas; }
    // ------------------------------------------------------------------------
    VkDescriptorSetLayout getDescriptorSetLayout() const
                                                { return m_descriptor_layout; }
    // ------------------------------------------------------------------------
    // Valid to bind if isReady()
    const VkDescriptorSet* getDescriptorSet() const
                                                  { return &m_descriptor_set; }
    // ------------------------------------------------------------------------
    VkBuffer getDataBuffer() const                    { return m_data_buffer; }
    // ------------------------------------------------------------------------
    VkBuffer getGeometryBuffer() const            { return m_geometry_buffer; }
    // ------------------------------------------------------------------------
    uint32_t getGeometryCount() const              { return m_geometry_count; }
    // ------------------------------------------------------------------------
    uint32_t getTriangleCount() const              { return m_triangle_count; }
};   // GEVulkanRTScene

}

#endif
