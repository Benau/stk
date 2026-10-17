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
class GEVulkanDynamicBuffer;

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
// GERTVertex and uint32_t of the data buffer. Only m_uv_shift, m_hue and
// m_vertex_color can change after the build (GEVulkanRTScene::update), the
// rest is baked with the acceleration structure
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

    // Where the (mutable) GERTGeometry of a geometry comes from, the order is
    // the one of the geometries of the BLAS
    struct GeometrySource
    {
        // Grabbed, so it's valid until clear()
        irr::scene::ISceneNode* m_node;
        unsigned m_material_index;
    };

    // CPU copy of the GERTGeometry array as it was built (the baked fields
    // are final, the others are the initial ones), update() copies it and
    // writes the current values of the others directly into the host buffer
    std::vector<GERTGeometry> m_geometry_infos;

    std::vector<GeometrySource> m_geometry_sources;

    // GERTGeometry array, one host visible buffer per getCurrentBufferIdx()
    // (frames in flight + 1) so it's written directly without any barrier or
    // fence, like the ObjectData of GEVulkanDrawCall::generate
    std::unique_ptr<GEVulkanDynamicBuffer> m_geometry_data;

    // Set 0 of the ray traced pass: the TLAS, the data buffer and the geometry
    // buffer (see solid_rt.frag), one for each buffer of m_geometry_data and
    // updated after the scene is built
    VkDescriptorSetLayout m_descriptor_layout;

    VkDescriptorPool m_descriptor_pool;

    std::vector<VkDescriptorSet> m_descriptor_sets;

    VkBuffer m_blas_buffer;

    VmaAllocation m_blas_memory;

    VkAccelerationStructureKHR m_blas;

    // The top level acceleration structure of a buffer index (frames in
    // flight + 1 of them, same as m_geometry_data), with the host visible
    // buffer of its VkAccelerationStructureInstanceKHR array which is written
    // directly (never read) before the fence is waited, so a frame never
    // touches what the gpu is still reading
    struct TopLevel
    {
        VkBuffer m_instance_buffer;
        VmaAllocation m_instance_memory;
        void* m_instance_mapped;
        VkBuffer m_tlas_buffer;
        VmaAllocation m_tlas_memory;
        VkAccelerationStructureKHR m_tlas;
        // ------------------------------------------------------------------
        TopLevel()
        {
            m_instance_buffer = VK_NULL_HANDLE;
            m_instance_memory = VK_NULL_HANDLE;
            m_instance_mapped = NULL;
            m_tlas_buffer = VK_NULL_HANDLE;
            m_tlas_memory = VK_NULL_HANDLE;
            m_tlas = VK_NULL_HANDLE;
        }
    };

    std::vector<TopLevel> m_top_levels;

    // Number of instances the instance buffers and the TLAS can hold
    uint32_t m_instance_capacity;

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
    // Keep infos (after all offsets are final) and write it to all buffers
    bool createGeometryData(const std::vector<GERTGeometry>& infos);
    // ------------------------------------------------------------------------
    void updateDescriptorSet();
    // ------------------------------------------------------------------------
    bool buildAccelerationStructures(
        const std::vector<VkAccelerationStructureGeometryKHR>& geometries,
        const std::vector<VkAccelerationStructureBuildRangeInfoKHR>& ranges);
    // ------------------------------------------------------------------------
    // The instance of the static BLAS (identity, world space triangles, custom
    // index 0 which is the first GERTGeometry of the BLAS)
    VkAccelerationStructureInstanceKHR getStaticInstance() const;
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
    bool isReady() const
    {
        return !m_top_levels.empty() &&
            m_top_levels[0].m_tlas != VK_NULL_HANDLE;
    }
    // ------------------------------------------------------------------------
    VkAccelerationStructureKHR getTLAS(unsigned buffer_idx) const
                                    { return m_top_levels[buffer_idx].m_tlas; }
    // ------------------------------------------------------------------------
    VkDescriptorSetLayout getDescriptorSetLayout() const
                                                { return m_descriptor_layout; }
    // ------------------------------------------------------------------------
    // Valid to bind if isReady(), buffer_idx is GEVulkanDriver::
    // getCurrentBufferIdx() of the frame (update() writes that buffer)
    const VkDescriptorSet* getDescriptorSet(unsigned buffer_idx) const
                                     { return &m_descriptor_sets[buffer_idx]; }
    // ------------------------------------------------------------------------
    VkBuffer getDataBuffer() const                    { return m_data_buffer; }
    // ------------------------------------------------------------------------
    // Once per frame in GEVulkanDriver::endScene before the fence is waited
    // (like GEVulkanDrawCall::generate, so the buffer of
    // GEVulkanDriver::getCurrentBufferIdx() is not used by the gpu): write all
    // GERTGeometry of this frame directly into the mapped host buffer, with
    // the uv shift (texture matrix), hue and custom vertex color refreshed
    // from the materials of the nodes, so animated textures and color changes
    // of the static nodes are seen in the reflection. The acceleration
    // structures are never touched. Nothing is written if not isReady()
    void update();
    // ------------------------------------------------------------------------
    uint32_t getGeometryCount() const              { return m_geometry_count; }
    // ------------------------------------------------------------------------
    uint32_t getTriangleCount() const              { return m_triangle_count; }
};   // GEVulkanRTScene

}

#endif
