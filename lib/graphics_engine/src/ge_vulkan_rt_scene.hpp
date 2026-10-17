#ifndef HEADER_GE_VULKAN_RT_SCENE_HPP
#define HEADER_GE_VULKAN_RT_SCENE_HPP

#include "ge_vma.hpp"
#include "vulkan_wrapper.h"

#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace irr
{
    namespace scene { class IMesh; class ISceneNode; }
}

namespace GE
{
class GEVulkanDriver;
class GEVulkanDynamicBuffer;
class GEVulkanMeshCache;

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
    // GERTGeometryFlag
    uint32_t m_flags;
};

enum GERTGeometryFlag : uint32_t
{
    // The vertices are in the mesh cache of the driver (m_vertex_offset is a
    // vertex index of GEVulkanMeshCache), not in the data buffer
    GRTGF_CACHE_VERTEX = 1,
    // The indices are 16 bit ones in the mesh cache (m_index_offset is a
    // 16 bit index from the start of the mesh cache buffer)
    GRTGF_CACHE_INDEX = 2
};

// All the static nodes (those with GRTT_STATIC in the render info of the
// materials) are joined into a single bottom level acceleration structure of
// world space triangles, which is shared by all cameras. The scene should be
// built after the nodes are added and cleared before the scene is cleared.
//
// If GEConfig::m_dynamic_ray_tracing is true (and the mesh cache of the driver
// is used, GEVulkanFeatures::supportsBaseVertexRendering), the nodes with
// GRTT_DYNAMIC (and no skinning) are found every frame (update), each unique
// mesh of them has its own BLAS in the model space, which is built from the
// vertices and indices of the mesh cache (no copy of them, the shader reads
// them there too). It's created once when it's first seen, and destroyed some
// frames after the last node using it is gone, so a node which is added,
// removed or uses another mesh by setMesh only changes the instances. Every
// node is an instance of it in the TLAS with its transform. The TLAS is built
// every frame the instances changed. When the mesh cache is created again (a
// mesh is added or removed) all of them are created again, as the offsets and
// the buffer are new
class GEVulkanRTScene
{
public:
    enum
    {
        // Instances of a TLAS, the first one is the static BLAS
        MAX_INSTANCES = 256,
        // GERTGeometry of all the dynamic instances of a frame, after the
        // ones of the static BLAS
        MAX_DYNAMIC_GEOMETRIES = 1024
    };
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
        // Scratch of the builds recorded for this index (the TLAS and the
        // new BLAS), grown when needed
        VkBuffer m_scratch_buffer;
        VmaAllocation m_scratch_memory;
        VkDeviceSize m_scratch_size;
        VkDeviceAddress m_scratch_address;
        // Instances written to m_instance_buffer
        uint32_t m_instance_count;
        // The instance buffer / the geometry buffer has only what the build
        // has (the static BLAS), so the dynamic ray tracing was off
        bool m_instance_static;
        bool m_geometry_static;
        // The TLAS has to be built again with the instance buffer
        bool m_dirty;
        // GEVulkanMeshCache::getUpdateCount of the mesh cache buffer in the
        // descriptor set of this index
        uint64_t m_cache_version;
        // Slots of the mesh texture descriptor used by the dynamic
        // instances, alive until this index is used again
        std::vector<std::shared_ptr<int> > m_slots;
        // ------------------------------------------------------------------
        TopLevel()
        {
            m_scratch_buffer = VK_NULL_HANDLE;
            m_scratch_memory = VK_NULL_HANDLE;
            m_scratch_size = 0;
            m_scratch_address = 0;
            m_instance_count = 1;
            m_instance_static = true;
            m_geometry_static = true;
            m_dirty = false;
            m_cache_version = 0;
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

    // The scratch size of building a TLAS with m_instance_capacity instances
    VkDeviceSize m_tlas_scratch_size;

    // Model space BLAS of a mesh, only the geometries (mesh buffers) in
    // m_buffer_mask are in it, the same mesh used with other materials is
    // another one. The mesh is not grabbed, they are all destroyed when the
    // mesh cache changes (which is when a mesh is gone)
    struct MeshKey
    {
        irr::scene::IMesh* m_mesh;
        uint64_t m_buffer_mask;
        bool operator<(const MeshKey& o) const
        {
            return m_mesh != o.m_mesh ? m_mesh < o.m_mesh :
                m_buffer_mask < o.m_buffer_mask;
        }
    };

    struct MeshBLAS
    {
        // The baked fields (offsets in the mesh cache, shading type and flags)
        // of each geometry
        std::vector<GERTGeometry> m_infos;
        std::vector<VkAccelerationStructureGeometryKHR> m_as_geometries;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> m_as_ranges;
        VkDeviceSize m_build_scratch_size;
        VkBuffer m_blas_buffer;
        VmaAllocation m_blas_memory;
        VkAccelerationStructureKHR m_blas;
        VkDeviceAddress m_address;
        uint64_t m_last_used_frame;
        // ------------------------------------------------------------------
        MeshBLAS()
        {
            m_build_scratch_size = 0;
            m_blas_buffer = VK_NULL_HANDLE;
            m_blas_memory = VK_NULL_HANDLE;
            m_blas = VK_NULL_HANDLE;
            m_address = 0;
            m_last_used_frame = 0;
        }
    };

    std::map<MeshKey, std::unique_ptr<MeshBLAS> > m_mesh_blas;

    // Created but not built (in the next buildTopLevel)
    std::vector<MeshBLAS*> m_pending_blas;

    // GEVulkanMeshCache::getUpdateCount of the mesh cache the BLAS were
    // created with
    uint64_t m_cache_version;

    uint64_t m_frame_counter;

    bool m_warned_dynamic_limit;

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
    void setupTopLevelBuild(const TopLevel& top_level,
        VkAccelerationStructureGeometryKHR& geometry,
        VkAccelerationStructureBuildGeometryInfoKHR& info) const;
    // ------------------------------------------------------------------------
    MeshBLAS* getMeshBLAS(irr::scene::IMesh* mesh, uint64_t buffer_mask,
                          const std::vector<unsigned>& buffer_indices,
                          const std::vector<uint32_t>& shading_types,
                          GEVulkanMeshCache* mesh_cache);
    // ------------------------------------------------------------------------
    // The mesh cache buffer, or the data buffer which is not read (if there is
    // no mesh cache buffer) for the descriptor set
    VkBuffer getMeshCacheDescriptorBuffer(GEVulkanMeshCache* mesh_cache) const;
    // ------------------------------------------------------------------------
    void writeMeshCacheDescriptor(unsigned buffer_idx,
                                  GEVulkanMeshCache* mesh_cache);
    // ------------------------------------------------------------------------
    void destroyMeshBLAS(MeshBLAS* blas);
    // ------------------------------------------------------------------------
    void destroyUnusedMeshBLAS();
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
    // (like GEVulkanDrawCall::generate, so the buffers of
    // GEVulkanDriver::getCurrentBufferIdx() are not used by the gpu), writes
    // directly into the mapped host buffers of this frame:
    // - all GERTGeometry, with the uv shift (texture matrix), hue and custom
    //   vertex color refreshed from the materials of the nodes, so animated
    //   textures and color changes are seen in the reflection
    // - the instances (the static BLAS and one for each node with
    //   GRTT_DYNAMIC found under root), the BLAS of new meshes are created
    //   (not built, see buildTopLevel)
    // Only if GEConfig::m_dynamic_ray_tracing is true, otherwise the buffers
    // are restored (once) to what the build wrote. Nothing is written if not
    // isReady(). root is not changed
    void update(irr::scene::ISceneNode* root);
    // ------------------------------------------------------------------------
    // Record the builds of the new BLAS and the TLAS of the current buffer
    // index if its instances changed in update, outside any render pass and
    // before the ray tracing pass. The TLAS is visible to the fragment shaders
    // after it
    void buildTopLevel(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    uint32_t getGeometryCount() const              { return m_geometry_count; }
    // ------------------------------------------------------------------------
    uint32_t getTriangleCount() const              { return m_triangle_count; }
};   // GEVulkanRTScene

}

#endif
