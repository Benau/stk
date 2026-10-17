#ifndef HEADER_GE_VULKAN_MESH_CACHE_HPP
#define HEADER_GE_VULKAN_MESH_CACHE_HPP

#include "vulkan_wrapper.h"
#include "ge_vma.hpp"
#include "../source/Irrlicht/CMeshCache.h"

#include <cstdint>

namespace GE
{
class GEVulkanDriver;
class GEVulkanMeshCache : public irr::scene::CMeshCache
{
private:
    GEVulkanDriver* m_vk;

    uint64_t m_irrlicht_cache_time, m_ge_cache_time;

    VkBuffer m_buffer;

    VmaAllocation m_memory;

    size_t m_ibo_offset, m_skinning_vbo_offset, m_aabb_offset, m_aabb_size;

    // Increased every time the buffer is created again (all offsets of the
    // mesh buffers are new then)
    uint64_t m_update_count;
public:
    // ------------------------------------------------------------------------
    GEVulkanMeshCache();
    // ------------------------------------------------------------------------
    virtual void meshCacheChanged();
    // ------------------------------------------------------------------------
    void updateCache();
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    VkBuffer getBuffer() const                             { return m_buffer; }
    // ------------------------------------------------------------------------
    uint64_t getUpdateCount() const                  { return m_update_count; }
    // ------------------------------------------------------------------------
    size_t getIBOOffset() const                        { return m_ibo_offset; }
    // ------------------------------------------------------------------------
    size_t getSkinningVBOOffset() const       { return m_skinning_vbo_offset; }
    // ------------------------------------------------------------------------
    size_t getAABBOffset() const                      { return m_aabb_offset; }
    // ------------------------------------------------------------------------
    size_t getAABBSize() const                          { return m_aabb_size; }
};   // GEVulkanMeshCache

}

#endif
