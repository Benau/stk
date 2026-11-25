#ifndef HEADER_GE_VULKAN_STATIC_DRAW_SOURCE_HPP
#define HEADER_GE_VULKAN_STATIC_DRAW_SOURCE_HPP

#include "ge_vma.hpp"
#include "vulkan_wrapper.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace GE
{
class GEVulkanDrawCall;
class GEVulkanStaticSceneManager;

// ============================================================================
// The part of the GPU generated draws of the static scene which is the same
// for every view: one record (object data, level of detail distances and
// bounding box) for each mesh buffer of each static node, and one indirect
// command template for each (shader, mesh buffer) pair. It is built once for
// the scene, so all draw calls (split screen views, and later the shadows)
// share it, each of them only has the destination, see
// GEVulkanStaticDrawBatches
//
// Nodes which can't be drawn this way (see build) stay on the CPU, the draw
// calls register them as usual
class GEVulkanStaticDrawSource
{
public:
    struct Range
    {
        unsigned m_first;
        unsigned m_count;
    };

    // The culling shader of each kind of view, they only differ in the test
    // of the view and the statistics (see static_draws_common.glsl)
    enum Variant : unsigned
    {
        SV_VIEW,   // A camera: frustum, with the polygon statistics
        SV_SHADOW, // A directional light: frustum without near plane
        SV_OMNI,   // A bucket of omni lights: spheres of the lights
        SV_COUNT
    };

private:
    uint64_t m_mesh_generation;

    // Per entry of the static scene manager
    std::vector<bool> m_on_cpu;

    std::vector<unsigned> m_cpu_entries;

    // The commands of each shader are contiguous
    std::map<std::string, Range> m_ranges;

    // The shadow draw calls use the non PBR fallback of the shaders, so a
    // shader of them can have commands which are not next to each other
    std::map<std::string, std::vector<Range> > m_shadow_ranges;

    // Whether the shadow draw calls have a pipeline for all the commands
    bool m_shadow_usable;

    // The texture descriptor slot of each command if the textures are bound
    // for each draw, otherwise empty
    std::vector<int> m_materials;

    unsigned m_command_count, m_source_count;

    // The records, then the command templates
    VkBuffer m_buffer;

    VmaAllocation m_memory;

    VkDeviceSize m_commands_offset;

    VkDescriptorSetLayout m_compute_set_layout;

    VkPipelineLayout m_pipeline_layout;

    // The pipeline of the shadow views are optional (null if it fails)
    VkPipeline m_pipelines[SV_COUNT];

    // ------------------------------------------------------------------------
    bool createBuffer(const void* sources, const void* commands);
    // ------------------------------------------------------------------------
    bool createPipelines();
    // ------------------------------------------------------------------------
    VkPipeline createPipeline(Variant variant);
    // ------------------------------------------------------------------------
    void useCPUOnly(size_t entry_count);
public:
    // ------------------------------------------------------------------------
    GEVulkanStaticDrawSource();
    // ------------------------------------------------------------------------
    ~GEVulkanStaticDrawSource();
    // ------------------------------------------------------------------------
    void build(const GEVulkanStaticSceneManager& scene, GEVulkanDrawCall& dc);
    // ------------------------------------------------------------------------
    // Whether the draws are still valid for the meshes in the mesh cache
    bool isCurrent() const;
    // ------------------------------------------------------------------------
    bool isUsable() const                         { return !m_ranges.empty(); }
    // ------------------------------------------------------------------------
    bool onCPU(unsigned entry) const                { return m_on_cpu[entry]; }
    // ------------------------------------------------------------------------
    const std::vector<unsigned>& getCPUEntries() const
                                                      { return m_cpu_entries; }
    // ------------------------------------------------------------------------
    const std::map<std::string, Range>& getRanges() const  { return m_ranges; }
    // ------------------------------------------------------------------------
    // Like getRanges, but with the shaders of the shadow draw calls
    const std::map<std::string, std::vector<Range> >& getShadowRanges() const
                                                    { return m_shadow_ranges; }
    // ------------------------------------------------------------------------
    // Whether the shadow views can use the draws of the variant, otherwise
    // the nodes have to be added to them on the CPU
    bool isShadowUsable(Variant variant) const
    {
        return m_shadow_usable && m_pipelines[variant] != VK_NULL_HANDLE;
    }
    // ------------------------------------------------------------------------
    int getMaterial(unsigned command) const
    {
        return m_materials.empty() ? -1 : m_materials[command];
    }
    // ------------------------------------------------------------------------
    unsigned getCommandCount() const                { return m_command_count; }
    // ------------------------------------------------------------------------
    unsigned getSourceCount() const                  { return m_source_count; }
    // ------------------------------------------------------------------------
    VkBuffer getBuffer() const                             { return m_buffer; }
    // ------------------------------------------------------------------------
    static VkDeviceSize getSourceSize();
    // ------------------------------------------------------------------------
    VkDeviceSize getCommandsOffset() const        { return m_commands_offset; }
    // ------------------------------------------------------------------------
    VkDescriptorSetLayout getComputeSetLayout() const
                                               { return m_compute_set_layout; }
    // ------------------------------------------------------------------------
    VkPipelineLayout getPipelineLayout() const    { return m_pipeline_layout; }
    // ------------------------------------------------------------------------
    VkPipeline getPipeline(Variant variant) const
                                               { return m_pipelines[variant]; }
    // ------------------------------------------------------------------------
    static unsigned getGroupSize()                               { return 64; }
};   // GEVulkanStaticDrawSource

}

#endif
