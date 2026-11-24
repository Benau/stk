#include "ge_vulkan_static_draw_source.hpp"

#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_render_info.hpp"
#include "ge_spm_buffer.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_lod_node.hpp"
#include "ge_vulkan_mesh_cache.hpp"
#include "ge_vulkan_shader_manager.hpp"
#include "ge_vulkan_static_scene_manager.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include "IMeshSceneNode.h"

#include <cstring>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
// Must match static_draws.comp
struct Source
{
    ObjectData m_object;
    irr::core::vector3df m_origin;
    int m_min_distance;
    irr::core::vector3df m_min_edge;
    int m_max_distance;
    irr::core::vector3df m_max_edge;
    unsigned m_command;
};
static_assert(sizeof(ObjectData) == 64 && sizeof(Source) == 112,
    "The records must match static_draws.comp");
static_assert(sizeof(VkDrawIndexedIndirectCommand) == 20,
    "The commands must match static_draws.comp");

// The shader (and the draw call, to find the group) of one command
typedef std::tuple<std::string, int, GESPMBuffer*> BatchKey;

}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanStaticDrawSource::GEVulkanStaticDrawSource()
                        : m_mesh_generation(0), m_command_count(0),
                          m_source_count(0), m_buffer(VK_NULL_HANDLE),
                          m_memory(NULL), m_commands_offset(0),
                          m_compute_set_layout(VK_NULL_HANDLE),
                          m_pipeline_layout(VK_NULL_HANDLE),
                          m_pipeline(VK_NULL_HANDLE)
{
}   // GEVulkanStaticDrawSource

// ----------------------------------------------------------------------------
GEVulkanStaticDrawSource::~GEVulkanStaticDrawSource()
{
    GEVulkanDriver* vk = getVKDriver();
    if (m_buffer != VK_NULL_HANDLE)
        vk->waitIdle();
    VkDevice device = vk->getDevice();
    vkDestroyPipeline(device, m_pipeline, NULL);
    vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    vkDestroyDescriptorSetLayout(device, m_compute_set_layout, NULL);
    vmaDestroyBuffer(vk->getVmaAllocator(), m_buffer, m_memory);
}   // ~GEVulkanStaticDrawSource

// ----------------------------------------------------------------------------
VkDeviceSize GEVulkanStaticDrawSource::getSourceSize()
{
    return sizeof(Source);
}   // getSourceSize

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawSource::isCurrent() const
{
    return m_mesh_generation == getVKDriver()->getVulkanMeshCache()
        ->getUpdateCount();
}   // isCurrent

// ----------------------------------------------------------------------------
void GEVulkanStaticDrawSource::useCPUOnly(size_t entry_count)
{
    m_ranges.clear();
    m_materials.clear();
    m_command_count = m_source_count = 0;
    m_on_cpu.assign(entry_count, true);
    m_cpu_entries.resize(entry_count);
    std::iota(m_cpu_entries.begin(), m_cpu_entries.end(), 0u);
}   // useCPUOnly

// ----------------------------------------------------------------------------
void GEVulkanStaticDrawSource::build(const GEVulkanStaticSceneManager& scene,
                                     GEVulkanDrawCall& dc)
{
    GEVulkanDriver* vk = getVKDriver();
    m_mesh_generation = vk->getVulkanMeshCache()->getUpdateCount();
    // The textures are bound for each draw, so only draws of the same
    // texture can be put in a command
    const bool bind_textures = !dc.m_texture_descriptor->isSingleDescriptor();
    std::map<BatchKey, std::vector<Source> > batches;
    for (const auto& entry : scene.getEntries())
    {
        std::vector<GEVulkanLODNode::Level> levels;
        if (entry.m_lod)
            entry.m_lod->getDrawLevels(levels);
        else
            levels.push_back({entry.m_node, 0, -1});
        irr::scene::ISceneNode* highest = entry.m_lod ?
            entry.m_lod->getHighestLevel() : entry.m_node;
        // Same as the distance of GEVulkanLODNode::getLevelForView
        const irr::core::vector3df origin =
            highest->getAbsoluteTransformation().getTranslation();

        // All the levels of an entry or none, otherwise the level which stays
        // on the CPU would be drawn together with the ones of the GPU
        std::vector<std::pair<BatchKey, Source> > records;
        bool on_gpu = true;
        for (const auto& level : levels)
        {
            auto* node = static_cast<irr::scene::IMeshSceneNode*>(
                level.m_node);
            irr::scene::IMesh* mesh = node->getMesh();
            node->refreshTextureDescriptorID();
            for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
            {
                const irr::video::SMaterial& material = node->getMaterial(i);
                const auto& info = material.getRenderInfo();
                const std::string& shader = GEMaterialManager::getShader(
                    dc.m_fallback_materials[node->getMaterialType(i)]);
                auto pipeline = dc.m_graphics_pipelines.find(shader);
                const int slot = node->getTextureDescriptorID(i);
                // Transparent materials stay on the CPU (see also
                // GEVulkanDrawCall::renderPipeline): they are blended in
                // the order of the distance to the camera, but the objects
                // here are compacted by the GPU in an arbitrary order and
                // drawn together. Glow outlines and texture matrices (the
                // object data has no room for them) need the CPU data too
                if (slot < 0 || material.TextureLayer[0].TextureMatrix ||
                    info->hasTransparencySetting() ||
                    info->getGlowOutlineColor().color ||
                    pipeline == dc.m_graphics_pipelines.end() ||
                    !pipeline->second.m_pipelines.count(GVPT_SOLID))
                {
                    on_gpu = false;
                    break;
                }
                GESPMBuffer* buffer = static_cast<GESPMBuffer*>(
                    mesh->getMeshBuffer(i));
                irr::core::aabbox3df bounds = buffer->getBoundingBox();
                node->getAbsoluteTransformation().transformBoxEx(bounds);
                Source source = {};
                source.m_object.init(node, slot, 0, material);
                source.m_origin = origin;
                source.m_min_distance = level.m_min_distance_sq;
                source.m_max_distance = level.m_max_distance_sq;
                source.m_min_edge = bounds.MinEdge;
                source.m_max_edge = bounds.MaxEdge;
                records.emplace_back(BatchKey(shader, bind_textures ? slot : -1,
                    buffer), source);
            }
            if (!on_gpu)
                break;
        }
        if (on_gpu)
        {
            for (auto& record : records)
                batches[record.first].push_back(record.second);
        }
        else
            m_cpu_entries.push_back(m_on_cpu.size());
        m_on_cpu.push_back(!on_gpu);
    }
    if (batches.empty())
        return;

    // One command for each batch, the sources of a command are contiguous
    // and the objects of the destination use the same indices
    std::vector<Source> sources;
    std::vector<VkDrawIndexedIndirectCommand> commands;
    for (auto& batch : batches)
    {
        const GESPMBuffer* mesh = std::get<2>(batch.first);
        Range& range = m_ranges[std::get<0>(batch.first)];
        if (range.m_count == 0)
            range.m_first = commands.size();
        range.m_count++;
        if (bind_textures)
            m_materials.push_back(std::get<1>(batch.first));
        VkDrawIndexedIndirectCommand command = {};
        command.indexCount = mesh->getIndexCount();
        command.firstIndex = mesh->getIBOOffset();
        command.vertexOffset = mesh->getVBOOffset();
        command.firstInstance = sources.size();
        for (Source& source : batch.second)
        {
            source.m_command = commands.size();
            sources.push_back(source);
        }
        commands.push_back(command);
    }
    m_command_count = commands.size();
    m_source_count = sources.size();

    const VkPhysicalDeviceLimits& limits =
        vk->getPhysicalDeviceProperties().limits;
    const VkDeviceSize objects_size = m_source_count * sizeof(ObjectData);
    if (m_source_count * sizeof(Source) > limits.maxStorageBufferRange ||
        objects_size > limits.maxStorageBufferRange ||
        (m_source_count + getGroupSize() - 1) / getGroupSize() >
        limits.maxComputeWorkGroupCount[0] ||
        !createBuffer(sources.data(), commands.data()) || !createPipeline())
    {
        logMessage(GLL_WARNING,
            "Static GPU draws are unavailable, using the CPU");
        useCPUOnly(scene.getEntries().size());
        return;
    }
    logMessage(GLL_INFO, "Static GPU draws: %u commands, %u records, "
        "%zu entries on the CPU", m_command_count, m_source_count,
        m_cpu_entries.size());
}   // build

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawSource::createBuffer(const void* sources,
                                            const void* commands)
{
    GEVulkanDriver* vk = getVKDriver();
    const VkDeviceSize sources_size = m_source_count * sizeof(Source);
    const VkDeviceSize commands_size =
        m_command_count * sizeof(VkDrawIndexedIndirectCommand);
    m_commands_offset = sources_size + getPadding(sources_size,
        vk->getPhysicalDeviceProperties().limits
        .minStorageBufferOffsetAlignment);
    const VkDeviceSize size = m_commands_offset + commands_size;

    VkBuffer staging_buffer = VK_NULL_HANDLE;
    VmaAllocation staging_memory = VK_NULL_HANDLE;
    VmaAllocationCreateInfo staging_info = {};
    staging_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    staging_info.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    staging_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if (!vk->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, staging_info,
        staging_buffer, staging_memory))
        return false;
    uint8_t* mapped = NULL;
    if (vmaMapMemory(vk->getVmaAllocator(), staging_memory,
        (void**)&mapped) != VK_SUCCESS)
    {
        vmaDestroyBuffer(vk->getVmaAllocator(), staging_buffer,
            staging_memory);
        return false;
    }
    memcpy(mapped, sources, sources_size);
    memset(mapped + sources_size, 0, m_commands_offset - sources_size);
    memcpy(mapped + m_commands_offset, commands, commands_size);
    vmaUnmapMemory(vk->getVmaAllocator(), staging_memory);
    vmaFlushAllocation(vk->getVmaAllocator(), staging_memory, 0, size);

    // The commands are copied to the destination of each view before the
    // dispatch, so the sources and the templates never change
    VmaAllocationCreateInfo local_info = {};
    local_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    const bool created = vk->createBuffer(size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT, local_info, m_buffer, m_memory);
    if (created)
        vk->copyBuffer(staging_buffer, m_buffer, size);
    vmaDestroyBuffer(vk->getVmaAllocator(), staging_buffer, staging_memory);
    return created;
}   // createBuffer

// ----------------------------------------------------------------------------
bool GEVulkanStaticDrawSource::createPipeline()
{
    VkDevice device = getVKDriver()->getDevice();
    // The camera of the view, the records, then the destination commands and
    // objects
    VkDescriptorSetLayoutBinding bindings[4] = {};
    for (unsigned i = 0; i < 4; i++)
    {
        bindings[i].binding = i;
        bindings[i].descriptorType = i == 0 ?
            VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC :
            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo set_info = {};
    set_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_info.bindingCount = 4;
    set_info.pBindings = bindings;
    if (vkCreateDescriptorSetLayout(device, &set_info, NULL,
        &m_compute_set_layout) != VK_SUCCESS)
        return false;

    // The camera position for the levels of detail
    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.size = sizeof(float) * 4;
    VkPipelineLayoutCreateInfo layout_info = {};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &m_compute_set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(device, &layout_info, NULL,
        &m_pipeline_layout) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    try
    {
        pipeline_info.stage.module =
            GEVulkanShaderManager::getShader("static_draws.comp");
    }
    catch (const std::runtime_error& e)
    {
        logMessage(GLL_WARNING, "%s", e.what());
        return false;
    }
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_pipeline_layout;
    return vkCreateComputePipelines(device,
        getVKDriver()->getPipelineCache(), 1, &pipeline_info, NULL,
        &m_pipeline) == VK_SUCCESS;
}   // createPipeline

}
