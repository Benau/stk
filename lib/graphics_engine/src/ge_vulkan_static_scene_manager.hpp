#ifndef HEADER_GE_VULKAN_STATIC_SCENE_MANAGER_HPP
#define HEADER_GE_VULKAN_STATIC_SCENE_MANAGER_HPP

#include "ISceneNode.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace GE
{
class GEVulkanLODNode;
class GEVulkanDrawCall;
class GEVulkanSceneManager;
class GEVulkanStaticDrawSource;
class StaticSceneBranch;

// ============================================================================
class GEVulkanStaticSceneManager
{
public:
    using Node = irr::scene::ISceneNode;
    struct Entry
    {
        Node* m_node;
        GEVulkanLODNode* m_lod;
    };
private:
    std::vector<StaticSceneBranch*> m_branches;
    std::vector<Entry> m_entries;
    std::vector<Node*> m_rt_meshes;
    std::vector<Node*> m_rt_grass;
    // Whether the draws can be generated on the GPU (see
    // GEVulkanStaticDrawSource), it's decided when the scene is split
    bool m_gpu_draws;
    bool m_split;
    bool m_live_static;

    // Shared by the draw calls, created when the first one needs it
    std::shared_ptr<GEVulkanStaticDrawSource> m_draw_source;

    // ------------------------------------------------------------------------
    void splitChildren(Node* parent);
    // ------------------------------------------------------------------------
    void cache(Node* node);
public:
    // ------------------------------------------------------------------------
    GEVulkanStaticSceneManager() : m_gpu_draws(false), m_split(false),
                                   m_live_static(true)                       {}
    // ------------------------------------------------------------------------
    ~GEVulkanStaticSceneManager();
    // ------------------------------------------------------------------------
    void split(GEVulkanSceneManager* manager, bool gpu_draws);
    // ------------------------------------------------------------------------
    void release();
    // ------------------------------------------------------------------------
    void registerNodes(GEVulkanSceneManager* manager,
                       const irr::core::matrix4& camera_transform);
    // ------------------------------------------------------------------------
    bool isSplit() const                                    { return m_split; }
    // ------------------------------------------------------------------------
    bool useGPUDraws() const                            { return m_gpu_draws; }
    // ------------------------------------------------------------------------
    const std::vector<Entry>& getEntries() const          { return m_entries; }
    // ------------------------------------------------------------------------
    // NULL if the draws are not generated on the GPU
    std::shared_ptr<GEVulkanStaticDrawSource>
        getDrawSource(GEVulkanDrawCall& dc);
    // ------------------------------------------------------------------------
    bool hasLiveStatic() const                        { return m_live_static; }
    // ------------------------------------------------------------------------
    const std::vector<Node*>& getRTMeshes() const       { return m_rt_meshes; }
    // ------------------------------------------------------------------------
    const std::vector<Node*>& getRTGrass() const         { return m_rt_grass; }
    // ------------------------------------------------------------------------
    void detectDeferred(GEVulkanSceneManager* manager) const;
};   // GEVulkanStaticSceneManager

}

#endif
