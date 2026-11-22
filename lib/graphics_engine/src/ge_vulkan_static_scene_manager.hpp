#ifndef HEADER_GE_VULKAN_STATIC_SCENE_MANAGER_HPP
#define HEADER_GE_VULKAN_STATIC_SCENE_MANAGER_HPP

#include "ISceneNode.h"

#include <vector>

namespace GE
{
class GEVulkanLODNode;
class GEVulkanSceneManager;
class StaticSceneBranch;

// ============================================================================
class GEVulkanStaticSceneManager
{
private:
    using Node = irr::scene::ISceneNode;
    struct Entry
    {
        Node* m_node;
        GEVulkanLODNode* m_lod;
    };
    std::vector<StaticSceneBranch*> m_branches;
    std::vector<Entry> m_entries;
    std::vector<Node*> m_rt_meshes;
    std::vector<Node*> m_rt_grass;
    bool m_split;
    bool m_live_static;

    // ------------------------------------------------------------------------
    void splitChildren(Node* parent);
    // ------------------------------------------------------------------------
    void cache(Node* node);
public:
    // ------------------------------------------------------------------------
    GEVulkanStaticSceneManager() : m_split(false), m_live_static(true)       {}
    // ------------------------------------------------------------------------
    ~GEVulkanStaticSceneManager();
    // ------------------------------------------------------------------------
    void split(GEVulkanSceneManager* manager);
    // ------------------------------------------------------------------------
    void release();
    // ------------------------------------------------------------------------
    void registerNodes(GEVulkanSceneManager* manager,
                       const irr::core::matrix4& camera_transform);
    // ------------------------------------------------------------------------
    bool isSplit() const                                    { return m_split; }
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
