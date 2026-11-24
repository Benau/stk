#ifndef HEADER_GE_VULKAN_LOD_NODE_HPP
#define HEADER_GE_VULKAN_LOD_NODE_HPP

#include "ISceneNode.h"
#include <vector>

namespace GE
{
class GEVulkanLODNode
{
public:
    struct Level
    {
        irr::scene::ISceneNode* m_node;
        int m_min_distance_sq;
        int m_max_distance_sq;
    };
    virtual void getDrawLevels(std::vector<Level>& levels) = 0;
    // ------------------------------------------------------------------------
    virtual ~GEVulkanLODNode() {}
    // ------------------------------------------------------------------------
    virtual irr::scene::ISceneNode* getLevelForView(
        const irr::core::matrix4& camera_transform) = 0;
    // ------------------------------------------------------------------------
    virtual irr::scene::ISceneNode* getHighestLevel() = 0;
    // ------------------------------------------------------------------------
    virtual bool hasOnlyLevels() const = 0;
};   // GEVulkanLODNode

}

#endif
