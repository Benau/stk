#ifndef HEADER_GE_VULKAN_LOD_NODE_HPP
#define HEADER_GE_VULKAN_LOD_NODE_HPP

#include "ISceneNode.h"

namespace GE
{
class GEVulkanLODNode
{
public:
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
