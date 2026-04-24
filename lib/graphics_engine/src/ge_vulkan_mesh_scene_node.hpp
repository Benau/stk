#ifndef HEADER_GE_VULKAN_MESH_SCENE_NODE_HPP
#define HEADER_GE_VULKAN_MESH_SCENE_NODE_HPP

#include "../source/Irrlicht/CMeshSceneNode.h"
#include <vector>

namespace GE
{
class GESPM;

class GEVulkanMeshSceneNode : public irr::scene::CMeshSceneNode
{
private:
    bool m_remove_from_mesh_cache;

    std::weak_ptr<bool> m_transparency_observer;

    irr::core::array<irr::video::E_MATERIAL_TYPE> m_ge_materials;

    std::vector<int> m_texture_descriptor_ids;

    std::vector<std::weak_ptr<int> > m_texture_descriptor_ids_observer;

    irr::core::quaternion m_abs_rotation;

    irr::core::vector3df m_abs_scale;
public:
    // ------------------------------------------------------------------------
    GEVulkanMeshSceneNode(irr::scene::IMesh* mesh,
        irr::scene::ISceneNode* parent, irr::scene::ISceneManager* mgr, irr::s32 id,
        const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& rotation = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& scale = irr::core::vector3df(1.0f, 1.0f, 1.0f));
    // ------------------------------------------------------------------------
    ~GEVulkanMeshSceneNode();
    // ------------------------------------------------------------------------
    void setRemoveFromMeshCache(bool val)   { m_remove_from_mesh_cache = val; }
    // ------------------------------------------------------------------------
    virtual void OnRegisterSceneNode();
    // ------------------------------------------------------------------------
    virtual void setMesh(irr::scene::IMesh* mesh)
    {
        CMeshSceneNode::setMesh(mesh);
        m_transparency_observer.reset();
        m_ge_materials.clear();
        m_texture_descriptor_ids.resize(getMaterialCount());
        m_texture_descriptor_ids_observer.clear();
        m_texture_descriptor_ids_observer.resize(getMaterialCount());
    }
    // ------------------------------------------------------------------------
    virtual irr::video::E_MATERIAL_TYPE getMaterialType(irr::u32 i)
    {
        if (m_ge_materials.empty())
            return CMeshSceneNode::getMaterialType(i);
        return m_ge_materials[i];
    }
    // ------------------------------------------------------------------------
    virtual irr::s32 getTextureDescriptorID(irr::u32 i) const
    {
        if (m_texture_descriptor_ids.empty())
            return CMeshSceneNode::getTextureDescriptorID(i);
        return m_texture_descriptor_ids[i];
    }
    // ------------------------------------------------------------------------
    virtual void updateAbsolutePosition();
    // ------------------------------------------------------------------------
    virtual const irr::core::quaternion& getAbsoluteRotation() const
                                                     { return m_abs_rotation; }
    // ------------------------------------------------------------------------
    virtual const irr::core::vector3df& getAbsoluteScale() const
                                                        { return m_abs_scale; }
};   // GEVulkanMeshSceneNode

}

#endif
