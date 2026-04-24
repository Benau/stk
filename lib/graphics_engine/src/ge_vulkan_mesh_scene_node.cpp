#include "ge_vulkan_mesh_scene_node.hpp"

#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_render_info.hpp"
#include "ge_spm.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include "IMeshCache.h"
#include "ISceneManager.h"

namespace GE
{
GEVulkanMeshSceneNode::GEVulkanMeshSceneNode(irr::scene::IMesh* mesh,
    irr::scene::ISceneNode* parent, irr::scene::ISceneManager* mgr, irr::s32 id,
    const irr::core::vector3df& position,
    const irr::core::vector3df& rotation,
    const irr::core::vector3df& scale)
    : irr::scene::CMeshSceneNode(mesh, parent, mgr, id, position, rotation,
                                 scale)
{
    m_remove_from_mesh_cache = false;
    m_texture_descriptor_ids.resize(getMaterialCount());
    m_texture_descriptor_ids_observer.resize(getMaterialCount());
    setAbsoluteRotationScale(AbsoluteTransformation, m_abs_rotation,
        m_abs_scale);
}   // GEVulkanMeshSceneNode

// ----------------------------------------------------------------------------
GEVulkanMeshSceneNode::~GEVulkanMeshSceneNode()
{
    if (m_remove_from_mesh_cache)
        SceneManager->getMeshCache()->removeMesh(Mesh);
}   // ~GEVulkanMeshSceneNode

// ----------------------------------------------------------------------------
void GEVulkanMeshSceneNode::OnRegisterSceneNode()
{
    if (!IsVisible)
        return;

    if (m_ge_materials.empty() ||
        (m_first_render_info &&
        m_first_render_info->hasTransparencySetting() &&
        m_transparency_observer.expired()))
    {
        m_ge_materials.clear();
        for (unsigned i = 0; i < getMaterialCount(); i++)
            m_ge_materials.push_back(getMaterial(i).MaterialType);
        if (m_first_render_info)
        {
            m_transparency_observer =
                m_first_render_info->getTransparencyObserver();
            if (m_first_render_info->isTransparent())
            {
                video::E_MATERIAL_TYPE ghost =
                    GEMaterialManager::getIrrMaterialType("ghost");
                for (unsigned i = 0; i < m_ge_materials.size(); i++)
                {
                    auto* material = GEMaterialManager::getMaterial(
                        m_ge_materials[i]);
                    // Use real transparent shader first
                    if (material->isTransparent())
                        continue;
                    m_ge_materials[i] = ghost;
                }
            }
        }
    }

    for (unsigned i = 0; i < m_texture_descriptor_ids.size(); i++)
    {
        if (m_texture_descriptor_ids_observer[i].expired())
        {
            GEVulkanTextureDescriptor* td =
                getVKDriver()->getMeshTextureDescriptor();
            // Ghost shader uses the same srgb setting so we can use the
            // original material
            video::SMaterial& m = getMaterial(i);
            std::shared_ptr<int>& slot = td->getTextureID(m,
                GEMaterialManager::getMaterial(m.MaterialType));
            m_texture_descriptor_ids_observer[i] = slot;
            m_texture_descriptor_ids[i] = *slot;
        }
    }

    SceneManager->registerNodeForRendering(this, scene::ESNRP_SOLID);
    ISceneNode::OnRegisterSceneNode();
}   // OnRegisterSceneNode

// ----------------------------------------------------------------------------
void GEVulkanMeshSceneNode::updateAbsolutePosition()
{
    scene::CMeshSceneNode::updateAbsolutePosition();
    if (UpdatedAbsTrans)
    {
        setAbsoluteRotationScale(AbsoluteTransformation, m_abs_rotation,
            m_abs_scale);
    }
}   // updateAbsolutePosition

}
