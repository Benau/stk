#include "ge_vulkan_animated_mesh_scene_node.hpp"

#include "ge_animation.hpp"
#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_render_info.hpp"
#include "ge_spm.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_scene_manager.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include "ISceneManager.h"
#include "../../../lib/irrlicht/source/Irrlicht/CBoneSceneNode.h"
#include <limits>

namespace GE
{
GEVulkanAnimatedMeshSceneNode::GEVulkanAnimatedMeshSceneNode(irr::scene::IAnimatedMesh* mesh,
    irr::scene::ISceneNode* parent, irr::scene::ISceneManager* mgr, irr::s32 id,
    const irr::core::vector3df& position,
    const irr::core::vector3df& rotation,
    const irr::core::vector3df& scale)
    : irr::scene::CAnimatedMeshSceneNode(mesh, parent, mgr, id, position,
                                         rotation, scale)
{
    m_saved_transition_frame = -1.0f;
    m_skinning_offset = 0;
    setAbsoluteRotationScale(AbsoluteTransformation, m_abs_rotation,
        m_abs_scale);
}   // GEVulkanAnimatedMeshSceneNode

// ----------------------------------------------------------------------------
void GEVulkanAnimatedMeshSceneNode::OnRegisterSceneNode()
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
void GEVulkanAnimatedMeshSceneNode::setMesh(irr::scene::IAnimatedMesh* mesh)
{
    GESPM* spm = dynamic_cast<GESPM*>(mesh);
    if (!spm)
        return;
    CAnimatedMeshSceneNode::setMesh(spm);
    m_texture_descriptor_ids.resize(getMaterialCount());
    m_texture_descriptor_ids_observer.clear();
    m_texture_descriptor_ids_observer.resize(getMaterialCount());
    m_transparency_observer.reset();
    m_ge_materials.clear();
    cleanJoints();
    if (spm->isStatic())
        return;

    unsigned bone_idx = 0;
    for (Armature& arm : spm->getArmatures())
    {
        for (const std::string& bone_name : arm.m_joint_names)
        {
            m_joint_nodes[bone_name] = new CBoneSceneNode(this,
                SceneManager, 0, bone_idx++, bone_name.c_str());
            m_joint_nodes.at(bone_name)->drop();
            m_joint_nodes.at(bone_name)->setSkinningSpace(EBSS_GLOBAL);
        }
    }
}   // setMesh

// ----------------------------------------------------------------------------
void GEVulkanAnimatedMeshSceneNode::OnAnimate(irr::u32 time_ms)
{
    GESPM* spm = static_cast<GESPM*>(Mesh);
    if (!spm || spm->isStatic())
    {
        IAnimatedMeshSceneNode::OnAnimate(time_ms);
        return;
    }

    // first frame
    if (LastTimeMs == 0)
        LastTimeMs = time_ms;

    // set CurrentFrameNr
    buildFrameNr(time_ms - LastTimeMs);
    LastTimeMs = time_ms;

    GEVulkanSceneManager* sm =
        static_cast<GEVulkanSceneManager*>(SceneManager);
    GEVulkanDrawCall* dc = sm->getActiveDrawCall();
    if (dc)
    {
        spm->getSkinningMatrices(getFrameNr(), dc->getSkinningOffset(
            spm->getJointCount(), &m_skinning_offset),
            m_saved_transition_frame, TransitingBlend);
    }
    recursiveUpdateAbsolutePosition();

    for (Armature& arm : spm->getArmatures())
    {
        for (unsigned i = 0; i < arm.m_joint_names.size(); i++)
        {
            m_joint_nodes.at(arm.m_joint_names[i])->setAbsoluteTransformation
                (AbsoluteTransformation * arm.m_world_matrices[i].first);
        }
    }

    IAnimatedMeshSceneNode::OnAnimate(time_ms);
}   // OnAnimate

// ----------------------------------------------------------------------------
irr::scene::IBoneSceneNode* GEVulkanAnimatedMeshSceneNode::getJointNode(const irr::c8* joint_name)
{
    auto ret = m_joint_nodes.find(joint_name);
    if (ret != m_joint_nodes.end())
        return ret->second;
    return NULL;
}   // getJointNode

// ----------------------------------------------------------------------------
irr::scene::IBoneSceneNode* GEVulkanAnimatedMeshSceneNode::getJointNode(irr::u32 joint_id)
{
    irr::u32 idx = 0;
    for (auto& p : m_joint_nodes)
    {
        if (joint_id == idx)
            return p.second;
        idx++;
    }
    return NULL;
}   // getJointNode

// ----------------------------------------------------------------------------
void GEVulkanAnimatedMeshSceneNode::setTransitionTime(irr::f32 Time)
{
    if (Time == 0.0f)
    {
        TransitingBlend = TransitionTime = Transiting = 0;
        m_saved_transition_frame = -1.0;
    }
    else
    {
        const u32 ttime = (u32)core::floor32(Time * 1000.0f);
        TransitionTime = ttime;
        Transiting = core::reciprocal((f32)TransitionTime);
        TransitingBlend = 0.0f;
        m_saved_transition_frame = getFrameNr();
    }
}   // setTransitionTime

// ----------------------------------------------------------------------------
void GEVulkanAnimatedMeshSceneNode::updateAbsolutePosition()
{
    scene::CAnimatedMeshSceneNode::updateAbsolutePosition();
    if (UpdatedAbsTrans)
    {
        setAbsoluteRotationScale(AbsoluteTransformation, m_abs_rotation,
            m_abs_scale);
    }
}   // updateAbsolutePosition

}
