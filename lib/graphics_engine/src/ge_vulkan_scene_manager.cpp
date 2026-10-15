#include "ge_vulkan_scene_manager.hpp"

#include "../source/Irrlicht/os.h"

#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_vulkan_animated_mesh_scene_node.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_deferred_fbo.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_mesh_cache.hpp"
#include "ge_vulkan_mesh_scene_node.hpp"
#include "ge_vulkan_shadow_fbo.hpp"
#include "ge_vulkan_skybox_renderer.hpp"
#include "ge_vulkan_target_group.hpp"
#include "ge_vulkan_texture_descriptor.hpp"

#include "IBillboardSceneNode.h"
#include "ILightSceneNode.h"
#include <algorithm>
#include <sstream>

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanSceneManager::GEVulkanSceneManager(irr::video::IVideoDriver* driver,
                                           irr::io::IFileSystem* fs,
                                           irr::gui::ICursorControl* cursor_control,
                                           irr::gui::IGUIEnvironment* gui_environment)
                    : CSceneManager(driver, fs, cursor_control,
                                    new GEVulkanMeshCache(), gui_environment)
{
    m_current_group = NULL;
    m_current_entry = NULL;
    resetDetectDeferred();
    // CSceneManager grabbed it
    getMeshCache()->drop();
}   // GEVulkanSceneManager

// ----------------------------------------------------------------------------
GEVulkanSceneManager::~GEVulkanSceneManager()
{
}   // ~GEVulkanSceneManager

// ----------------------------------------------------------------------------
void GEVulkanSceneManager::clear()
{
    irr::scene::CSceneManager::clear();
    GEVulkanDriver* vk = static_cast<GEVulkanDriver*>(getVideoDriver());
    vk->waitIdle();
    resetDrawCalls();
    vk->getMeshTextureDescriptor()->clear();
}   // clear

// ----------------------------------------------------------------------------
irr::scene::ICameraSceneNode* GEVulkanSceneManager::addCameraSceneNode(
                                                irr::scene::ISceneNode* parent,
                                          const irr::core::vector3df& position,
                                            const irr::core::vector3df& lookat,
                                                 irr::s32 id, bool make_active)
{
    if (!parent)
        parent = this;

    irr::scene::ICameraSceneNode* node = new GEVulkanCameraSceneNode(parent,
        this, id, position, lookat);

    if (make_active)
        setActiveCamera(node);
    node->drop();

    return node;
}   // addCameraSceneNode

// ----------------------------------------------------------------------------
irr::scene::IAnimatedMeshSceneNode* GEVulkanSceneManager::addAnimatedMeshSceneNode(
    irr::scene::IAnimatedMesh* mesh, irr::scene::ISceneNode* parent,
    irr::s32 id,
    const irr::core::vector3df& position,
    const irr::core::vector3df& rotation,
    const irr::core::vector3df& scale,
    bool alsoAddIfMeshPointerZero)
{
    if (!alsoAddIfMeshPointerZero && (!mesh ||
        mesh->getMeshType() != irr::scene::EAMT_SPM))
        return NULL;

    if (!parent)
        parent = this;

    irr::scene::IAnimatedMeshSceneNode* node =
        new GEVulkanAnimatedMeshSceneNode(mesh, parent, this, id, position,
        rotation, scale);
    node->drop();
    node->setMesh(mesh);
    return node;
}   // addAnimatedMeshSceneNode

// ----------------------------------------------------------------------------
irr::scene::IMeshSceneNode* GEVulkanSceneManager::addMeshSceneNode(
    irr::scene::IMesh* mesh,
    irr::scene::ISceneNode* parent, irr::s32 id,
    const irr::core::vector3df& position,
    const irr::core::vector3df& rotation,
    const irr::core::vector3df& scale,
    bool alsoAddIfMeshPointerZero)
{
    if (!alsoAddIfMeshPointerZero && !mesh)
        return NULL;

    bool convert_irrlicht_mesh = false;
    if (mesh)
    {
        for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
        {
            irr::scene::IMeshBuffer* b = mesh->getMeshBuffer(i);
            if (b->getVertexType() != irr::video::EVT_SKINNED_MESH)
            {
                if (!getGEConfig()->m_convert_irrlicht_mesh)
                {
                    return irr::scene::CSceneManager::addMeshSceneNode(
                        mesh, parent, id, position, rotation, scale,
                        alsoAddIfMeshPointerZero);
                }
                else
                {
                    convert_irrlicht_mesh = true;
                    break;
                }
            }
        }
    }

    if (!parent)
        parent = this;

    if (convert_irrlicht_mesh)
    {
        irr::scene::IAnimatedMesh* spm = convertIrrlichtMeshToSPM(mesh);
        std::stringstream oss;
        oss << (uint64_t)spm;
        getMeshCache()->addMesh(oss.str().c_str(), spm);
        mesh = spm;
    }

    GEVulkanMeshSceneNode* vulkan_node =
        new GEVulkanMeshSceneNode(mesh, parent, this, id, position, rotation,
        scale);
    irr::scene::IMeshSceneNode* node = vulkan_node;
    node->drop();

    if (convert_irrlicht_mesh)
    {
        vulkan_node->setRemoveFromMeshCache(true);
        mesh->drop();
    }
    return node;
}   // addMeshSceneNode

// ----------------------------------------------------------------------------
GEVulkanDrawEntry* GEVulkanSceneManager::acquireEntry(GEVulkanDriver* vk)
{
    GEVulkanFBOTexture* target = vk->getRTTTexture();
    GEVulkanTargetGroup* group = NULL;
    for (auto& g : m_groups)
    {
        if (g->m_target == target && (!target || !g->m_observer.expired()))
        {
            group = g.get();
            break;
        }
    }
    if (!group)
    {
        m_groups.emplace_back(new GEVulkanTargetGroup);
        group = m_groups.back().get();
        group->m_target = target;
        if (target)
            group->m_observer = target->getTextureObserver();
    }
    if (group->m_used == 0)
        m_frame_groups.push_back(group);
    group->m_clear_color = vk->getRTTClearColor();
    m_current_group = group;
    // Don't use the cache of GEVulkanDriver (it's for draw calls which can
    // be used for any target)
    return &group->acquireEntry();
}   // acquireEntry

// ----------------------------------------------------------------------------
void GEVulkanSceneManager::drawAll(irr::u32 flags)
{
    // The actual rendering is in GEVulkanDriver::buildCommandBuffers, which
    // renders every group of the draw calls to its own target. So this can be
    // called with a separate render target (GEVulkanDriver::setRenderTarget)
    // the same way as the default framebuffer
    GEVulkanDriver* vk = static_cast<GEVulkanDriver*>(getVideoDriver());
    m_current_group = NULL;
    m_current_entry = NULL;
    bool paused = vk->getSwapChainImageViews().empty();

    static_cast<GEVulkanMeshCache*>(getMeshCache())->updateCache();
    GEVulkanCameraSceneNode* cam = NULL;
    if (getActiveCamera())
    {
        cam = static_cast<
            GEVulkanCameraSceneNode*>(getActiveCamera());
        // Each drawAll has its own draw call (no matter which camera is used),
        // which is later rendered to the current render target of the driver
        // (default framebuffer or the one of setRenderTarget), this is needed
        // before OnAnimate because of getActiveDrawCall
        if (!paused)
            m_current_entry = acquireEntry(vk);
    }
    OnAnimate(os::Timer::getTime());
    // Mobile STK will destroy swapchain when the main thread is paused
    if (paused)
        return;

    if (cam)
    {
        GEVulkanCamera* camera = &m_current_entry->m_camera;
        GEVulkanDrawCall* dc = &m_current_entry->m_draw_call;
        camera->update(cam);
        dc->prepare(camera);
        GEVulkanShadowFBO* sfbo = dc->getShadowFBO();
        if (sfbo)
            sfbo->prepare(camera, dc->getLightHandler());
        OnRegisterSceneNode();
        dc->generate(vk);
        if (sfbo)
            sfbo->generate();
    }
    m_current_group = NULL;
    m_current_entry = NULL;
}   // drawAll

// ----------------------------------------------------------------------------
irr::u32 GEVulkanSceneManager::registerNodeForRendering(
    irr::scene::ISceneNode* node,
    irr::scene::E_SCENE_NODE_RENDER_PASS pass)
{
    if (!m_current_entry)
        return 0;

    GEVulkanDrawCall* dc = &m_current_entry->m_draw_call;
    if (node->getType() == irr::scene::ESNT_SKY_BOX)
    {
        getVKDriver()->getSkyBoxRenderer()->addSkyBox(node);
        return 1;
    }

    if (node->getType() == irr::scene::ESNT_LIGHT)
    {
        dc->addLightNode(
            static_cast<irr::scene::ILightSceneNode*>(node));
        return 1;
    }

    if (node->getType() == irr::scene::ESNT_BILLBOARD ||
        node->getType() == irr::scene::ESNT_PARTICLE_SYSTEM)
    {
        dc->addBillboardNode(node, node->getType());
        return 1;
    }

    if ((node->getType() == irr::scene::ESNT_ANIMATED_MESH &&
        pass != irr::scene::ESNRP_SOLID) ||
        (node->getType() == irr::scene::ESNT_MESH &&
        pass != irr::scene::ESNRP_SOLID))
        return 0;

    dc->addNode(node);
    GEVulkanShadowFBO* sfbo = dc->getShadowFBO();
    if (sfbo)
        sfbo->addNode(node);
    return 1;
}   // registerNodeForRendering

// ----------------------------------------------------------------------------
void GEVulkanSceneManager::detectDeferred(irr::scene::ISceneNode* node)
{
    if (node->isVisible())
    {
        switch (node->getType())
        {
        case irr::scene::ESNT_LIGHT:
        {
            auto l = static_cast<irr::scene::ILightSceneNode*>(node);
            if (l->getLightType() == irr::video::ELT_POINT)
                m_pointlight_count++;
            else if (l->getLightType() == irr::video::ELT_SPOT)
                m_spotlight_count++;
            break;
        }
        case irr::scene::ESNT_ANIMATED_MESH:
        case irr::scene::ESNT_MESH:
        {
            for (unsigned i = 0; i < node->getMaterialCount(); i++)
            {
                irr::video::SMaterial& m = node->getMaterial(i);
                if (GEMaterialManager::getShader(m.MaterialType) == "displace")
                    m_displace_count++;
            }
            break;
        }
        default:
            break;
        }
    }
    for (unsigned i = 0; i < node->getChildren().size(); i++)
        detectDeferred(node->getChildren()[i]);
}   // detectDeferred

// ----------------------------------------------------------------------------
GEAutoDeferredType GEVulkanSceneManager::getDetectDeferredResult() const
{
    if (m_displace_count > 0)
        return GADT_DISPLACE;
#if defined(TILED_GPU)
    if (m_spotlight_count > 0 ||
        m_pointlight_count > getGEConfig()->m_max_pointlights / 2)
        return GADT_SINGLE_PASS;
#endif
    return GADT_DISABLED;
}   // getDetectDeferredResult

// ----------------------------------------------------------------------------
void GEVulkanSceneManager::detectAutoDeferred()
{
    GEVulkanDriver* gevk = static_cast<GEVulkanDriver*>(getVideoDriver());
    if (gevk->getSeparateRTTTexture())
        return;
    bool prev_deferred = needsDeferredRendering();
    GEAutoDeferredType prev_deferred_type =
        getGEConfig()->m_auto_deferred_type;
    resetDetectDeferred();
    detectDeferred(this);
    getGEConfig()->m_auto_deferred_type = getDetectDeferredResult();
    if (needsDeferredRendering() != prev_deferred ||
        prev_deferred_type != getGEConfig()->m_auto_deferred_type)
        gevk->updateDriver();
}   // detectAutoDeferred

// ----------------------------------------------------------------------------
void GEVulkanSceneManager::endFrame(GEVulkanDriver* vk)
{
    m_current_group = NULL;
    m_current_entry = NULL;
    m_frame_groups.clear();
    // The default target is always kept (its draw calls have created pipelines
    // which are expensive to create on some drivers)
    GEVulkanFBOTexture* default_target = vk->getDefaultRTTTexture();
    auto should_remove = [&](const std::unique_ptr<GEVulkanTargetGroup>& g)
    {
        return (g->m_target && g->m_observer.expired()) ||
            (g->m_target != default_target && g->m_unused_frames >
            GEVulkanDriver::getMaxFrameInFlight() + 1);
    };
    bool need_wait = false;
    for (auto& g : m_groups)
    {
        g->m_unused_frames = g->m_used > 0 ? 0 : g->m_unused_frames + 1;
        g->resetUsed();
        need_wait = need_wait || should_remove(g);
    }
    if (need_wait)
    {
        // Draw calls of removed groups may be used by frames in flight
        vk->waitIdle();
        m_groups.erase(std::remove_if(m_groups.begin(), m_groups.end(),
            should_remove), m_groups.end());
    }
}   // endFrame

// ----------------------------------------------------------------------------
void GEVulkanSceneManager::resetDrawCalls()
{
    m_current_group = NULL;
    m_current_entry = NULL;
    m_frame_groups.clear();
    m_groups.clear();
}   // resetDrawCalls

// ----------------------------------------------------------------------------
irr::scene::ILightSceneNode* GEVulkanSceneManager::getSunNode(
                                                  irr::scene::ISceneNode* node)
{
    if (node->isVisible())
    {
        switch (node->getType())
        {
        case irr::scene::ESNT_LIGHT:
        {
            auto l = static_cast<irr::scene::ILightSceneNode*>(node);
            if (l->getLightType() == irr::video::ELT_DIRECTIONAL)
                return l;
            break;
        }
        default:
            break;
        }
    }
    irr::scene::ILightSceneNode* sun = NULL;
    for (unsigned i = 0; i < node->getChildren().size(); i++)
    {
        sun = getSunNode(node->getChildren()[i]);
        if (sun)
            break;
    }
    return sun;
}   // getSunNode

// ----------------------------------------------------------------------------
irr::scene::ISceneNode* GEVulkanSceneManager::getSkyBoxNode(
                                                  irr::scene::ISceneNode* node)
{
    if (node->isVisible() && node->getType() == irr::scene::ESNT_SKY_BOX)
        return node;
    irr::scene::ISceneNode* sky = NULL;
    for (unsigned i = 0; i < node->getChildren().size(); i++)
    {
        sky = getSkyBoxNode(node->getChildren()[i]);
        if (sky)
            break;
    }
    return sky;
}   // getSkyBoxNode

// ----------------------------------------------------------------------------
GEVulkanDrawCall* GEVulkanSceneManager::getActiveDrawCall() const
{
    // Only valid when drawAll is processing
    return m_current_entry ? &m_current_entry->m_draw_call : NULL;
}   // getActiveDrawCall

}
