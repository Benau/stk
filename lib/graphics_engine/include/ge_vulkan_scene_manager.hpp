#ifndef HEADER_GE_VULKAN_SCENE_MANAGER_HPP
#define HEADER_GE_VULKAN_SCENE_MANAGER_HPP

#include "../source/Irrlicht/CSceneManager.h"
#include <memory>
#include <vector>

namespace GE
{
class GEVulkanCamera;
class GEVulkanCameraSceneNode;
class GEVulkanDrawCall;
class GEVulkanDriver;
enum GEAutoDeferredType : unsigned;

struct GEVulkanDrawEntry;
struct GEVulkanTargetGroup;

class GEVulkanSceneManager : public irr::scene::CSceneManager
{
private:
    unsigned m_pointlight_count, m_spotlight_count, m_displace_count;

    std::vector<std::unique_ptr<GEVulkanTargetGroup> > m_groups;

    // Groups which are used in this frame, in the order of the first drawAll()
    std::vector<GEVulkanTargetGroup*> m_frame_groups;

    GEVulkanTargetGroup* m_current_group;

    GEVulkanDrawEntry* m_current_entry;

    // ------------------------------------------------------------------------
    GEVulkanDrawEntry* acquireEntry(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    void resetDetectDeferred()
    {
        m_pointlight_count = m_spotlight_count = m_displace_count = 0;
    }
    // ------------------------------------------------------------------------
    GEAutoDeferredType getDetectDeferredResult() const;
    // ------------------------------------------------------------------------
    void detectDeferred(irr::scene::ISceneNode* node);
public:
    // ------------------------------------------------------------------------
    GEVulkanSceneManager(irr::video::IVideoDriver* driver,
                         irr::io::IFileSystem* fs,
                         irr::gui::ICursorControl* cursor_control,
                         irr::gui::IGUIEnvironment* gui_environment);
    // ------------------------------------------------------------------------
    ~GEVulkanSceneManager();
    // ------------------------------------------------------------------------
    virtual irr::scene::ICameraSceneNode* addCameraSceneNode(
        irr::scene::ISceneNode* parent = 0,
        const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& lookat = irr::core::vector3df(0, 0, 100),
        irr::s32 id = -1, bool make_active = true);
    // ------------------------------------------------------------------------
    virtual irr::scene::IAnimatedMeshSceneNode* addAnimatedMeshSceneNode(
        irr::scene::IAnimatedMesh* mesh, irr::scene::ISceneNode* parent = NULL,
        irr::s32 id = -1,
        const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& rotation = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& scale = irr::core::vector3df(1.0f, 1.0f, 1.0f),
        bool alsoAddIfMeshPointerZero = false);
    // ------------------------------------------------------------------------
    virtual irr::scene::IMeshSceneNode* addMeshSceneNode(irr::scene::IMesh* mesh,
        irr::scene::ISceneNode* parent = NULL, irr::s32 id = -1,
        const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& rotation = irr::core::vector3df(0, 0, 0),
        const irr::core::vector3df& scale = irr::core::vector3df(1.0f, 1.0f, 1.0f),
        bool alsoAddIfMeshPointerZero = false);
    // ------------------------------------------------------------------------
    virtual void clear();
    // ------------------------------------------------------------------------
    virtual void drawAll(irr::u32 flags = 0xFFFFFFFF);
    // ------------------------------------------------------------------------
    virtual irr::u32 registerNodeForRendering(irr::scene::ISceneNode* node,
        irr::scene::E_SCENE_NODE_RENDER_PASS pass = irr::scene::ESNRP_AUTOMATIC);
    // ------------------------------------------------------------------------
    // Check if the (GEConfig) deferred type should be changed by the nodes of
    // the scene, it's called when a camera is created
    void detectAutoDeferred();
    // ------------------------------------------------------------------------
    // Groups which have draw calls in the current frame, GEVulkanDriver
    // renders them (the group of the default framebuffer is the last one) in
    // buildCommandBuffers
    std::vector<GEVulkanTargetGroup*>& getFrameGroups()
                                                     { return m_frame_groups; }
    // ------------------------------------------------------------------------
    // Called at the end of every frame by GEVulkanDriver, groups which are not
    // used for long or have deleted targets will be removed
    void endFrame(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    // Destroy all draw calls, they will be created again when needed
    void resetDrawCalls();
    // ------------------------------------------------------------------------
    irr::scene::ILightSceneNode* getSunNode(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    irr::scene::ISceneNode* getSkyBoxNode(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    GEVulkanDrawCall* getActiveDrawCall() const;
};   // GEVulkanSceneManager

}

#endif
