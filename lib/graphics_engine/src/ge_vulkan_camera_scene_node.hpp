#ifndef HEADER_GE_VULKAN_CAMERA_SCENE_NODE_HPP
#define HEADER_GE_VULKAN_CAMERA_SCENE_NODE_HPP

#include "../source/Irrlicht/CCameraSceneNode.h"
#include "SViewFrustum.h"
#include "vulkan_wrapper.h"

#include <array>
#include <cstdio>
#include <memory>

namespace GE
{
struct GEVulkanCameraUBO
{
irr::core::matrix4 m_view_matrix;
irr::core::matrix4 m_projection_matrix;
irr::core::matrix4 m_inverse_view_matrix;
irr::core::matrix4 m_inverse_projection_matrix;
irr::core::matrix4 m_projection_view_matrix;
irr::core::matrix4 m_inverse_projection_view_matrix;
irr::core::rectf   m_viewport;
irr::core::rectf   m_screensize;
};

class GEVulkanDrawCall;
class GEVulkanDriver;
class GEVulkanDynamicBuffer;

// The scene node that is created by the scene manager (STK and irrlicht need an
// ICameraSceneNode, and the reverse Z projection is kept here). It has no draw
// call, every GEVulkanSceneManager::drawAll() takes a snapshot of it in a
// GEVulkanCamera, so the same camera node can be used to render different
// targets (default framebuffer, separate RTT) in the same frame
class GEVulkanCameraSceneNode : public irr::scene::CCameraSceneNode
{
private:
    irr::core::matrix4 m_reverse_z_projection_matrix;

    irr::core::rect<irr::s32> m_viewport;
public:
    // ------------------------------------------------------------------------
    GEVulkanCameraSceneNode(irr::scene::ISceneNode* parent,
                            irr::scene::ISceneManager* mgr, irr::s32 id,
          const irr::core::vector3df& position = irr::core::vector3df(0, 0, 0),
         const irr::core::vector3df& lookat = irr::core::vector3df(0, 0, 100));
    // ------------------------------------------------------------------------
    ~GEVulkanCameraSceneNode() {}
    // ------------------------------------------------------------------------
    void setViewPort(const irr::core::rect<irr::s32>& area)
                                                         { m_viewport = area; }
    // ------------------------------------------------------------------------
    const irr::core::rect<irr::s32>& getViewPort() const { return m_viewport; }
    // ------------------------------------------------------------------------
    const irr::core::matrix4& getReverseZProjectionMatrix() const
                                      { return m_reverse_z_projection_matrix; }
    // ------------------------------------------------------------------------
    virtual void recalculateProjectionMatrix()
    {
        CCameraSceneNode::recalculateProjectionMatrix();
        if (IsOrthogonal)
        {
            m_reverse_z_projection_matrix.buildProjectionMatrixOrthoLH(
                Fovy, Fovy, ZFar, ZNear);
        }
        else
        {
            m_reverse_z_projection_matrix.buildProjectionMatrixPerspectiveFovLH(
                Fovy, Aspect, ZFar, ZNear);
        }
    }
    // ------------------------------------------------------------------------
    virtual void setProjectionMatrix(const irr::core::matrix4& projection,
                                     bool isOrthogonal)
    {
        CCameraSceneNode::setProjectionMatrix(projection, isOrthogonal);
        printf("Calling setProjectionMatrix directly will ignore reverse Z\n");
    }
};   // GEVulkanCameraSceneNode

// A snapshot of a GEVulkanCameraSceneNode which is taken every time
// GEVulkanSceneManager::drawAll() is called. It is used (together with a
// GEVulkanDrawCall) to render later in GEVulkanDriver::buildCommandBuffers(),
// so the node can change or be removed after drawAll()
class GEVulkanCamera
{
private:
    GEVulkanCameraUBO m_ubo_data;

    irr::scene::SViewFrustum m_view_area;

    irr::core::vector3df m_absolute_position;

    irr::core::vector3df m_target;

    irr::f32 m_near, m_far;

    irr::core::rect<irr::s32> m_viewport;

    GEVulkanDynamicBuffer* m_camera_ubo;

    unsigned m_camera_ubo_count;

    const unsigned m_ubo_padding;
    // ------------------------------------------------------------------------
    static const unsigned getFrustumSize()    { return sizeof(float) * 4 * 6; }
public:
    // ------------------------------------------------------------------------
    static const unsigned getCameraUBOSize()
    {
#ifdef GPU_CULLING
        const unsigned frustum_size = getFrustumSize();
#else
        const unsigned frustum_size = 0;
#endif
        return sizeof(GEVulkanCameraUBO) + frustum_size;
    }
    // ------------------------------------------------------------------------
    GEVulkanCamera();
    // ------------------------------------------------------------------------
    ~GEVulkanCamera();
    // ------------------------------------------------------------------------
    // Copy the state of the node, and calculate the uniform data for the
    // current render target (see GEVulkanDriver::getRTTTexture), so this has
    // to be called when drawAll() is called (not at the time of rendering)
    void update(GEVulkanCameraSceneNode* node);
    // ------------------------------------------------------------------------
    const irr::core::rect<irr::s32>& getViewPort() const { return m_viewport; }
    // ------------------------------------------------------------------------
    irr::core::matrix4 getPVM() const;
    // ------------------------------------------------------------------------
    const GEVulkanCameraUBO* const getUBOData() const   { return &m_ubo_data; }
    // ------------------------------------------------------------------------
    const irr::scene::SViewFrustum* getViewFrustum() const
                                                       { return &m_view_area; }
    // ------------------------------------------------------------------------
    const irr::core::matrix4& getViewMatrix() const
                     { return m_view_area.getTransform(irr::video::ETS_VIEW); }
    // ------------------------------------------------------------------------
    const irr::core::vector3df& getAbsolutePosition() const
                                                { return m_absolute_position; }
    // ------------------------------------------------------------------------
    const irr::core::vector3df& getTarget() const          { return m_target; }
    // ------------------------------------------------------------------------
    irr::f32 getNearValue() const                            { return m_near; }
    // ------------------------------------------------------------------------
    irr::f32 getFarValue() const                              { return m_far; }
    // ------------------------------------------------------------------------
    void collectUBO(GEVulkanDriver* vk, GEVulkanDrawCall* dc,
                    VkCommandBuffer cmd = VK_NULL_HANDLE);
    // ------------------------------------------------------------------------
    unsigned getCameraUBOCount() const           { return m_camera_ubo_count; }
    // ------------------------------------------------------------------------
    GEVulkanDynamicBuffer* getCameraUBO() const        { return m_camera_ubo; }
    // ------------------------------------------------------------------------
    unsigned getPaddedCameraUBOSize() const
                                 { return getCameraUBOSize() + m_ubo_padding; }
};   // GEVulkanCamera

}

#endif
