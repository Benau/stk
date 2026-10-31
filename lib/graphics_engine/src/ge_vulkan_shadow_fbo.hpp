#ifndef HEADER_GE_VULKAN_SHADOW_FBO_HPP
#define HEADER_GE_VULKAN_SHADOW_FBO_HPP

#include "ge_vulkan_texture.hpp"

#include "aabbox3d.h"

#include <memory>
#include <vector>

namespace irr
{
    namespace scene
    {
        class ICameraSceneNode;
        class ILightSceneNode;
        class ISceneNode;
    }
}

namespace GE
{

class GEVulkanDriver;
class GEVulkanLightHandler;
class GEVulkanDrawCall;
class GEVulkanCamera;
class GEVulkanShadowDrawCall;
struct GEVulkanCameraUBO;

enum GEVulkanShadowCameraCascade : unsigned
{
    GVSCC_NEAR = 0,
    GVSCC_MIDDLE,
    GVSCC_FAR,
    GVSCC_COUNT
};

class GEVulkanShadowFBO : public GEVulkanTexture
{
private:
    GEVulkanDrawCall* m_master_dc;

protected:
    irr::scene::ILightSceneNode* m_sun;

    VkRenderPass m_rtt_render_pass;

    VkFramebuffer m_rtt_frame_buffer;

    // One image view per cascade layer, used as framebuffer attachments
    std::vector<VkImageView> m_frame_buffer_image_views;

    std::vector<std::unique_ptr<GEVulkanShadowDrawCall> > m_shadow_draw_calls;

    irr::core::matrix4 m_shadow_view_matrix;

    // Union of the transformed bounding boxes of the nodes registered for the
    // shadow pass, used to clip the cascade bodies to the scene. It only
    // grows so the cascade bounds don't change with the culled nodes.
    irr::core::aabbox3df m_scene_bbox;
    bool m_scene_bbox_valid = false;

    std::vector<irr::core::matrix4> m_shadow_projection_matrices;

    GEVulkanCameraUBO* m_shadow_camera_ubo_data;

    // ------------------------------------------------------------------------
    // The camera distance where the cascade ends (the far boundary of the
    // body is the sphere of this radius)
    static const float getSplitFar(GEVulkanShadowCameraCascade cascade)
    {
        const float cascade_far[GVSCC_COUNT] = { 10.0f, 45.0f, 200.0f };
        return cascade_far[cascade];
    }
    // ------------------------------------------------------------------------
    // The view depth where the cascade starts (the near boundary of the body
    // is the plane of constant view depth): the previous cascade is the
    // largest one which still covers the whole screen there, its far sphere
    // of radius split_far reaches the distance view_depth / min_ray_cos, so
    // view_depth = split_far * min_ray_cos. The first cascade starts at the
    // near plane of the camera, min_ray_cos is the cosine of the largest
    // angle between the view direction and a ray of the frustum
    static const float getSplitNear(GEVulkanShadowCameraCascade cascade,
                float cam_near, float min_ray_cos)
    {
        if (cascade == GVSCC_NEAR)
            return cam_near;
        return getSplitFar((GEVulkanShadowCameraCascade)(cascade - 1)) *
            min_ray_cos;
    }
public:
    // ------------------------------------------------------------------------
    GEVulkanShadowFBO(GEVulkanDriver* vk, unsigned shadow_size,
                      GEVulkanDrawCall* master_dc = NULL,
                      irr::scene::ILightSceneNode* sun = NULL,
                      unsigned layer_count = GVSCC_COUNT);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanShadowFBO();
    // ------------------------------------------------------------------------
    void createRTT();
    // ------------------------------------------------------------------------
    virtual void createDrawCalls();
    // ------------------------------------------------------------------------
    VkRenderPass getRTTRenderPass() const         { return m_rtt_render_pass; }
    // ------------------------------------------------------------------------
    VkFramebuffer getRTTFramebuffer() const      { return m_rtt_frame_buffer; }
    // ------------------------------------------------------------------------
    virtual void prepare(GEVulkanCamera* cam,
                         GEVulkanLightHandler* lh);
    // ------------------------------------------------------------------------
    virtual void addNode(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    virtual void generate();
    // ------------------------------------------------------------------------
    void uploadDynamicData(VkCommandBuffer cmd, bool& has_indirect);
    // ------------------------------------------------------------------------
    void render(VkCommandBuffer cmd, GEVulkanCamera* cam);
    // ------------------------------------------------------------------------
    virtual irr::core::matrix4 getShadowProjectionViewMatrix(
                                                          unsigned layer) const
    {
        return m_shadow_projection_matrices[layer] * m_shadow_view_matrix;
    }
    // ------------------------------------------------------------------------
    GEVulkanCameraUBO* getCameraUBO(unsigned i) const;
    // ------------------------------------------------------------------------
    GEVulkanDrawCall* getDrawCall(unsigned i) const;
    // ------------------------------------------------------------------------
    unsigned getDrawCallCount() const    { return m_shadow_draw_calls.size(); }
    // ------------------------------------------------------------------------
    GEVulkanDrawCall* getMasterDrawCall() const         { return m_master_dc; }
    // ------------------------------------------------------------------------
    virtual unsigned getLayerOffset() const                       { return 0; }
    // ------------------------------------------------------------------------
    // The number of draw calls that share a generated draw call with different
    // camera UBO offsets.
    virtual unsigned getSharingDrawCallCount() const              { return 0; }
    // ------------------------------------------------------------------------
    virtual int getLayerOwner(int pointlight_layer) const        { return -1; }

};   // GEVulkanShadowFBO

}   // namespace GE

#endif   // HEADER_GE_VULKAN_SHADOW_FBO_HPP
