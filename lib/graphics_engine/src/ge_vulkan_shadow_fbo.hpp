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
    // No distance limit of its own: the far boundary of its body is the
    // bounding box of the scene
    GVSCC_FARTHEST,
    GVSCC_COUNT
};

class GEVulkanShadowFBO : public GEVulkanTexture
{
protected:
    GEVulkanDrawCall* m_master_dc;

    irr::scene::ILightSceneNode* m_sun;

    VkRenderPass m_rtt_render_pass;

    VkFramebuffer m_rtt_frame_buffer;

    // One image view per cascade layer, used as framebuffer attachments
    std::vector<VkImageView> m_frame_buffer_image_views;

    std::vector<std::unique_ptr<GEVulkanShadowDrawCall> > m_shadow_draw_calls;

    irr::core::matrix4 m_shadow_view_matrix;

    // Union of the transformed bounding boxes of the nodes registered for the
    // shadow pass in the previous frame. The LISPSM projection is calculated
    // from the frustum slice of each cascade alone, so it doesn't move with
    // the scene; the scene clipped box only tightens the shadow map window
    // after the warp. prepare swaps it with m_next_scene_bbox, which is built
    // by the registration of the current frame, so the box follows the scene.
    irr::core::aabbox3df m_scene_bbox;
    bool m_scene_bbox_valid = false;
    irr::core::aabbox3df m_next_scene_bbox;
    bool m_next_scene_bbox_valid = false;

    std::vector<irr::core::matrix4> m_shadow_projection_matrices;

    GEVulkanCameraUBO* m_shadow_camera_ubo_data;

    // Number of draw calls (and camera UBOs), which is the number of array
    // layers except for the omni shadow atlas, whose single layer holds the
    // tiles of all the lights
    unsigned m_slot_count;

    // ------------------------------------------------------------------------
    // The camera distance where the cascade ends (the far boundary of the
    // body is the sphere of this radius). The last cascade has no distance
    // limit, its body is bounded by the scene bounding box in prepare().
    static const float getSplitFar(GEVulkanShadowCameraCascade cascade)
    {
        const float cascade_far[GVSCC_COUNT] =
            { 10.0f, 45.0f, 200.0f, INFINITY };
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
    // slot_count is the number of draw calls and camera UBOs, 0 means one per
    // layer
    GEVulkanShadowFBO(GEVulkanDriver* vk, unsigned shadow_size,
                      GEVulkanDrawCall* master_dc = NULL,
                      irr::scene::ILightSceneNode* sun = NULL,
                      unsigned layer_count = GVSCC_COUNT,
                      unsigned slot_count = 0);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanShadowFBO();
    // ------------------------------------------------------------------------
    void createRTT();
    // ------------------------------------------------------------------------
    virtual void createDrawCalls();
    // ------------------------------------------------------------------------
    // The layout of the image before the shadow pass. The omni atlas keeps
    // its content over the frames and is transitioned to the read only layout
    // at creation, the sun cascades are always cleared so they start from
    // undefined
    virtual VkImageLayout getShadowInitialLayout() const
    {
        return VK_IMAGE_LAYOUT_UNDEFINED;
    }
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
    // ------------------------------------------------------------------------
    unsigned getSlotCount() const                                  { return m_slot_count; }

protected:
    // ------------------------------------------------------------------------
    // Binds the data of one shadow draw call (borrowing the geometry of the
    // owner of its bucket) and submits it to the currently bound depth
    // target, it is used by render() of the subclasses
    void renderShadowDrawCall(VkCommandBuffer cmd, unsigned i,
                              GEVulkanCamera* cam, bool& rebind_base_vertex);
    // ------------------------------------------------------------------------
    void updateShadowDataDescriptorSets(GEVulkanCamera* cam);

};   // GEVulkanShadowFBO

}   // namespace GE

#endif   // HEADER_GE_VULKAN_SHADOW_FBO_HPP
