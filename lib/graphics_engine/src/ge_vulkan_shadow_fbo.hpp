#ifndef HEADER_GE_VULKAN_SHADOW_FBO_HPP
#define HEADER_GE_VULKAN_SHADOW_FBO_HPP

#include "ge_vulkan_texture.hpp"

#include <array>
#include <memory>

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
    irr::scene::ILightSceneNode* m_sun;

    VkRenderPass m_rtt_render_pass;

    VkFramebuffer m_rtt_frame_buffer;

    // One image view per cascade layer, used as framebuffer attachments
    std::array<VkImageView, GVSCC_COUNT> m_frame_buffer_image_views;

    std::array<std::unique_ptr<GEVulkanShadowDrawCall>, GVSCC_COUNT> m_shadow_draw_calls;

    irr::core::matrix4 m_shadow_view_matrix;

    irr::core::matrix4 m_shadow_projection_matrices[GVSCC_COUNT];

    GEVulkanCameraUBO* m_shadow_camera_ubo_data;

    // ------------------------------------------------------------------------
    static const float getSplitNear(GEVulkanShadowCameraCascade cascade)
    {
        const float cascade_near[GVSCC_COUNT] = { 1.0f, 9.0f, 40.0f };
        return cascade_near[cascade];
    }
    // ------------------------------------------------------------------------
    static const float getSplitFar(GEVulkanShadowCameraCascade cascade)
    {
        const float cascade_far[GVSCC_COUNT] = { 10.0f, 45.0f, 150.0f };
        return cascade_far[cascade];
    }
public:
    // ------------------------------------------------------------------------
    GEVulkanShadowFBO(GEVulkanDriver* vk, unsigned shadow_size,
                      irr::scene::ILightSceneNode* sun = NULL);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanShadowFBO();
    // ------------------------------------------------------------------------
    void createRTT();
    // ------------------------------------------------------------------------
    void createData();
    // ------------------------------------------------------------------------
    VkRenderPass getRTTRenderPass() const         { return m_rtt_render_pass; }
    // ------------------------------------------------------------------------
    VkFramebuffer getRTTFramebuffer() const      { return m_rtt_frame_buffer; }
    // ------------------------------------------------------------------------
    void prepare(irr::scene::ICameraSceneNode* cam, GEVulkanLightHandler* lh);
    // ------------------------------------------------------------------------
    void addNode(irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    void generate();
    // ------------------------------------------------------------------------
    void uploadDynamicData(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    void render(VkCommandBuffer cmd);
    // ------------------------------------------------------------------------
    irr::core::matrix4 getShadowProjectionViewMatrix(GEVulkanShadowCameraCascade cascade) const
    {
        return m_shadow_projection_matrices[cascade] * m_shadow_view_matrix;
    }
};   // GEVulkanShadowFBO

}   // namespace GE

#endif   // HEADER_GE_VULKAN_SHADOW_FBO_HPP
