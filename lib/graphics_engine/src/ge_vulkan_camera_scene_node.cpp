#include "ge_vulkan_camera_scene_node.hpp"

#include "ge_main.hpp"
#include "ge_culling_tool.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_dynamic_buffer.hpp"
#include "ge_vulkan_fbo_texture.hpp"
#include "ge_vulkan_light_handler.hpp"
#include "ge_vulkan_scene_manager.hpp"
#include "ge_vulkan_shadow_fbo.hpp"

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanCameraSceneNode::GEVulkanCameraSceneNode(irr::scene::ISceneNode* parent,
                                                 irr::scene::ISceneManager* mgr,
                                                 irr::s32 id,
                                           const irr::core::vector3df& position,
                                             const irr::core::vector3df& lookat)
                       : CCameraSceneNode(parent, mgr, id, position, lookat)
{
    static_cast<GEVulkanSceneManager*>(SceneManager)->detectAutoDeferred();
}   // GEVulkanCameraSceneNode

// ----------------------------------------------------------------------------
GEVulkanCamera::GEVulkanCamera()
              : m_near(0.0f), m_far(0.0f),
                m_ubo_padding(getPadding(getCameraUBOSize(),
                getVKDriver()->getPhysicalDeviceProperties().limits.
                minUniformBufferOffsetAlignment))
{
    m_camera_ubo = new GEVulkanDynamicBuffer(
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        0/*will be grown later so forceUpdateDataDescriptorSets works*/,
        GEVulkanDriver::getMaxFrameInFlight() + 1,
        GEVulkanDynamicBuffer::supportsHostTransfer() ? 0 :
        GEVulkanDriver::getMaxFrameInFlight() + 1);
    m_camera_ubo_count = 0;
}   // GEVulkanCamera

// ----------------------------------------------------------------------------
GEVulkanCamera::~GEVulkanCamera()
{
    delete m_camera_ubo;
}   // ~GEVulkanCamera

// ----------------------------------------------------------------------------
void GEVulkanCamera::update(GEVulkanCameraSceneNode* node)
{
    // Updates the view frustum of the node
    node->render();
    m_view_area = *node->getViewFrustum();
    m_absolute_position = node->getAbsolutePosition();
    m_target = node->getTarget();
    m_near = node->getNearValue();
    m_far = node->getFarValue();
    GEVulkanDriver* vk = getVKDriver();
    m_viewport = node->getViewPort();
    if (m_viewport.getWidth() <= 0 || m_viewport.getHeight() <= 0)
    {
        // The viewport of the node is never set (like a camera which is only
        // used for a separate RTT), so use the whole target
        irr::core::dimension2d<irr::u32> size =
            vk->getSeparateRTTTexture() ?
            vk->getSeparateRTTTexture()->getSize() :
            vk->getCurrentRenderTargetSize();
        m_viewport = irr::core::rect<irr::s32>(0, 0, size.Width, size.Height);
    }

    m_ubo_data.m_view_matrix = m_view_area.getTransform(irr::video::ETS_VIEW);
    m_ubo_data.m_projection_matrix = node->getReverseZProjectionMatrix();
    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    // Vulkan clip space has inverted Y and half Z
    irr::core::matrix4 clip;
    clip[5] = -1.0f;
    // Irrlicht actually produces a [0, 1] Direct3D/Vulkan-style depth range,
    // so the following are unnecessary.
    //clip[10] = 0.5f;
    //clip[14] = 0.5f;
    m_ubo_data.m_projection_matrix = clip * m_ubo_data.m_projection_matrix;
    if (!vk->getRTTTexture() || vk->getRTTTexture()->useSwapChainOutput())
    {
        m_ubo_data.m_projection_matrix = vk->getPreRotationMatrix() *
            m_ubo_data.m_projection_matrix;
    }

    irr::core::matrix4 mat;
    m_view_area.getTransform(irr::video::ETS_VIEW).getInverse(mat);
    m_ubo_data.m_inverse_view_matrix = mat;

    m_ubo_data.m_projection_matrix.getInverse(mat);
    m_ubo_data.m_inverse_projection_matrix = mat;

    mat = m_ubo_data.m_projection_matrix * m_ubo_data.m_view_matrix;

    m_ubo_data.m_projection_view_matrix = mat;

    m_ubo_data.m_projection_view_matrix.getInverse(
        m_ubo_data.m_inverse_projection_view_matrix);

    VkViewport vp = {};
    float scale = getGEConfig()->m_render_scale;
    if (vk->getSeparateRTTTexture())
        scale = 1.0f;
    vp.x = m_viewport.UpperLeftCorner.X * scale;
    vp.y = m_viewport.UpperLeftCorner.Y * scale;
    vp.width = m_viewport.getWidth() * scale;
    vp.height = m_viewport.getHeight() * scale;
    vk->getRotatedViewport(&vp, true/*handle_rtt*/);

    m_ubo_data.m_viewport.UpperLeftCorner.X = vp.x;
    m_ubo_data.m_viewport.UpperLeftCorner.Y = vp.y;
    m_ubo_data.m_viewport.LowerRightCorner.X = vp.width;
    m_ubo_data.m_viewport.LowerRightCorner.Y = vp.height;

    if (!vk->getRTTTexture() || vk->getRTTTexture()->useSwapChainOutput())
    {
        vp.x = vp.y = 0.0f;
        vp.width = vk->getCurrentRenderTargetSize().Width;
        vp.height = vk->getCurrentRenderTargetSize().Height;
        vk->getRotatedViewport(&vp, true/*handle_rtt*/);
        m_ubo_data.m_screensize.UpperLeftCorner.X = vp.width;
        m_ubo_data.m_screensize.UpperLeftCorner.Y = vp.height;
        m_ubo_data.m_screensize.LowerRightCorner.X = 0.0;
        m_ubo_data.m_screensize.LowerRightCorner.Y = 0.0;
    }
    else
    {
        m_ubo_data.m_screensize.UpperLeftCorner.X = vk->getRTTTexture()->getSize().Width;
        m_ubo_data.m_screensize.UpperLeftCorner.Y = vk->getRTTTexture()->getSize().Height;
        m_ubo_data.m_screensize.LowerRightCorner.X = 0.0;
        m_ubo_data.m_screensize.LowerRightCorner.Y = 0.0;
    }
}   // update

// ----------------------------------------------------------------------------
irr::core::matrix4 GEVulkanCamera::getPVM() const
{
    // Use the original unedited matrix for culling
    return m_view_area.getTransform(irr::video::ETS_PROJECTION) *
        m_view_area.getTransform(irr::video::ETS_VIEW);
}   // getPVM

// ----------------------------------------------------------------------------
void GEVulkanCamera::collectUBO(GEVulkanDriver* vk,
                                         GEVulkanDrawCall* dc,
                                         VkCommandBuffer cmd)
{
    std::vector<std::pair<void*, size_t> > data_uploading;

    GEVulkanLightHandler* lh = dc->getLightHandler();
    GEVulkanShadowFBO* sfbo = dc->getShadowFBO();

    unsigned max_size = getPaddedCameraUBOSize();
    m_camera_ubo_count = 1;

    if (lh)
        max_size += GEVulkanLightHandler::getMaxSize();
    if (sfbo)
    {
        m_camera_ubo_count += sfbo->getDrawCallCount();
        max_size += getPaddedCameraUBOSize() * sfbo->getDrawCallCount();
    }
    m_camera_ubo->resizeIfNeeded(max_size);

    unsigned offset = 0;
    dc->setCameraUBOOffset(offset);
    data_uploading.emplace_back(&m_ubo_data, sizeof(GEVulkanCameraUBO));
#ifdef GPU_CULLING
    data_uploading.emplace_back(dc->getCullingTool()->getFrustumData(),
        getFrustumSize());
#endif
    if (m_ubo_padding > 0)
        data_uploading.emplace_back((void*)NULL, m_ubo_padding);
    offset += getPaddedCameraUBOSize();

    if (sfbo)
    {
        for (unsigned i = 0; i < sfbo->getDrawCallCount(); i++)
        {
            GEVulkanDrawCall* sdc = sfbo->getDrawCall(i);
            sdc->setCameraUBOOffset(offset);
            data_uploading.emplace_back(sfbo->getCameraUBO(i),
                sizeof(GEVulkanCameraUBO));
#ifdef GPU_CULLING
            data_uploading.emplace_back(
                sdc->getCullingTool()->getFrustumData(), getFrustumSize());
#endif
            if (m_ubo_padding > 0)
                data_uploading.emplace_back((void*)NULL, m_ubo_padding);
            offset += getPaddedCameraUBOSize();
        }
    }

    if (lh)
    {
        data_uploading.emplace_back(lh->getData(), lh->getSize());
        dc->setLightDataOffset(offset);
    }
    else
        dc->setLightDataOffset(0);

    if (cmd == VK_NULL_HANDLE)
        cmd = vk->getCurrentCommandBuffer();
    m_camera_ubo->setCurrentData(data_uploading, cmd,
        vk->getCurrentBufferIdx());
}   // collectUBO

}
