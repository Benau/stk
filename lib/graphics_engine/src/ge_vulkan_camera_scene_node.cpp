#include "ge_vulkan_camera_scene_node.hpp"

#include "ge_main.hpp"
#include "ge_culling_tool.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_dynamic_buffer.hpp"
#include "ge_vulkan_fbo_texture.hpp"
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
                       : CCameraSceneNode(parent, mgr, id, position, lookat),
                         m_ubo_padding(getPadding(getCameraUBOSize(),
                         getVKDriver()->getPhysicalDeviceProperties().limits.
                         minUniformBufferOffsetAlignment))
{
    static_cast<GEVulkanSceneManager*>(SceneManager)->addDrawCall(this);
    m_camera_ubo = new GEVulkanDynamicBuffer(
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        0/*will be grown later so forceUpdateDataDescriptorSets works*/,
        GEVulkanDriver::getMaxFrameInFlight() + 1,
        GEVulkanDynamicBuffer::supportsHostTransfer() ? 0 :
        GEVulkanDriver::getMaxFrameInFlight() + 1);
    m_camera_ubo_count = 0;
}   // GEVulkanCameraSceneNode

// ----------------------------------------------------------------------------
GEVulkanCameraSceneNode::~GEVulkanCameraSceneNode()
{
    static_cast<GEVulkanSceneManager*>(SceneManager)->removeDrawCall(this);
    delete m_camera_ubo;
}   // ~GEVulkanCameraSceneNode

// ----------------------------------------------------------------------------
void GEVulkanCameraSceneNode::render()
{
    irr::scene::CCameraSceneNode::render();

    m_ubo_data.m_view_matrix = ViewArea.getTransform(irr::video::ETS_VIEW);
    m_ubo_data.m_projection_matrix = m_reverse_z_projection_matrix;
    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    // Vulkan clip space has inverted Y and half Z
    irr::core::matrix4 clip;
    clip[5] = -1.0f;
    // Irrlicht actually produces a [0, 1] Direct3D/Vulkan-style depth range,
    // so the following are unnecessary.
    //clip[10] = 0.5f;
    //clip[14] = 0.5f;
    m_ubo_data.m_projection_matrix = clip * m_ubo_data.m_projection_matrix;
    GEVulkanDriver* vk = getVKDriver();
    if (!vk->getRTTTexture() || vk->getRTTTexture()->useSwapChainOutput())
    {
        m_ubo_data.m_projection_matrix = vk->getPreRotationMatrix() *
            m_ubo_data.m_projection_matrix;
    }

    irr::core::matrix4 mat;
    ViewArea.getTransform(irr::video::ETS_VIEW).getInverse(mat);
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
}   // render

// ----------------------------------------------------------------------------
irr::core::matrix4 GEVulkanCameraSceneNode::getPVM() const
{
    // Use the original unedited matrix for culling
    return ViewArea.getTransform(irr::video::ETS_PROJECTION) *
        ViewArea.getTransform(irr::video::ETS_VIEW);
}   // getPVM

// ----------------------------------------------------------------------------
void GEVulkanCameraSceneNode::collectUBO(GEVulkanDriver* vk,
                                         GEVulkanDrawCall* dc,
                                         VkCommandBuffer cmd)
{
    m_camera_ubo_count = 0;
    unsigned offset = 0;
    std::vector<std::pair<void*, size_t> > data_uploading;

    dc->setCameraUBOOffset(0);
    data_uploading.emplace_back(&m_ubo_data, sizeof(GEVulkanCameraUBO));
    data_uploading.emplace_back(dc->getCullingTool()->getFrustumData(),
        getFrustumSize());
    if (m_ubo_padding > 0)
        data_uploading.emplace_back((void*)NULL, m_ubo_padding);
    offset += getCameraUBOSize() + m_ubo_padding;
    m_camera_ubo_count++;

    GEVulkanShadowFBO* sfbo = dc->getShadowFBO();
    if (sfbo)
    {
        for (unsigned i = 0; i < sfbo->getDrawCallCount(); i++)
        {
            GEVulkanDrawCall* sdc = sfbo->getDrawCall(i);
            sdc->setCameraUBOOffset(offset);
            data_uploading.emplace_back(sfbo->getCameraUBO(i),
                sizeof(GEVulkanCameraUBO));
            data_uploading.emplace_back(
                sdc->getCullingTool()->getFrustumData(), getFrustumSize());
            if (m_ubo_padding > 0)
                data_uploading.emplace_back((void*)NULL, m_ubo_padding);
            offset += getCameraUBOSize() + m_ubo_padding;
            m_camera_ubo_count++;
        }
    }
    if (cmd == VK_NULL_HANDLE)
        cmd = vk->getCurrentCommandBuffer();
    m_camera_ubo->setCurrentData(data_uploading, cmd,
        vk->getCurrentBufferIdx());
}   // collectUBO

}
