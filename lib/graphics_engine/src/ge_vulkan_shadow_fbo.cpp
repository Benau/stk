#include "ge_vulkan_shadow_fbo.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_light_handler.hpp"
#include "ge_vulkan_shadow_draw_call.hpp"

#include "ILightSceneNode.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanShadowFBO::GEVulkanShadowFBO(GEVulkanDriver* vk, unsigned shadow_size,
                                     GEVulkanDrawCall* master_dc,
                                     irr::scene::ILightSceneNode* sun,
                                     unsigned layer_count)
                 : GEVulkanTexture(), m_master_dc(master_dc), m_sun(sun),
                   m_rtt_render_pass(VK_NULL_HANDLE),
                   m_rtt_frame_buffer(VK_NULL_HANDLE),
                   m_frame_buffer_image_views({}),
                   m_shadow_camera_ubo_data(NULL)
{
    m_vk             = vk;
    m_vulkan_device  = m_vk->getDevice();
    m_image          = VK_NULL_HANDLE;
    m_vma_allocation = VK_NULL_HANDLE;
    m_has_mipmaps    = false;
    m_locked_data    = NULL;
    m_layer_count    = layer_count;
    m_image_view_type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;

    m_size = m_orig_size = irr::core::dimension2du(shadow_size, shadow_size);
    // Prefer D16 for bandwidth / fillrate; fall back to wider formats.
    std::vector<VkFormat> preferred =
    {
        VK_FORMAT_D16_UNORM,
        VK_FORMAT_D32_SFLOAT,
        VK_FORMAT_D24_UNORM_S8_UINT,
        VK_FORMAT_D32_SFLOAT_S8_UINT
    };
    m_internal_format = m_vk->findSupportedFormat(preferred,
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);

    if (!createImage(VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT))
        throw std::runtime_error("createImage failed for shadow map depth texture");

    if (!createImageView(VK_IMAGE_ASPECT_DEPTH_BIT))
        throw std::runtime_error("createImageView failed for shadow map depth texture");

    if (m_sun)
        m_sun->grab();
    else
    {
        VkCommandBuffer command_buffer =
            GEVulkanCommandLoader::beginSingleTimeCommands();
        transitionImageLayout(
            command_buffer, VK_IMAGE_LAYOUT_UNDEFINED,
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
        GEVulkanCommandLoader::endSingleTimeCommands(command_buffer);
    }
}   // GEVulkanShadowFBO

// ----------------------------------------------------------------------------
GEVulkanShadowFBO::~GEVulkanShadowFBO()
{
    delete [] m_shadow_camera_ubo_data;
    if (m_sun)
        m_sun->drop();
    // Destroy per-layer image views used as framebuffer attachments FIRST
    for (VkImageView view : m_frame_buffer_image_views)
    {
        if (view != VK_NULL_HANDLE)
            vkDestroyImageView(m_vulkan_device, view, NULL);
    }

    if (m_rtt_frame_buffer != VK_NULL_HANDLE)
    {
        vkDestroyFramebuffer(m_vk->getDevice(), m_rtt_frame_buffer, NULL);
        m_rtt_frame_buffer = VK_NULL_HANDLE;
    }

    if (m_rtt_render_pass != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(m_vk->getDevice(), m_rtt_render_pass, NULL);
        m_rtt_render_pass = VK_NULL_HANDLE;
    }
}   // ~GEVulkanShadowFBO

// ----------------------------------------------------------------------------
void GEVulkanShadowFBO::createRTT()
{
    // -------------------------------------------------------------------- //
    // Per-layer 2-D image views for use as framebuffer attachments         //
    // (and as VkRenderingAttachmentInfo::imageView for dynamic rendering). //
    // The main image view covers all layers as an array.                   //
    // -------------------------------------------------------------------- //
    m_frame_buffer_image_views.resize(m_layer_count, VK_NULL_HANDLE);
    for (unsigned i = 0; i < m_layer_count; i++)
    {
        VkImageViewCreateInfo view_info = {};
        view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image                           = m_image;
        view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format                          = m_internal_format;
        view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
        view_info.subresourceRange.baseMipLevel   = 0;
        view_info.subresourceRange.levelCount     = getMipmapLevels();
        view_info.subresourceRange.baseArrayLayer = i;
        view_info.subresourceRange.layerCount     = 1;

        VkResult result = vkCreateImageView(m_vulkan_device, &view_info, NULL,
            &m_frame_buffer_image_views[i]);
        if (result != VK_SUCCESS)
            throw std::runtime_error("GEVulkanShadowFBO: vkCreateImageView failed for layer");
    }

    m_shadow_projection_matrices.resize(m_layer_count);
    m_shadow_camera_ubo_data = new GEVulkanCameraUBO[m_layer_count];

    // When VK_KHR_dynamic_rendering is available we skip the traditional
    // render pass and framebuffer entirely — render() will drive individual
    // vkCmdBeginRendering calls per layer instead.
    if (GEVulkanFeatures::supportsDynamicRendering())
        return;

    // ------------------------------------------------------------------ //
    // One VkAttachmentDescription per cascade.                           //
    // ------------------------------------------------------------------ //
    std::vector<VkAttachmentDescription> shadow_attachments(m_layer_count);
    for (unsigned i = 0; i < m_layer_count; i++)
    {
        shadow_attachments[i]                 = {};
        shadow_attachments[i].format          = m_internal_format;
        shadow_attachments[i].samples         = VK_SAMPLE_COUNT_1_BIT;
        shadow_attachments[i].loadOp          = VK_ATTACHMENT_LOAD_OP_CLEAR;
        shadow_attachments[i].storeOp         = VK_ATTACHMENT_STORE_OP_STORE;
        shadow_attachments[i].stencilLoadOp   = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        shadow_attachments[i].stencilStoreOp  = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        shadow_attachments[i].initialLayout   = VK_IMAGE_LAYOUT_UNDEFINED;
        // Transition to SHADER_READ_ONLY so sampling works after the pass.
        shadow_attachments[i].finalLayout     =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    }

    // ------------------------------------------------------------------ //
    // One VkAttachmentReference per cascade, one subpass per cascade.    //
    // ------------------------------------------------------------------ //
    std::vector<VkAttachmentReference> shadow_attachment_refs(m_layer_count);
    for (unsigned i = 0; i < m_layer_count; i++)
    {
        shadow_attachment_refs[i]            = {};
        shadow_attachment_refs[i].attachment = i;
        shadow_attachment_refs[i].layout     =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    }

    std::vector<VkSubpassDescription> subpasses(m_layer_count);
    for (unsigned i = 0; i < m_layer_count; i++)
    {
        subpasses[i]                         = {};
        subpasses[i].pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpasses[i].colorAttachmentCount    = 0;
        subpasses[i].pColorAttachments       = NULL;
        subpasses[i].pDepthStencilAttachment = &shadow_attachment_refs[i];
    }

    // ------------------------------------------------------------------ //
    // Two dependencies per subpass (enter + exit).                       //
    // ------------------------------------------------------------------ //
    std::vector<VkSubpassDependency> dependencies(m_layer_count * 2);
    for (unsigned i = 0; i < m_layer_count; i++)
    {
        // External -> subpass i  (wait for previous shader read)
        dependencies[2 * i]                 = {};
        dependencies[2 * i].srcSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[2 * i].dstSubpass      = i;
        dependencies[2 * i].srcStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[2 * i].dstStageMask    = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependencies[2 * i].srcAccessMask   = VK_ACCESS_SHADER_READ_BIT;
        dependencies[2 * i].dstAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[2 * i].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

        // subpass i -> External  (make depth visible to shader reads)
        dependencies[2 * i + 1]                 = {};
        dependencies[2 * i + 1].srcSubpass      = i;
        dependencies[2 * i + 1].dstSubpass      = VK_SUBPASS_EXTERNAL;
        dependencies[2 * i + 1].srcStageMask    = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependencies[2 * i + 1].dstStageMask    = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dependencies[2 * i + 1].srcAccessMask   = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependencies[2 * i + 1].dstAccessMask   = VK_ACCESS_SHADER_READ_BIT;
        dependencies[2 * i + 1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
    }

    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = (uint32_t)shadow_attachments.size();
    render_pass_info.pAttachments    = shadow_attachments.data();
    render_pass_info.subpassCount    = (uint32_t)subpasses.size();
    render_pass_info.pSubpasses      = subpasses.data();
    render_pass_info.dependencyCount = (uint32_t)dependencies.size();
    render_pass_info.pDependencies   = dependencies.data();

    VkResult result = vkCreateRenderPass(m_vk->getDevice(), &render_pass_info,
                                         NULL, &m_rtt_render_pass);
    if (result != VK_SUCCESS)
        throw std::runtime_error("GEVulkanShadowFBO: vkCreateRenderPass failed");

    // ------------------------------------------------------------------ //
    // Single framebuffer that references all layer views.                //
    // ------------------------------------------------------------------ //
    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass      = m_rtt_render_pass;
    framebuffer_info.attachmentCount = (uint32_t)m_frame_buffer_image_views.size();
    framebuffer_info.pAttachments    = m_frame_buffer_image_views.data();
    framebuffer_info.width           = m_size.Width;
    framebuffer_info.height          = m_size.Height;
    framebuffer_info.layers          = 1;

    result = vkCreateFramebuffer(m_vk->getDevice(), &framebuffer_info,
                                 NULL, &m_rtt_frame_buffer);
    if (result != VK_SUCCESS)
        throw std::runtime_error("GEVulkanShadowFBO: vkCreateFramebuffer failed");
}   // createRTT

// ----------------------------------------------------------------------------
void GEVulkanShadowFBO::createDrawCalls()
{
    for (unsigned i = 0; i < m_layer_count; i++)
    {
        m_shadow_draw_calls.push_back(std::unique_ptr<GEVulkanShadowDrawCall>(
            new GEVulkanShadowDrawCall(this, (GEVulkanShadowCameraCascade)i)));
    }
}   // createDrawCalls

// ----------------------------------------------------------------------------
void GEVulkanShadowFBO::prepare(GEVulkanCamera* cam,
                                GEVulkanLightHandler* lh)
{
    irr::core::vector3df eyepos = cam->getAbsolutePosition();
    irr::core::vector3df viewdir = cam->getTarget() - eyepos;
    irr::core::vector3df lightdir = -m_sun->getPosition();
    viewdir = viewdir.normalize();
    lightdir = lightdir.normalize();

    irr::core::vector3df lsleft = lightdir.crossProduct(viewdir).normalize();
    irr::core::vector3df lsup = lsleft.crossProduct(lightdir).normalize();

    m_shadow_view_matrix = irr::core::matrix4();
    m_shadow_view_matrix(0, 0) = lsleft.X, m_shadow_view_matrix(0, 1) = lsup.X, m_shadow_view_matrix(0, 2) = lightdir.X;
    m_shadow_view_matrix(1, 0) = lsleft.Y, m_shadow_view_matrix(1, 1) = lsup.Y, m_shadow_view_matrix(1, 2) = lightdir.Y;
    m_shadow_view_matrix(2, 0) = lsleft.Z, m_shadow_view_matrix(2, 1) = lsup.Z, m_shadow_view_matrix(2, 2) = lightdir.Z;
    m_shadow_view_matrix(3, 0) = -lsleft.dotProduct(eyepos);
    m_shadow_view_matrix(3, 1) = -lsup.dotProduct(eyepos);
    m_shadow_view_matrix(3, 2) = -lightdir.dotProduct(eyepos);

    // Calculate light space perspective (warp) matrix * shadow perspective matrix
    float cosgamma = viewdir.dotProduct(lightdir);
    float singamma = sqrtf(std::max(0.0f, 1.0f - cosgamma * cosgamma));

    irr::core::matrix4 inv_cam_view;
    cam->getViewMatrix().getInverse(inv_cam_view);

    for (unsigned i = 0; i < GVSCC_COUNT; i++)
    {
        std::vector<irr::core::vector3df> points;
        irr::core::aabbox3df lsbody;
        irr::core::matrix4 lslightproj;

        // The cascade boundaries are distances from the camera, so the slice
        // is the part of the frustum between the two spheres of radius snear
        // and sfar. Interpolating between the near and far planes instead
        // would split by view depth, whose range depends on the field of
        // view (the screen corners are farther away than their depth). The
        // spheres make the covered body curved, so it is sampled over the
        // whole frustum instead of only at the corners.
        float snear = std::max(getSplitNear(GEVulkanShadowCameraCascade(i)),
                               cam->getNearValue());
        float sfar = std::min(getSplitFar(GEVulkanShadowCameraCascade(i)),
                              cam->getFarValue());
        sfar = std::max(sfar, snear);

        const irr::scene::SViewFrustum *frust = cam->getViewFrustum();
        irr::core::vector3df corners[4] =
        {
            frust->getFarLeftDown(), frust->getFarRightDown(),
            frust->getFarLeftUp(), frust->getFarRightUp()
        };
        for (unsigned j = 0; j < 4; j++)
            cam->getViewMatrix().transformVect(corners[j]);
        // Tangent of the frustum edges in camera view space
        const float left = corners[0].X / corners[0].Z;
        const float right = corners[1].X / corners[1].Z;
        const float bottom = corners[0].Y / corners[0].Z;
        const float top = corners[2].Y / corners[2].Z;
        const float radii[2] = { snear, sfar };
        for (unsigned a = 0; a < 5; a++)
        {
            for (unsigned b = 0; b < 5; b++)
            {
                irr::core::vector3df dir(left + (right - left) * (a / 4.0f),
                                         bottom + (top - bottom) * (b / 4.0f),
                                         1.0f);
                dir.normalize();
                for (unsigned r = 0; r < 2; r++)
                {
                    irr::core::vector3df point;
                    inv_cam_view.transformVect(point, dir * radii[r]);
                    points.push_back(point);
                }
            }
        }

        // lsbody as PSR (Potential Shadow Receiver) in camera view space
        for (unsigned j = 0; j < points.size(); j++)
        {
            irr::core::vector3df point = points[j];
            cam->getViewMatrix().transformVect(point);
            if (j == 0) lsbody.reset(point);
            else lsbody.addInternalPoint(point);
        }

        float znear = std::max(cam->getNearValue(), lsbody.MinEdge.Z);
        float zfar = std::min(cam->getFarValue(), lsbody.MaxEdge.Z);

        // lsbody as PSR (Potential Shadow Receiver) in light view space
        for (unsigned j = 0; j < points.size(); j++)
        {
            m_shadow_view_matrix.transformVect(points[j]);
            irr::core::vector3df point = points[j];
            if (j == 0) lsbody.reset(point);
            else lsbody.addInternalPoint(point);
        }

        lslightproj(2, 2) = 1.0 / lsbody.getExtent().Z;
        lslightproj(3, 2) = -lsbody.MinEdge.Z / lsbody.getExtent().Z;

        for (unsigned j = 0; j < points.size(); j++)
        {
            float vec[4];
            irr::core::vector3df point = points[j];
            lslightproj.transformVect(vec, point);
            point.X = vec[0] / vec[3];
            point.Y = vec[1] / vec[3];
            point.Z = vec[2] / vec[3];
            if (j == 0) lsbody.reset(point);
            else lsbody.addInternalPoint(point);
        }

        // d is the extent of the body along the projection axis (the light
        // space Y axis, which is the view direction projected in the shadow
        // map plane)
        float d = lsbody.getExtent().Y;
        // View depth extent of the body, used to calculate the optimal n
        float zdepth = zfar - znear;

        if (singamma > 0.02f && d > 0.0f && zdepth > 0.0f)
        {
            // Wimmer et al. 2004, section 4.5: when the view is tilted
            // towards the light the far distance has to be replaced by
            // znear + (zfar - znear) * sin(gamma), and n is measured along
            // the projected view direction so it is divided by sin(gamma)
            float z1 = znear + zdepth * singamma;
            float n = (znear + sqrtf(znear * z1)) / singamma;
            n = std::max(n, 0.001f);

            float f = n + d;

            irr::core::matrix4 lispsmproj;
            lispsmproj(0, 0) = n;
            lispsmproj(1, 1) = (f + n) / d;
            lispsmproj(2, 2) = n;
            lispsmproj(3, 1) = -2.0 * f * n / d;
            lispsmproj(1, 3) = 1;
            lispsmproj(3, 3) = 0;

            irr::core::vector3df point = eyepos;
            m_shadow_view_matrix.transformVect(point);

            float vec[4];
            lslightproj.transformVect(vec, point);
            point.X = vec[0] / vec[3];
            point.Y = vec[1] / vec[3];
            point.Z = vec[2] / vec[3];

            irr::core::matrix4 correct;
            correct(3, 0) = -point.X;
            correct(3, 1) = -lsbody.MinEdge.Y + n;

            lslightproj = lispsmproj * correct * lslightproj;

            for (unsigned j = 0; j < points.size(); j++)
            {
                irr::core::vector3df point = points[j];
                float vec[4];
                lslightproj.transformVect(vec, point);
                point.X = vec[0] / vec[3];
                point.Y = vec[1] / vec[3];
                point.Z = vec[2] / vec[3];
                if (j == 0) lsbody.reset(point);
                else lsbody.addInternalPoint(point);
            }
        }

        irr::core::matrix4 fittounitcube;
        fittounitcube(0, 0) = 2.0f / lsbody.getExtent().X;
        fittounitcube(1, 1) = 2.0f / lsbody.getExtent().Y;
        fittounitcube(3, 0) = -(lsbody.MinEdge.X + lsbody.MaxEdge.X) / lsbody.getExtent().X;
        fittounitcube(3, 1) = -(lsbody.MinEdge.Y + lsbody.MaxEdge.Y) / lsbody.getExtent().Y;
        lslightproj = fittounitcube * lslightproj;

        m_shadow_projection_matrices[i] = lslightproj;
    }

    // Build shadow camera UBOs
    for (unsigned i = 0; i < GVSCC_COUNT; i++)
    {
        m_shadow_camera_ubo_data[i].m_projection_view_matrix =
            m_shadow_projection_matrices[i] * m_shadow_view_matrix;
        m_shadow_camera_ubo_data[i].m_view_matrix = m_shadow_view_matrix;
        lh->setShadowMatrices(&m_shadow_camera_ubo_data[i],
            (GEVulkanShadowCameraCascade)i);
    }

    for (unsigned i = 0; i < GVSCC_COUNT; i++)
        m_shadow_draw_calls[i]->prepareShadow(i);
}   // prepare

// ----------------------------------------------------------------------------
void GEVulkanShadowFBO::addNode(irr::scene::ISceneNode* node)
{
    for (unsigned i = 0; i < GVSCC_COUNT; i++)
        m_shadow_draw_calls[i]->addNode(node);
}   // addNode

// ----------------------------------------------------------------------------
void GEVulkanShadowFBO::generate()
{
    for (unsigned i = 0; i < GVSCC_COUNT; i++)
        m_shadow_draw_calls[i]->generate(m_vk);
}   // generate

// ----------------------------------------------------------------------------
void GEVulkanShadowFBO::uploadDynamicData(VkCommandBuffer cmd,
                                          bool& has_indirect)
{
    for (unsigned i = 0; i < m_shadow_draw_calls.size(); i++)
    {
        if (!m_shadow_draw_calls[i]->getRenderState())
            continue;
        m_shadow_draw_calls[i]->uploadDynamicData(m_vk, has_indirect, cmd);
    }
}   // uploadDynamicData

// ----------------------------------------------------------------------------
void GEVulkanShadowFBO::render(VkCommandBuffer cmd,
                               GEVulkanCamera* cam)
{
    const bool vk_13 =
        m_vk->getPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_3;
    bool rebind_base_vertex = true;
    const bool bind_mesh_textures =
        GEVulkanFeatures::supportsBindMeshTexturesAtOnce();

    // Viewport and scissor are command-buffer state; set them once here
    // regardless of which render path we take below.
    VkViewport vp;
    vp.x = 0;
    vp.y = 0;
    vp.width = m_size.Width;
    vp.height = m_size.Height;
    vp.minDepth = 0;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);

    VkRect2D scissor;
    scissor.offset.x = vp.x;
    scissor.offset.y = vp.y;
    scissor.extent.width = vp.width;
    scissor.extent.height = vp.height;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    m_master_dc->updateDataDescriptorSets(m_vk, cam);

    if (!GEVulkanFeatures::supportsDynamicRendering())
    {
        // ---------------------------------------------------------------- //
        // Legacy subpass render-pass path.                                 //
        // ---------------------------------------------------------------- //
        std::vector<VkClearValue> shadow_clear(m_shadow_draw_calls.size());
        for (size_t i = 0; i < shadow_clear.size(); i++)
            shadow_clear[i].depthStencil = {1.0f, 0};

        VkRenderPassBeginInfo shadow_pass_info = {};
        shadow_pass_info.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        shadow_pass_info.clearValueCount   = (uint32_t)shadow_clear.size();
        shadow_pass_info.pClearValues      = shadow_clear.data();
        shadow_pass_info.renderArea.offset = {0, 0};
        shadow_pass_info.renderPass        = m_rtt_render_pass;
        shadow_pass_info.framebuffer       = m_rtt_frame_buffer;
        shadow_pass_info.renderArea.extent = { m_size.Width, m_size.Height };
        vkCmdBeginRenderPass(cmd, &shadow_pass_info, VK_SUBPASS_CONTENTS_INLINE);

        for (int i = 0; i < (int)m_shadow_draw_calls.size(); i++)
        {
            if (m_shadow_draw_calls[i]->getRenderState())
            {
                if (bind_mesh_textures)
                    m_shadow_draw_calls[i]->bindAllMaterials(cmd);
                else
                    rebind_base_vertex = true;
                m_shadow_draw_calls[i]->updateDataDescriptorSets(m_vk, cam);

                // Determine whether this layer must borrow from a bucket owner.
                int bucket_owner = -1;
                if (getSharingDrawCallCount() != 0)
                {
                    int pointlight_layer = i - (int)getLayerOffset();
                    bucket_owner = getLayerOwner(pointlight_layer);
                    if (bucket_owner != -1)
                        bucket_owner += getLayerOffset();
                }
                if (bucket_owner != -1 && bucket_owner != i)
                {
                    m_shadow_draw_calls[bucket_owner].get()
                        ->swapDrawCallData(m_shadow_draw_calls[i].get());
                }
                m_shadow_draw_calls[i]->renderPipeline(m_vk, cmd, GVPT_DEPTH,
                    rebind_base_vertex);
                if (bucket_owner != -1 && bucket_owner != i)
                {
                    m_shadow_draw_calls[bucket_owner].get()
                        ->swapDrawCallData(m_shadow_draw_calls[i].get());
                }
            }
            if (i != (int)m_shadow_draw_calls.size() - 1)
                vkCmdNextSubpass(cmd, VK_SUBPASS_CONTENTS_INLINE);
        }
        vkCmdEndRenderPass(cmd);
    }
    else
    {
        // ---------------------------------------------------------------- //
        // VK_KHR_dynamic_rendering path.                                   //
        //                                                                  //
        // Layers with getRenderState()==false are skipped entirely — no    //
        // vkCmdBeginRendering is issued for them.  We transition ALL       //
        // layers up front in a single pipeline barrier so they are in a    //
        // legal layout for both writing (rendered) and sampling (skipped). //
        // ---------------------------------------------------------------- //
        const uint32_t count = (uint32_t)m_shadow_draw_calls.size();

        std::vector<VkImageMemoryBarrier> pre_barriers;
        pre_barriers.reserve(count);
        for (uint32_t i = 0; i < count; i++)
        {
            VkImageMemoryBarrier b = {};
            b.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            b.srcAccessMask       = VK_ACCESS_SHADER_READ_BIT;
            b.image               = m_image;
            b.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
            b.subresourceRange.baseMipLevel   = 0;
            b.subresourceRange.levelCount     = 1;
            b.subresourceRange.baseArrayLayer = i;
            b.subresourceRange.layerCount     = 1;
            // UNDEFINED as old layout discards contents — safe because we
            // either clear (rendered layers) or don't need prior data
            // (skipped layers).
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;

            if (m_shadow_draw_calls[i]->getRenderState())
            {
                b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                                  VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
                b.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            }
            else
            {
                // Skipped layers land immediately in the sampling layout so
                // shaders can read them (as cleared/empty shadow maps).
                b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                b.newLayout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            }
            pre_barriers.push_back(b);
        }

        if (!pre_barriers.empty())
        {
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_DEPENDENCY_BY_REGION_BIT,
                0, NULL, 0, NULL,
                (uint32_t)pre_barriers.size(), pre_barriers.data());
        }

        for (int i = 0; i < (int)m_shadow_draw_calls.size(); i++)
        {
            if (!m_shadow_draw_calls[i]->getRenderState())
                continue;

            // Begin a rendering scope for this single cascade layer.
            VkClearValue clear_val;
            clear_val.depthStencil = {1.0f, 0};

            VkRenderingAttachmentInfo depth_attachment = {};
            depth_attachment.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
            depth_attachment.imageView   = m_frame_buffer_image_views[i];
            depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            depth_attachment.loadOp      = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depth_attachment.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
            depth_attachment.clearValue  = clear_val;

            VkRenderingInfo rendering_info   = {};
            rendering_info.sType             = VK_STRUCTURE_TYPE_RENDERING_INFO;
            rendering_info.renderArea.offset = {0, 0};
            rendering_info.renderArea.extent = {m_size.Width, m_size.Height};
            rendering_info.layerCount        = 1;
            rendering_info.pDepthAttachment  = &depth_attachment;

            if (vk_13)
                vkCmdBeginRendering(cmd, &rendering_info);
            else
                vkCmdBeginRenderingKHR(cmd, &rendering_info);

            if (bind_mesh_textures)
                m_shadow_draw_calls[i]->bindAllMaterials(cmd);
            else
                rebind_base_vertex = true;
            m_shadow_draw_calls[i]->updateDataDescriptorSets(m_vk, cam);

            // Determine whether this layer must borrow from a bucket owner.
            int bucket_owner = -1;
            if (getSharingDrawCallCount() != 0)
            {
                int pointlight_layer = i - (int)getLayerOffset();
                bucket_owner = getLayerOwner(pointlight_layer);
                if (bucket_owner != -1)
                    bucket_owner += getLayerOffset();
            }
            if (bucket_owner != -1 && bucket_owner != i)
            {
                m_shadow_draw_calls[bucket_owner].get()
                    ->swapDrawCallData(m_shadow_draw_calls[i].get());
            }
            m_shadow_draw_calls[i]->renderPipeline(m_vk, cmd, GVPT_DEPTH,
                rebind_base_vertex);
            if (bucket_owner != -1 && bucket_owner != i)
            {
                m_shadow_draw_calls[bucket_owner].get()
                    ->swapDrawCallData(m_shadow_draw_calls[i].get());
            }

            if (vk_13)
                vkCmdEndRendering(cmd);
            else
                vkCmdEndRenderingKHR(cmd);

            // Transition this layer to the sampling layout so shadow
            // shaders can read it after the barrier.
            VkImageMemoryBarrier post_barrier = {};
            post_barrier.sType                = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            post_barrier.srcAccessMask        = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
            post_barrier.dstAccessMask        = VK_ACCESS_SHADER_READ_BIT;
            post_barrier.oldLayout            = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            post_barrier.newLayout            = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
            post_barrier.image                = m_image;
            post_barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
            post_barrier.subresourceRange.baseMipLevel   = 0;
            post_barrier.subresourceRange.levelCount     = 1;
            post_barrier.subresourceRange.baseArrayLayer = (uint32_t)i;
            post_barrier.subresourceRange.layerCount     = 1;
            vkCmdPipelineBarrier(cmd,
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                VK_DEPENDENCY_BY_REGION_BIT,
                0, NULL, 0, NULL, 1, &post_barrier);
        }
    }
}   // render

// ----------------------------------------------------------------------------
GEVulkanCameraUBO* GEVulkanShadowFBO::getCameraUBO(unsigned i) const
{
    return &m_shadow_camera_ubo_data[i];
}   // getCameraUBO

// ----------------------------------------------------------------------------
GEVulkanDrawCall* GEVulkanShadowFBO::getDrawCall(unsigned i) const
{
    return m_shadow_draw_calls.at(i).get();
}   // getDrawCall

}   // namespace GE
