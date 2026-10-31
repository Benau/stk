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

namespace
{
// ----------------------------------------------------------------------------
// The direction of the frustum cone with the largest dot product to the axis.
// The maximum is at the axis itself when it is inside the cone, otherwise at
// an edge of the cone where the axis projected onto the edge plane is still
// inside the cone, or at a corner of the cone.
irr::core::vector3df getConeExtremeDirection(const irr::core::vector3df& axis,
    const irr::core::vector3df* plane_normals,
    const irr::core::vector3df* cone_corners)
{
    bool inside = true;
    for (unsigned i = 0; i < 4; i++)
    {
        if (plane_normals[i].dotProduct(axis) < -1e-5f)
        {
            inside = false;
            break;
        }
    }
    if (inside)
        return axis;

    float best_dot = -2.0f;
    irr::core::vector3df best = cone_corners[0];
    for (unsigned i = 0; i < 4; i++)
    {
        irr::core::vector3df dir = axis - plane_normals[i] *
            axis.dotProduct(plane_normals[i]);
        if (dir.getLengthSQ() < 1e-8f)
            continue;
        dir.normalize();
        bool in_cone = true;
        for (unsigned j = 0; j < 4; j++)
        {
            if (plane_normals[j].dotProduct(dir) < -1e-5f)
            {
                in_cone = false;
                break;
            }
        }
        if (!in_cone)
            continue;
        float dot = axis.dotProduct(dir);
        if (dot > best_dot)
        {
            best_dot = dot;
            best = dir;
        }
    }
    for (unsigned i = 0; i < 4; i++)
    {
        float dot = axis.dotProduct(cone_corners[i]);
        if (dot > best_dot)
        {
            best_dot = dot;
            best = cone_corners[i];
        }
    }
    return best;
}   // getConeExtremeDirection

// ----------------------------------------------------------------------------
// The maximum of axis.dir over the unit directions of the cone patch.
float getConeSupport(const irr::core::vector3df& axis,
    const irr::core::vector3df* plane_normals,
    const irr::core::vector3df* cone_corners)
{
    float len = axis.getLength();
    if (len < 1e-12f)
        return 0.0f;
    irr::core::vector3df dir = axis / len;
    irr::core::vector3df extreme = getConeExtremeDirection(dir,
        plane_normals, cone_corners);
    return len * dir.dotProduct(extreme);
}   // getConeSupport

// ----------------------------------------------------------------------------
// The maximum of x/(y+c) over the far spherical patch of the cascade body,
// where x = sfar*(a.dir) and y = sfar*(b.dir) (a and b are light space axes
// rotated to view space). The level sets of x/(y+c) are planes through the
// light space line x = 0, y = -c, so the maximum is the largest t where the
// plane x = t*(y+c) still touches the patch: it is the root of
// F(t) = max((a - t*b).dir) - t*k with k = c/sfar. F is strictly decreasing
// because y + c is positive on the body, so a bisection finds it.
float getPatchWarpedSup(const irr::core::vector3df& a,
    const irr::core::vector3df& b, float k, float bound,
    const irr::core::vector3df* plane_normals,
    const irr::core::vector3df* cone_corners)
{
    float lo = -bound, hi = bound;
    for (unsigned it = 0; it < 40; it++)
    {
        float mid = (lo + hi) * 0.5f;
        if (getConeSupport(a - b * mid, plane_normals, cone_corners) -
            mid * k >= 0.0f)
            lo = mid;
        else
            hi = mid;
    }
    return (lo + hi) * 0.5f;
}   // getPatchWarpedSup
}

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
    // At a view depth z the rays of the frustum reach the distance z /
    // min_ray_cos at most. The pbr shaders calculate the same near depths of
    // the cascades from it (see getShadowFactor of utils/pbr_light.glsl)
    float min_ray_cos = 1.0f;
    for (unsigned j = 0; j < 4; j++)
    {
        min_ray_cos = std::min(min_ray_cos,
            corners[j].Z / corners[j].getLength());
    }

    // The frustum is the cone from the camera through its far rectangle. The
    // side planes of the cone and its corner directions are used to find the
    // exact bounds of the spherical far patch of the cascade bodies.
    irr::core::vector3df cone_normals[4] =
    {
        irr::core::vector3df(1.0f, 0.0f, -left).normalize(),
        irr::core::vector3df(-1.0f, 0.0f, right).normalize(),
        irr::core::vector3df(0.0f, 1.0f, -bottom).normalize(),
        irr::core::vector3df(0.0f, -1.0f, top).normalize()
    };
    irr::core::vector3df cone_corners[4] =
    {
        irr::core::vector3df(left, bottom, 1.0f).normalize(),
        irr::core::vector3df(right, bottom, 1.0f).normalize(),
        irr::core::vector3df(right, top, 1.0f).normalize(),
        irr::core::vector3df(left, top, 1.0f).normalize()
    };
    irr::core::vector3df light_axes[6] =
    {
        lsleft, -lsleft, lsup, -lsup, lightdir, -lightdir
    };

    float split_near[GVSCC_COUNT];
    for (unsigned i = 0; i < GVSCC_COUNT; i++)
    {
        irr::core::aabbox3df lsbody;
        irr::core::matrix4 lslightproj;

        // The cascades are split by distance and by view depth: the far
        // boundary of a cascade is the sphere of radius sfar, the near one is
        // the plane at depth snear. The near depth is the largest one where
        // the far sphere of the previous cascade still covers the whole
        // screen (its radius is split_far * min_ray_cos), so the bodies of
        // the two cascades don't leave any gap at the screen corners, where
        // the distance of a pixel is much larger than its depth.
        float snear = std::max(getSplitNear(
            (GEVulkanShadowCameraCascade)i, cam->getNearValue(), min_ray_cos),
            cam->getNearValue());
        float sfar = std::min(getSplitFar(GEVulkanShadowCameraCascade(i)),
                              cam->getFarValue());
        sfar = std::max(sfar, snear);
        split_near[i] = snear;

        // The body is the convex hull of the near plane rectangle and the far
        // spherical patch. Every face of its light space bound is supported
        // either by a corner of the rectangle or by a point of the patch where
        // the face touches the distance sphere, so the bound is tight: the 4
        // corners and the extreme point of the patch along each of the 6 light
        // space axis directions give all the supports. No interpolated convex
        // hull is needed (the patch is curved and the hull of sampled points
        // would underestimate it).
        irr::core::vector3df body_points[4 + 6];
        for (unsigned j = 0; j < 4; j++)
        {
            // The near rectangle point along the direction of a cone corner,
            // scaled to the view depth of the near plane
            irr::core::vector3df near_point = cone_corners[j] *
                (snear / cone_corners[j].Z);
            inv_cam_view.transformVect(body_points[j], near_point);
        }
        for (unsigned j = 0; j < 6; j++)
        {
            irr::core::vector3df axis;
            cam->getViewMatrix().rotateVect(axis, light_axes[j]);
            axis.normalize();
            irr::core::vector3df dir = getConeExtremeDirection(axis,
                cone_normals, cone_corners);
            inv_cam_view.transformVect(body_points[4 + j], dir * sfar);
        }

        // lsbody as PSR (Potential Shadow Receiver) in light view space
        irr::core::vector3df near_points_ls[4];
        for (unsigned j = 0; j < 4 + 6; j++)
        {
            irr::core::vector3df point = body_points[j];
            m_shadow_view_matrix.transformVect(point);
            if (j < 4)
                near_points_ls[j] = point;
            if (j == 0) lsbody.reset(point);
            else lsbody.addInternalPoint(point);
        }

        // Only the far cascade is shrunk to the scene. The near cascades keep
        // their fixed body, the near boundary of the next cascade relies on
        // the far distance of the previous one. The parts of the far body
        // outside of the bounding box of the scene can't contain any shadow
        // receiver, so they are removed by clipping the body by the light
        // space bounding box of the scene.
        bool clip_xy = false;
        if (i == GVSCC_FAR && m_scene_bbox_valid)
        {
            irr::core::vector3df scene_edges[8];
            m_scene_bbox.getEdges(scene_edges);
            irr::core::aabbox3df scene_body;
            for (unsigned j = 0; j < 8; j++)
            {
                m_shadow_view_matrix.transformVect(scene_edges[j]);
                if (j == 0) scene_body.reset(scene_edges[j]);
                else scene_body.addInternalPoint(scene_edges[j]);
            }
            if (lsbody.intersectsWithBox(scene_body))
            {
                irr::core::aabbox3df clipped = lsbody.intersect(scene_body);
                irr::core::vector3df extent = clipped.getExtent();
                // A degenerate body would break the projection matrices
                if (extent.X > 1e-3f && extent.Y > 1e-3f && extent.Z > 1e-3f)
                {
                    // Only a narrowing of X or Y changes the fit, a clip of Z
                    // alone keeps the X/Y bounds of the body
                    clip_xy = clipped.MinEdge.X != lsbody.MinEdge.X ||
                        clipped.MaxEdge.X != lsbody.MaxEdge.X ||
                        clipped.MinEdge.Y != lsbody.MinEdge.Y ||
                        clipped.MaxEdge.Y != lsbody.MaxEdge.Y;
                    lsbody = clipped;
                }
            }
        }

        // The 8 corners of the (clipped) body in light view space
        irr::core::vector3df ls_points[8];
        lsbody.getEdges(ls_points);

        // The view depth range of the body is fixed by the split distances:
        // the far cascade is still built with its full 200 distance and is
        // only clipped inside the scene bounding box
        float znear = snear;
        float zfar = sfar;

        lslightproj(2, 2) = 1.0 / lsbody.getExtent().Z;
        lslightproj(3, 2) = -lsbody.MinEdge.Z / lsbody.getExtent().Z;

        for (unsigned j = 0; j < 8; j++)
        {
            float vec[4];
            irr::core::vector3df point = ls_points[j];
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

        // Exact warped X bounds of the body, calculated in the LISPSM branch
        float exact_fit_min_x = 0.0f, exact_fit_max_x = 0.0f;
        bool has_exact_fit_x = false;

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

            // The warped X' = n*x/(y+c) of the body, c = n - MinY (the
            // denominator y + c is positive on the body). It is bounded
            // exactly by the near rectangle corners (a linear fractional
            // function is extreme at the corners of the rectangle) and by the
            // spherical patch, but the warped corners of the box pair the
            // largest X with the smallest Y and overstate X', which leaves a
            // wide unused margin in the map. The warped box corners are only
            // used when the scene clip narrowed the X/Y bounds, because the
            // clipped body shape is not tracked.
            if (!clip_xy)
            {
                float c = n - lsbody.MinEdge.Y;
                irr::core::vector3df lsleft_view, lsup_view;
                cam->getViewMatrix().rotateVect(lsleft_view, lsleft);
                cam->getViewMatrix().rotateVect(lsup_view, lsup);
                float k = c / sfar;
                float bound = sfar / n;
                float hmax = getPatchWarpedSup(lsleft_view, lsup_view, k,
                    bound, cone_normals, cone_corners);
                float hmin = -getPatchWarpedSup(-lsleft_view, lsup_view, k,
                    bound, cone_normals, cone_corners);
                for (unsigned j = 0; j < 4; j++)
                {
                    float h = near_points_ls[j].X /
                        (near_points_ls[j].Y + c);
                    hmin = std::min(hmin, h);
                    hmax = std::max(hmax, h);
                }
                exact_fit_min_x = n * hmin;
                exact_fit_max_x = n * hmax;
                has_exact_fit_x = exact_fit_max_x > exact_fit_min_x + 1e-6f;
            }

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

            for (unsigned j = 0; j < 8; j++)
            {
                irr::core::vector3df point = ls_points[j];
                float vec[4];
                lslightproj.transformVect(vec, point);
                point.X = vec[0] / vec[3];
                point.Y = vec[1] / vec[3];
                point.Z = vec[2] / vec[3];
                if (j == 0) lsbody.reset(point);
                else lsbody.addInternalPoint(point);
            }
        }

        // Fit the warped body to the unit cube. The warp is monotone in Y, so
        // the warped corners give the exact Y bounds. For X the exact bounds
        // of the body are used when known, the corners of the box alone are
        // too loose.
        float fit_min_x = lsbody.MinEdge.X;
        float fit_max_x = lsbody.MaxEdge.X;
        if (has_exact_fit_x)
        {
            fit_min_x = exact_fit_min_x;
            fit_max_x = exact_fit_max_x;
        }
        irr::core::matrix4 fittounitcube;
        fittounitcube(0, 0) = 2.0f / (fit_max_x - fit_min_x);
        fittounitcube(1, 1) = 2.0f / lsbody.getExtent().Y;
        fittounitcube(3, 0) = -(fit_min_x + fit_max_x) / (fit_max_x - fit_min_x);
        fittounitcube(3, 1) = -(lsbody.MinEdge.Y + lsbody.MaxEdge.Y) / lsbody.getExtent().Y;
        lslightproj = fittounitcube * lslightproj;

        m_shadow_projection_matrices[i] = lslightproj;
    }

    // The pbr shaders select the cascades with these near depths and blend
    // them with min_ray_cos, they have to match the bodies calculated here
    lh->setShadowSplit(irr::core::vector3df(split_near[0], split_near[1],
        split_near[2]), min_ray_cos);

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
    irr::core::aabbox3df bb = node->getTransformedBoundingBox();
    if (m_scene_bbox_valid)
        m_scene_bbox.addInternalBox(bb);
    else
    {
        m_scene_bbox = bb;
        m_scene_bbox_valid = true;
    }
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
