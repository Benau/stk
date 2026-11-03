#include "ge_vulkan_omni_shadow_fbo.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_light_handler.hpp"
#include "ge_vulkan_omni_shadow_draw_call.hpp"

#include "ILightSceneNode.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace GE
{

// ----------------------------------------------------------------------------
// Shadow mapping strategy per light type, the point light is the same as
// bevy's fetch_point_shadow: a cube map of the light, every face a 90 degree
// perspective projection with far = radius, and the face is picked by the
// largest absolute component of the direction to the fragment. The atlas
// stores the faces of the cube as separate tiles instead of a cube image, so
// the shader projects with the matrix of the face it picked instead of
// sampling a texture cube.
//
// Point light  :  the 6 faces of a cube map. The face order of the tiles is
//                 +X, -X, +Y, -Y, +Z, -Z, the order of the faces of
//                 getOmniShadowCoord in pbr_light.glsl. The view matrix of a
//                 face looks along its forward with the standard cube map up
//                 hint, the 90 degree projection has the Y flip of the engine
//                 cameras (the pipelines render with VK_FRONT_FACE_CLOCKWISE
//                 and cull back faces, see the spot light below).
//
// Spot light   :  1 perspective map of the cone in the first slot of the
//                 light. Its camera matrices hold the view matrix of the
//                 light and the projection of the cone (fov = 2 * (outer cone
//                 + padding), far = radius). The other 5 slots of the light
//                 are unused.
//
// All the maps are tiles of a single 2 x shadow_size depth atlas (a single
// array layer), so the point and spot lights share one map and one barrier:
//
//   slot = light_id * OMNI_FACES_PER_LIGHT + face
//   the tiles are a square grid of ceil(sqrt(slots)) x ceil(sqrt(slots))
//   tiles, the tile resolution is the atlas size divided by it
//
// With shadow_size 1024 and 8 lights (48 slots) the atlas is 2048 x 2048 and
// every tile is 292 x 292, the total area of the point light shadows is the
// same as the atlas. The shaders compute the same grid from u_shadow_size and
// u_point_shadow_limit (getOmniShadowAtlasRect of pbr_light.glsl) and sample
// the tiles with PCF clamped to them.
//
// The tiles are cached: a bucket (2 lights, 12 slots, the geometry is shared
// by the draw calls of a bucket) is only rendered again when the data of one
// of its lights changed or a moving node reaches one of its lights. The tiles
// of a light which lost its shadow are cleared to 1.0 once.
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// The forward of each face of the cube map and the up hint of its look at, in
// the +X, -X, +Y, -Y, +Z, -Z order of the face selection of
// getOmniShadowCoord. The up hint is the one of the standard cube map faces,
// so the projection of every face matches the usual cube map uv.
struct GCubeFace
{
    irr::core::vector3df m_forward;
    irr::core::vector3df m_up_hint;
};
static const GCubeFace g_cube_faces[OMNI_FACES_PER_LIGHT] =
{
    { irr::core::vector3df(1.0f, 0.0f, 0.0f),
      irr::core::vector3df(0.0f, 1.0f, 0.0f) },
    { irr::core::vector3df(-1.0f, 0.0f, 0.0f),
      irr::core::vector3df(0.0f, 1.0f, 0.0f) },
    { irr::core::vector3df(0.0f, 1.0f, 0.0f),
      irr::core::vector3df(0.0f, 0.0f, -1.0f) },
    { irr::core::vector3df(0.0f, -1.0f, 0.0f),
      irr::core::vector3df(0.0f, 0.0f, 1.0f) },
    { irr::core::vector3df(0.0f, 0.0f, 1.0f),
      irr::core::vector3df(0.0f, 1.0f, 0.0f) },
    { irr::core::vector3df(0.0f, 0.0f, -1.0f),
      irr::core::vector3df(0.0f, 1.0f, 0.0f) }
};

// ----------------------------------------------------------------------------
// The view matrix of a cube map face: the rows of the world to light space
// transform are right, up and forward, the translation follows from the
// position of the light, same layout as the spot light below.
static irr::core::matrix4 buildFaceView(const irr::core::vector3df& pos,
                                        const GCubeFace& face)
{
    irr::core::vector3df forward = face.m_forward;
    forward.normalize();
    irr::core::vector3df right = face.m_up_hint.crossProduct(forward);
    right.normalize();
    irr::core::vector3df up = forward.crossProduct(right);

    irr::core::matrix4 view;
    view(0, 0) = right.X;   view(0, 1) = up.X;   view(0, 2) = forward.X;
    view(1, 0) = right.Y;   view(1, 1) = up.Y;   view(1, 2) = forward.Y;
    view(2, 0) = right.Z;   view(2, 1) = up.Z;   view(2, 2) = forward.Z;
    view(3, 0) = -right.dotProduct(pos);
    view(3, 1) = -up.dotProduct(pos);
    view(3, 2) = -forward.dotProduct(pos);
    return view;
}   // buildFaceView

// ----------------------------------------------------------------------------
static bool equalFloat(float a, float b)
{
    return std::fabs(a - b) <= 1e-5f;
}   // equalFloat

// ----------------------------------------------------------------------------
static bool equalVector(const irr::core::vector3df& a,
                        const irr::core::vector3df& b)
{
    return equalFloat(a.X, b.X) && equalFloat(a.Y, b.Y) &&
        equalFloat(a.Z, b.Z);
}   // equalVector

// ----------------------------------------------------------------------------
GEVulkanOmniShadowFBO::GEVulkanOmniShadowFBO(GEVulkanDriver* vk,
                                             unsigned shadow_size,
                                             GEVulkanDrawCall* master_dc)
                     : GEVulkanShadowFBO(vk, shadow_size * 2, master_dc, NULL,
                                         1,
                                         getGEConfig()->m_point_shadow_limit *
                                         OMNI_FACES_PER_LIGHT),
                       m_light_handler(NULL),
                       m_has_cached_nodes(false)
{
    const unsigned atlas_size = shadow_size * 2;
    // The slots are a square grid of tiles which covers all of them, the
    // shaders (getOmniShadowTilesPerRow of pbr_light.glsl) compute the same
    // values
    m_tiles_per_row = 1;
    while (m_tiles_per_row * m_tiles_per_row < m_slot_count)
        m_tiles_per_row++;
    m_tile_size = atlas_size / m_tiles_per_row;
    m_slot_states.resize(m_slot_count);
}   // GEVulkanOmniShadowFBO

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::createDrawCalls()
{
    for (unsigned i = 0; i < m_slot_count; i++)
    {
        m_shadow_draw_calls.push_back(std::unique_ptr<GEVulkanShadowDrawCall>(
            new GEVulkanOmniShadowDrawCall(this, i)));
    }
}   // createDrawCalls

// ----------------------------------------------------------------------------
unsigned GEVulkanOmniShadowFBO::getSharingDrawCallCount() const
{
    // 2 lights (6 face slots each) share one built geometry
    const unsigned count = 2 * OMNI_FACES_PER_LIGHT;
    static_assert((count % OMNI_FACES_PER_LIGHT) == 0,
        "it should be a multiple of OMNI_FACES_PER_LIGHT");
    return count;
}   // getSharingDrawCallCount

// ----------------------------------------------------------------------------
int GEVulkanOmniShadowFBO::getLayerOwner(int pointlight_layer) const
{
    if (pointlight_layer >= 0)
    {
        return (pointlight_layer / getSharingDrawCallCount()) *
            getSharingDrawCallCount();
    }
    return -1;
}   // getLayerOwner

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::buildLightMatrices(unsigned light_id)
{
    const irr::core::vector3df& pos =
        m_light_handler->getLightPosition(light_id);
    const float radius = m_light_handler->getLightRadius(light_id);
    m_lights.push_back({irr::core::vector3df(pos.X, pos.Y, pos.Z), radius});

    const unsigned slot = light_id * OMNI_FACES_PER_LIGHT;

    if (m_light_handler->getLightIsSpot(light_id))
    {
        // The cone is in front of the light, a perspective map of it has a
        // much better resolution than a paraboloid map and no seam at the
        // cone border. The basis matches the sampling of the shader, which
        // projects with the same matrix (no per vertex normalization).
        irr::core::vector3df forward =
            m_light_handler->getLightDirection(light_id);
        forward.normalize();
        irr::core::vector3df up_hint(0.0f, 1.0f, 0.0f);
        if (fabsf(forward.Y) > 0.99f)
            up_hint.set(1.0f, 0.0f, 0.0f);
        irr::core::vector3df right = up_hint.crossProduct(forward);
        right.normalize();
        irr::core::vector3df up = forward.crossProduct(right);

        irr::core::matrix4 view;
        view(0, 0) = right.X;   view(0, 1) = up.X;   view(0, 2) = forward.X;
        view(1, 0) = right.Y;   view(1, 1) = up.Y;   view(1, 2) = forward.Y;
        view(2, 0) = right.Z;   view(2, 1) = up.Z;   view(2, 2) = forward.Z;
        view(3, 0) = -right.dotProduct(pos);
        view(3, 1) = -up.dotProduct(pos);
        view(3, 2) = -forward.dotProduct(pos);

        // 2 degrees of padding so the PCF taps at the border of the cone
        // stay inside the map, and enough far distance to cover the range of
        // the light
        const float outer_cone =
            m_light_handler->getLightOuterCone(light_id);
        const float half_fov = std::min(outer_cone + 0.035f, 1.5f);
        const float near_value = std::max(std::min(radius * 0.02f,
            radius * 0.5f), 0.02f);
        irr::core::matrix4 projection;
        projection.buildProjectionMatrixPerspectiveFovLH(half_fov * 2.0f,
            1.0f, near_value, std::max(radius, near_value + 0.01f));
        // The engine's camera projections flip the clip Y for the Vulkan
        // framebuffer (see GEVulkanCamera), the window has no negative height,
        // so the light projection needs the same flip: with the clockwise
        // front face of the pipelines, a projection without it culls the
        // faces that point to the light. The shader samples the map with this
        // same matrix, so the flip is consistent there.
        irr::core::matrix4 clip;
        clip[5] = -1.0f;
        projection = clip * projection;

        GEVulkanCameraUBO* ubo = &m_shadow_camera_ubo_data[slot];
        ubo->m_view_matrix = view;
        ubo->m_projection_view_matrix = projection * view;
        ubo->m_shadow_flags[0] = 0.0f;
        m_shadow_projection_matrices[slot] =
            ubo->m_projection_view_matrix;
        m_light_handler->setLightShadowMatrices(ubo, light_id, 0);
        return;
    }

    // The 6 faces of the cube map: a 90 degree perspective projection of the
    // light sphere per face. The near plane is bevy's POINT_LIGHT_NEAR_Z for
    // a normal light, lower for a tiny one so it stays in front of the far
    // plane (radius).
    const float near_value = std::min(0.1f,
        std::max(radius * 0.5f, 0.01f));
    irr::core::matrix4 projection;
    projection.buildProjectionMatrixPerspectiveFovLH(irr::core::PI * 0.5f,
        1.0f, near_value, std::max(radius, near_value + 0.01f));
    // Same Y flip as the spot light below, for the winding of the face
    // culling
    irr::core::matrix4 clip;
    clip[5] = -1.0f;
    projection = clip * projection;

    for (unsigned face = 0; face < OMNI_FACES_PER_LIGHT; face++)
    {
        GEVulkanCameraUBO* ubo =
            &m_shadow_camera_ubo_data[slot + face];
        ubo->m_view_matrix = buildFaceView(pos, g_cube_faces[face]);
        ubo->m_projection_view_matrix =
            projection * ubo->m_view_matrix;
        ubo->m_shadow_flags[0] = 0.0f;
        m_shadow_projection_matrices[slot + face] =
            ubo->m_projection_view_matrix;
        m_light_handler->setLightShadowMatrices(ubo, light_id, face);
    }
}   // buildLightMatrices

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::prepare(GEVulkanCamera* cam,
                                    GEVulkanLightHandler* lh)
{
    m_light_handler = lh;
    m_nodes.clear();
    m_next_cached_nodes.clear();
}   // prepare

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::addNode(irr::scene::ISceneNode* n)
{
    if (n->getType() != irr::scene::ESNT_MESH &&
        n->getType() != irr::scene::ESNT_ANIMATED_MESH)
        return;

    m_nodes.push_back(n);
    NodeBox& box = m_next_cached_nodes[n];
    box.m_box = n->getTransformedBoundingBox();
    // The vertices of an animated mesh change without the box changing
    box.m_animated = n->getType() == irr::scene::ESNT_ANIMATED_MESH;
}   // addNode

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::addTileClear(unsigned slot)
{
    m_clear_tiles.push_back(slot);
}   // addTileClear

// ----------------------------------------------------------------------------
bool GEVulkanOmniShadowFBO::bucketIntersectsBox(
    unsigned bucket, const irr::core::aabbox3df& b) const
{
    const unsigned share = getSharingDrawCallCount();
    const unsigned first_light = bucket * share / OMNI_FACES_PER_LIGHT;
    const unsigned light_end = std::min(
        first_light + share / OMNI_FACES_PER_LIGHT,
        (unsigned)m_lights.size());
    for (unsigned i = first_light; i < light_end; i++)
    {
        const irr::core::vector3df& lp = m_lights[i].m_position;
        const irr::core::vector3df closest(
            std::max(b.MinEdge.X, std::min(lp.X, b.MaxEdge.X)),
            std::max(b.MinEdge.Y, std::min(lp.Y, b.MaxEdge.Y)),
            std::max(b.MinEdge.Z, std::min(lp.Z, b.MaxEdge.Z)));
        if ((lp - closest).getLengthSQ() <=
            m_lights[i].m_radius * m_lights[i].m_radius)
            return true;
    }
    return false;
}   // bucketIntersectsBox

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::generate()
{
    assert(m_light_handler != NULL);

    const unsigned shadow_limit = getGEConfig()->m_point_shadow_limit;
    const unsigned light_count = std::min(m_light_handler->getLightCount(),
        shadow_limit);
    // Non-occluded lights are stable-partitioned to the front of the rendered
    // light array by GEVulkanLightHandler::generate(). Only those have maps
    // (the others are cleared to 1.0 and sample as fully lit).
    const unsigned non_occluded = std::min(
        m_light_handler->getNonOccludedLightCount(), light_count);

    m_lights.clear();
    for (unsigned i = 0; i < non_occluded; i++)
        buildLightMatrices(i);

    // The nodes which moved (or appeared/disappeared) since the last frame.
    // A caster which can cast into the sphere of a light is inside it (a ray
    // from the light to a receiver is inside the sphere), so a bucket only
    // changes when one of its lights reaches such a node.
    std::vector<irr::core::aabbox3df> moved_boxes;
    for (auto& p : m_next_cached_nodes)
    {
        auto it = m_cached_nodes.find(p.first);
        if (it == m_cached_nodes.end() || p.second.m_animated ||
            !it->second.m_box.MinEdge.equals(p.second.m_box.MinEdge) ||
            !it->second.m_box.MaxEdge.equals(p.second.m_box.MaxEdge))
        {
            moved_boxes.push_back(p.second.m_box);
        }
    }
    for (auto& p : m_cached_nodes)
    {
        if (m_next_cached_nodes.find(p.first) == m_next_cached_nodes.end())
            moved_boxes.push_back(p.second.m_box);
    }

    m_dirty_buckets.clear();
    m_clear_tiles.clear();

    const unsigned share = getSharingDrawCallCount();
    const unsigned bucket_count =
        (m_slot_count + share - 1) / share;
    for (unsigned b = 0; b < bucket_count; b++)
    {
        bool dirty = !m_has_cached_nodes;
        const unsigned slot_begin = b * share;
        const unsigned slot_end = std::min(slot_begin + share, m_slot_count);
        for (unsigned slot = slot_begin; slot < slot_end; slot++)
        {
            const unsigned light = slot / OMNI_FACES_PER_LIGHT;
            const unsigned face = slot % OMNI_FACES_PER_LIGHT;
            SlotState current;
            if (light < non_occluded)
            {
                current.m_spot = m_light_handler->getLightIsSpot(light);
                current.m_active = face == 0 || !current.m_spot;
                current.m_rendered = true;
                current.m_position = m_lights[light].m_position;
                current.m_radius = m_lights[light].m_radius;
                if (current.m_spot)
                {
                    current.m_direction =
                        m_light_handler->getLightDirection(light);
                    current.m_outer_cone =
                        m_light_handler->getLightOuterCone(light);
                }
            }
            SlotState& old = m_slot_states[slot];
            if (!current.m_active)
            {
                // The tile keeps an old shadow (or was never cleared), clear
                // it once to fully lit
                if (!old.m_rendered || old.m_active)
                    addTileClear(slot);
                old.m_active = false;
                old.m_rendered = true;
                continue;
            }
            if (!old.m_rendered || !old.m_active ||
                old.m_spot != current.m_spot ||
                !equalFloat(old.m_radius, current.m_radius) ||
                !equalVector(old.m_position, current.m_position) ||
                !equalVector(old.m_direction, current.m_direction) ||
                !equalFloat(old.m_outer_cone, current.m_outer_cone))
            {
                dirty = true;
            }
            old = current;
        }
        if (!dirty)
        {
            for (const irr::core::aabbox3df& box : moved_boxes)
            {
                if (bucketIntersectsBox(b, box))
                {
                    dirty = true;
                    break;
                }
            }
        }
        if (dirty)
            m_dirty_buckets.push_back(b);
    }

    // Only the active slots of a dirty bucket are submitted, the others
    // borrow the geometry of the first slot of the bucket at render time
    for (size_t i = 0; i < m_shadow_draw_calls.size(); i++)
        m_shadow_draw_calls[i]->setRenderState(false);
    for (unsigned b : m_dirty_buckets)
    {
        const unsigned slot_begin = b * share;
        const unsigned slot_end = std::min(slot_begin + share, m_slot_count);
        bool any_active = false;
        for (unsigned slot = slot_begin; slot < slot_end; slot++)
        {
            if (m_slot_states[slot].m_active)
            {
                any_active = true;
                break;
            }
        }
        if (!any_active)
            continue;

        for (unsigned slot = slot_begin; slot < slot_end; slot++)
        {
            if (m_slot_states[slot].m_active)
                m_shadow_draw_calls[slot]->prepareShadow(slot);
        }
        // The owner of the bucket also has to be prepared if it is a spot
        // light face which is not rendered: the other slots borrow its data
        m_shadow_draw_calls[slot_begin]->prepareShadow(slot_begin);
        for (irr::scene::ISceneNode* node : m_nodes)
            m_shadow_draw_calls[slot_begin]->addNode(node);
        m_shadow_draw_calls[slot_begin]->generate(m_vk);
    }

    m_cached_nodes.swap(m_next_cached_nodes);
    m_next_cached_nodes.clear();
    m_has_cached_nodes = true;
}   // generate

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::renderTile(VkCommandBuffer cmd,
                                       GEVulkanCamera* cam, unsigned slot,
                                       bool draw, bool& rebind_base_vertex)
{
    const bool vk_13 =
        m_vk->getPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_3;
    VkRect2D area;
    area.offset.x = (int32_t)((slot % m_tiles_per_row) * m_tile_size);
    area.offset.y = (int32_t)((slot / m_tiles_per_row) * m_tile_size);
    area.extent.width = m_tile_size;
    area.extent.height = m_tile_size;

    // The viewport and the scissor are in pixels of the atlas, the render
    // area of the rendering scope confines the clear to the tile so the
    // other tiles keep their content
    VkViewport vp;
    vp.x = (float)area.offset.x;
    vp.y = (float)area.offset.y;
    vp.width = (float)m_tile_size;
    vp.height = (float)m_tile_size;
    vp.minDepth = 0;
    vp.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &vp);
    vkCmdSetScissor(cmd, 0, 1, &area);

    VkClearValue clear_value;
    clear_value.depthStencil = {1.0f, 0};

    if (GEVulkanFeatures::supportsDynamicRendering())
    {
        VkRenderingAttachmentInfo depth_attachment = {};
        depth_attachment.sType =
            VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.imageView = m_frame_buffer_image_views[0];
        depth_attachment.imageLayout =
            VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue = clear_value;

        VkRenderingInfo rendering_info = {};
        rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        rendering_info.renderArea = area;
        rendering_info.layerCount = 1;
        rendering_info.pDepthAttachment = &depth_attachment;
        if (vk_13)
            vkCmdBeginRendering(cmd, &rendering_info);
        else
            vkCmdBeginRenderingKHR(cmd, &rendering_info);
    }
    else
    {
        VkRenderPassBeginInfo pass_info = {};
        pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        pass_info.renderPass = m_rtt_render_pass;
        pass_info.framebuffer = m_rtt_frame_buffer;
        pass_info.renderArea = area;
        pass_info.clearValueCount = 1;
        pass_info.pClearValues = &clear_value;
        vkCmdBeginRenderPass(cmd, &pass_info, VK_SUBPASS_CONTENTS_INLINE);
    }

    if (draw)
    {
        if (GEVulkanFeatures::supportsBindMeshTexturesAtOnce())
            m_shadow_draw_calls[slot]->bindAllMaterials(cmd);
        else
            rebind_base_vertex = true;
        renderShadowDrawCall(cmd, slot, cam, rebind_base_vertex);
    }

    if (GEVulkanFeatures::supportsDynamicRendering())
    {
        if (vk_13)
            vkCmdEndRendering(cmd);
        else
            vkCmdEndRenderingKHR(cmd);
    }
    else
        vkCmdEndRenderPass(cmd);
}   // renderTile

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::render(VkCommandBuffer cmd,
                                   GEVulkanCamera* cam)
{
    if (m_dirty_buckets.empty() && m_clear_tiles.empty())
        return;

    const bool dynamic = GEVulkanFeatures::supportsDynamicRendering();
    bool rebind_base_vertex = true;

    updateShadowDataDescriptorSets(cam);

    if (dynamic)
    {
        // One barrier for all the tiles, the layout can't be transitioned per
        // tile: the rendering scopes below only transition / clear the tiles
        // which changed
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.image = m_image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
            0, 0, NULL, 0, NULL, 1, &barrier);
    }

    const unsigned share = getSharingDrawCallCount();
    for (unsigned b : m_dirty_buckets)
    {
        const unsigned slot_begin = b * share;
        const unsigned slot_end = std::min(slot_begin + share, m_slot_count);
        for (unsigned slot = slot_begin; slot < slot_end; slot++)
        {
            if (!m_slot_states[slot].m_active)
                continue;
            renderTile(cmd, cam, slot, true, rebind_base_vertex);
        }
    }
    for (unsigned slot : m_clear_tiles)
        renderTile(cmd, cam, slot, false, rebind_base_vertex);

    if (dynamic)
    {
        VkImageMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
        barrier.image = m_image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, NULL, 0, NULL, 1, &barrier);
    }
}   // render

}   // namespace GE
