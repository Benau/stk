#include "ge_vulkan_omni_shadow_fbo.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_light_handler.hpp"
#include "ge_vulkan_omni_shadow_draw_call.hpp"

#include "ILightSceneNode.h"

#include <algorithm>
#include <cassert>
#include <cmath>

// M_PI may not be defined on MSVC without this.
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace GE
{

// ----------------------------------------------------------------------------
// Shadow mapping strategy per light type
//
// Point light  :  6 cubemap faces at 90° FOV each.
//                 Face selected at sample time by getFaceIndex(fragPos - lightPos).
//   0 : +X
//   1 : -X
//   2 : +Y
//   3 : -Y
//   4 : +Z
//   5 : -Z
//
// All faces use a 90° symmetric perspective projection.
//
// Spot light   :  1 face (slot 0) using a perspective whose full-angle FOV
//                 equals 2 * OuterCone.  The view is aligned with the cone
//                 direction.  Faces 1-5 are disabled.
//                 The shader detects spotlights via sscale != 0 and always
//                 samples face 0, skipping getFaceIndex entirely.
//
// layer = light_id * OMNI_FACES_PER_LIGHT + face + getLayerOffset()
// ----------------------------------------------------------------------------

static inline float deg2rad(float d)
{
    return d * (float)(M_PI / 180.0);
}   // deg2rad

// A small padding added to the spotlight FOV so cone-edge fragments are
// never clipped by projection precision.
static const float kSpotFovPaddingDeg = 1.0f;

// ----------------------------------------------------------------------------
// Standard perspective matrix (90° cube face)
// ----------------------------------------------------------------------------
static irr::core::matrix4 buildPerspective(float fov,
                                           float near_plane,
                                           float far_plane)
{
    float tan_half = tanf(deg2rad(fov) * 0.5f);

    irr::core::matrix4 m;

    m(0,0) = 1.0f / tan_half;
    m(1,1) = 1.0f / tan_half;

    m(2,2) = far_plane / (far_plane - near_plane);
    m(2,3) = 1.0f;

    m(3,2) = -(far_plane * near_plane) / (far_plane - near_plane);
    m(3,3) = 0.0f;

    return m;
}   // buildPerspective

// ----------------------------------------------------------------------------
// Cubemap face view matrices
// ----------------------------------------------------------------------------
static irr::core::matrix4 buildFaceViewMatrix(unsigned face,
                                              const irr::core::vector3df& pos)
{
    irr::core::vector3df target;
    irr::core::vector3df up;

    switch (face)
    {
        case 0: // +X
            target = pos + irr::core::vector3df(1,0,0);
            up     = irr::core::vector3df(0,-1,0);
            break;

        case 1: // -X
            target = pos + irr::core::vector3df(-1,0,0);
            up     = irr::core::vector3df(0,-1,0);
            break;

        case 2: // +Y
            target = pos + irr::core::vector3df(0,1,0);
            up     = irr::core::vector3df(0,0,1);
            break;

        case 3: // -Y
            target = pos + irr::core::vector3df(0,-1,0);
            up     = irr::core::vector3df(0,0,-1);
            break;

        case 4: // +Z
            target = pos + irr::core::vector3df(0,0,1);
            up     = irr::core::vector3df(0,-1,0);
            break;

        case 5: // -Z
            target = pos + irr::core::vector3df(0,0,-1);
            up     = irr::core::vector3df(0,-1,0);
            break;

        default:
            assert(false);
            target = pos;
            up     = irr::core::vector3df(0,1,0);
    }

    irr::core::matrix4 view;
    view.buildCameraLookAtMatrixLH(pos, target, up);

    return view;
}   // buildFaceViewMatrix

// ----------------------------------------------------------------------------
// buildSpotViewMatrix – a camera at pos looking along dir.
// ----------------------------------------------------------------------------
static irr::core::matrix4 buildSpotViewMatrix(const irr::core::vector3df& pos,
                                              const irr::core::vector3df& dir)
{
    // Choose an up vector that is never parallel to dir.
    irr::core::vector3df up(0.0f, 1.0f, 0.0f);
    if (fabsf(dir.Y) > 0.99f)
        up = irr::core::vector3df(1.0f, 0.0f, 0.0f);

    irr::core::matrix4 view;
    view.buildCameraLookAtMatrixLH(pos, pos + dir, up);
    return view;
}   // buildSpotViewMatrix

// ----------------------------------------------------------------------------
GEVulkanOmniShadowFBO::GEVulkanOmniShadowFBO(GEVulkanDriver* vk,
                                             unsigned shadow_size,
                                             GEVulkanDrawCall* master_dc,
                                             irr::scene::ILightSceneNode* sun)
                     : GEVulkanShadowFBO(vk, shadow_size, master_dc, sun,
                                         getGEConfig()->m_shadow_type ==
                                         GST_COMBINED ?
                                         getGEConfig()->m_point_shadow_limit *
                                         OMNI_FACES_PER_LIGHT + GVSCC_COUNT:
                                         getGEConfig()->m_point_shadow_limit *
                                         OMNI_FACES_PER_LIGHT)
{
    m_light_handler = NULL;
}   // GEVulkanOmniShadowFBO

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::createDrawCalls()
{
    for (unsigned i = 0; i < m_layer_count; i++)
    {
        m_shadow_draw_calls.push_back(std::unique_ptr<GEVulkanShadowDrawCall>(
            new GEVulkanOmniShadowDrawCall(this, i)));
    }
}   // createDrawCalls

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::buildSingleFaceMatrices(unsigned light_id)
{
    const irr::core::vector3df& pos = m_light_handler->getLightPosition(light_id);
    const float  radius             = m_light_handler->getLightRadius(light_id);
    const float  outer_cone         = m_light_handler->getLightOuterCone(light_id);
    const irr::core::vector3df dir  = m_light_handler->getLightDirection(light_id);
    m_lights.push_back({core::vector3df(pos.X, pos.Y, pos.Z), radius});

    // Full-angle FOV = 2 * outer_cone (radians → degrees) + padding.
    const float fov_deg =
        2.0f * outer_cone * (float)(180.0 / M_PI) + kSpotFovPaddingDeg;

    const irr::core::matrix4 proj = buildPerspective(fov_deg, 0.2f, radius);
    const irr::core::matrix4 view = buildSpotViewMatrix(pos, dir);

    // Only face slot 0 is used for spotlights.
    const unsigned layer = light_id * OMNI_FACES_PER_LIGHT + 0 + getLayerOffset();
    m_shadow_projection_matrices[layer] = proj * view;
    GEVulkanCameraUBO& ubo = m_shadow_camera_ubo_data[layer];
    ubo.m_projection_view_matrix = m_shadow_projection_matrices[layer];
    m_light_handler->setLightShadowMatrices(&ubo, light_id, 0);
    // Slots 1-5 stay uninitialised; those draw calls are disabled below so
    // the GPU never samples those layers for this light.
}   // buildSingleFaceMatrices

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::buildSixFaceMatrices(unsigned light_id)
{
    const irr::core::vector3df& pos = m_light_handler->getLightPosition(light_id);
    const float radius              = m_light_handler->getLightRadius(light_id);
    const irr::core::matrix4 proj   = buildPerspective(90.0f, 0.2f, radius);
    m_lights.push_back({core::vector3df(pos.X, pos.Y, pos.Z), radius});

    for (unsigned face = 0; face < OMNI_FACES_PER_LIGHT; face++)
    {
        const unsigned layer =
            light_id * OMNI_FACES_PER_LIGHT + face + getLayerOffset();
        const irr::core::matrix4 view = buildFaceViewMatrix(face, pos);
        m_shadow_projection_matrices[layer] = proj * view;
        GEVulkanCameraUBO& ubo = m_shadow_camera_ubo_data[layer];
        ubo.m_projection_view_matrix = m_shadow_projection_matrices[layer];
        m_light_handler->setLightShadowMatrices(&ubo, light_id, face);
    }
}   // buildSixFaceMatrices

// ----------------------------------------------------------------------------
void GEVulkanOmniShadowFBO::generate()
{
    assert(m_light_handler != NULL);

    const unsigned shadow_limit = getGEConfig()->m_point_shadow_limit;
    const unsigned light_count = std::min(m_light_handler->getLightCount(),
        shadow_limit);
    // Non-occluded lights are stable-partitioned to the front of the rendered
    // light array by GEVulkanLightHandler::generate().  Shadow cubemaps are
    // only built for those; occluded lights get depth layers cleared to 1.0.
    const unsigned non_occluded = std::min(
        m_light_handler->getNonOccludedLightCount(), light_count);

    // -------------------------------------------------------------------------
    // Phase 1 – build projection-view matrices for active, non-occluded lights.
    // -------------------------------------------------------------------------
    m_lights.clear();
    for (unsigned i = 0; i < non_occluded; i++)
    {
        if (m_light_handler->getLightIsSpot(i))
            buildSingleFaceMatrices(i);
        else
            buildSixFaceMatrices(i);
    }

    // -------------------------------------------------------------------------
    // Phase 2 – schedule draw calls.
    //
    // Important ordering note: GEVulkanOmniShadowDrawCall::prepareShadow()
    // resets m_render_state to true internally.  For any face we want to
    // disable we must NOT call prepareShadow() first; we call setRenderState()
    // directly instead.  The two paths below never overlap, so the ordering is
    // safe.
    // -------------------------------------------------------------------------
    for (unsigned i = 0; i < shadow_limit; i++)
    {
        const bool single_face = (i < non_occluded) &&
            m_light_handler->getLightIsSpot(i);

        for (unsigned face = 0; face < OMNI_FACES_PER_LIGHT; face++)
        {
            const unsigned pointlight_layer = i * OMNI_FACES_PER_LIGHT + face;
            const unsigned layer = pointlight_layer + getLayerOffset();

            if (i >= non_occluded || (single_face && face != 0))
            {
                // Disabled: clear the depth layer to 1.0 (fully lit) via
                // VkRenderPass load-op, but submit no geometry.
                m_shadow_draw_calls[layer]->setRenderState(false);
                continue;
            }
            m_shadow_draw_calls[layer]->prepareShadow(layer);

            // Build geometry only for the owner of each bucket (face 0 of the
            // first light in the bucket).  All other layers in the bucket
            // share this geometry at render time.
            if ((pointlight_layer % getSharingDrawCallCount()) == 0)
            {
                for (irr::scene::ISceneNode* node : m_nodes)
                    m_shadow_draw_calls[layer]->addNode(node);
                m_shadow_draw_calls[layer]->generate(m_vk);
            }
        }
    }
}   // generate

}   // namespace GE
