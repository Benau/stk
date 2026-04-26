#include "ge_culling_tool.hpp"

#include "ge_main.hpp"
#include "ge_spm_buffer.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_omni_shadow_fbo.hpp"

#include "ISceneNode.h"
#include <algorithm>
#include <cassert>

namespace GE
{
// ----------------------------------------------------------------------------
void GECullingTool::init(GEVulkanCameraSceneNode* cam)
{
    mathPlaneFrustumf(&m_frustum[0].X, cam->getPVM());
    m_cam_bbox = cam->getViewFrustum()->getBoundingBox();
}   // init

// ----------------------------------------------------------------------------
void GECullingTool::initShadow(const GEVulkanShadowFBO* sfbo, unsigned layer)
{
    if (!m_skip_near_plane)
        m_skip_near_plane = true;
    mathPlaneFrustumf(&m_frustum[0].X,
        sfbo->getShadowProjectionViewMatrix(layer));
    m_omni_sfbo = dynamic_cast<const GEVulkanOmniShadowFBO*>(sfbo);
    const int pointlight_layer = (int)layer - (int)sfbo->getLayerOffset();
    if (m_omni_sfbo && pointlight_layer >= 0)
    {
        int layer_owner = sfbo->getLayerOwner(pointlight_layer);
        assert(layer_owner >= 0);
        m_bucket_light_start = layer_owner / OMNI_FACES_PER_LIGHT;
        m_bucket_light_end = std::min(
            (layer_owner + sfbo->getSharingDrawCallCount()) /
            OMNI_FACES_PER_LIGHT, (unsigned)m_omni_sfbo->getLights().size());
    }
    else
    {
        m_omni_sfbo          = NULL;
        m_bucket_light_start = 0;
        m_bucket_light_end   = 0;
    }
}   // initShadow

// ----------------------------------------------------------------------------
bool GECullingTool::isCulled(const irr::core::vector3df& center, float radius)
{
    for (int i = m_skip_near_plane ? 1 : 0; i < 6; i++)
    {
        irr::core::quaternion q(center.X, center.Y, center.Z, 1.0f);
        if (m_frustum[i].dotProduct(q) < -radius)
            return true;
    }
    return false;
}   // isCulled

// ----------------------------------------------------------------------------
bool GECullingTool::isCulled(irr::core::aabbox3df& bb)
{
    if (m_omni_sfbo)
    {
        auto& lights = m_omni_sfbo->getLights();
        for (unsigned i = m_bucket_light_start; i < m_bucket_light_end; i++)
        {
            const core::vector3df& lp = lights[i].m_position;
            // Clamp sphere center to box bounds to find the closest point on the box
            float closest_x = std::max(bb.MinEdge.X, std::min(lp.X, bb.MaxEdge.X));
            float closest_y = std::max(bb.MinEdge.Y, std::min(lp.Y, bb.MaxEdge.Y));
            float closest_z = std::max(bb.MinEdge.Z, std::min(lp.Z, bb.MaxEdge.Z));

            // Calculate squared distance from sphere center to the closest point
            float dx = lp.X - closest_x;
            float dy = lp.Y - closest_y;
            float dz = lp.Z - closest_z;

            float dmin = dx * dx + dy * dy + dz * dz;

            // If the squared distance is less than or equal to the squared radius, visible
            if (dmin <= (lights[i].m_radius * lights[i].m_radius))
                return false;
        }
        return true;
    }

    if (!m_skip_near_plane && !m_cam_bbox.intersectsWithBox(bb))
        return true;

    using namespace irr;
    using namespace core;
    quaternion edges[8] =
    {
        quaternion(bb.MinEdge.X, bb.MinEdge.Y, bb.MinEdge.Z, 1.0f),
        quaternion(bb.MaxEdge.X, bb.MinEdge.Y, bb.MinEdge.Z, 1.0f),
        quaternion(bb.MinEdge.X, bb.MaxEdge.Y, bb.MinEdge.Z, 1.0f),
        quaternion(bb.MaxEdge.X, bb.MaxEdge.Y, bb.MinEdge.Z, 1.0f),
        quaternion(bb.MinEdge.X, bb.MinEdge.Y, bb.MaxEdge.Z, 1.0f),
        quaternion(bb.MaxEdge.X, bb.MinEdge.Y, bb.MaxEdge.Z, 1.0f),
        quaternion(bb.MinEdge.X, bb.MaxEdge.Y, bb.MaxEdge.Z, 1.0f),
        quaternion(bb.MaxEdge.X, bb.MaxEdge.Y, bb.MaxEdge.Z, 1.0f)
    };

    for (int i = m_skip_near_plane ? 1 : 0; i < 6; i++)
    {
        bool culled = true;
        for (int j = 0; j < 8; j++)
        {
            if (m_frustum[i].dotProduct(edges[j]) >= 0.0)
            {
                culled = false;
                break;
            }
        }
        if (culled)
            return true;
    }
    return false;
}   // isCulled

// ----------------------------------------------------------------------------
bool GECullingTool::isCulled(GESPMBuffer* buffer, irr::scene::ISceneNode* node)
{
    irr::core::aabbox3df bb = buffer->getBoundingBox();
    node->getAbsoluteTransformation().transformBoxEx(bb);
    return isCulled(bb);
}   // isCulled

}
