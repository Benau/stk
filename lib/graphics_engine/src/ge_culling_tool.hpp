#ifndef HEADER_GE_CULLING_TOOL_HPP
#define HEADER_GE_CULLING_TOOL_HPP

#include "aabbox3d.h"
#include "quaternion.h"
#include "matrix4.h"

namespace irr
{
    namespace scene { class ISceneNode; }
}

namespace GE
{
class GESPMBuffer;
class GEVulkanCameraSceneNode;
class GEVulkanShadowFBO;
class GEVulkanOmniShadowFBO;

class GECullingTool
{
private:
    bool m_skip_near_plane;

    irr::core::quaternion m_frustum[6];

    irr::core::aabbox3df m_cam_bbox;

    const GEVulkanOmniShadowFBO* m_omni_sfbo;

    // Inclusive range [m_bucket_light_start, m_bucket_light_end) of the
    // lights in m_omni_sfbo that this draw call's bucket covers.
    // Only valid when m_omni_sfbo != NULL.
    unsigned m_bucket_light_start;
    unsigned m_bucket_light_end;

public:
    // ------------------------------------------------------------------------
    GECullingTool() : m_skip_near_plane(false), m_omni_sfbo(NULL),
                      m_bucket_light_start(0), m_bucket_light_end(0)         {}
    // ------------------------------------------------------------------------
    void init(GEVulkanCameraSceneNode* cam);
    // ------------------------------------------------------------------------
    void initShadow(const GEVulkanShadowFBO* sfbo, unsigned layer);
    // ------------------------------------------------------------------------
    bool isCulled(irr::core::aabbox3df& bb);
    // ------------------------------------------------------------------------
    bool isCulled(const irr::core::vector3df& center, float radius);
    // ------------------------------------------------------------------------
    bool isCulled(GESPMBuffer* buffer, irr::scene::ISceneNode* node);
    // ------------------------------------------------------------------------
    void* getFrustumData()                          { return &m_frustum[0].X; }
};   // GECullingTool

}

#endif
