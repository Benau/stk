// The test of a shadow view of a directional light: the frustum, and the
// objects flagged as small (see GEVulkanShadowDrawCall::isSmallForFarShadow)
// are skipped when the w of the camera of the level of detail is not zero
// (the far cascade)
#include "static_draws_frustum.glsl"

// ----------------------------------------------------------------------------
bool isInView(const Source source)
{
    if (u_lod.m_lod_camera.w != 0.0 && isSmall(source))
        return false;
    return isInFrustum(source);
}   // isInView
