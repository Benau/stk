// Needs utils/dof_common.glsl. The depth of the FBO is binding 2 of the set of
// GEVulkanFinalChain::getIOLayout(), the camera of the viewport is set 1
layout(set = 0, binding = 2) uniform sampler2D u_depth;

#include "camera.glsl"
#include "unproject_position.glsl"

// Distance from the camera to the point of the pixel. It is not the depth
// along the view direction: the point is unprojected from its own position on
// the screen with the inverse projection of the camera of the viewport (the
// same as the other passes), so what the pixel is at the side of the screen
// or in the center doesn't matter, and rotating the camera doesn't change the
// distance of a point (the length of a vector doesn't depend on the
// direction of the camera). Never reads outside of the viewport
float dofDistance(ivec2 pixel)
{
    pixel = clamp(pixel, pc.m_rect.xy, pc.m_rect.zw - 1);
    const float depth = texelFetch(u_depth, pixel, 0).x;
    // Reverse z, the clear value is 0
    if (depth == 0.0)
        return DOF_SKY_DISTANCE;
    const vec3 pos = getPosFromUVDepth(vec3(vec2(pixel) + 0.5, depth),
        u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
    return length(pos);
}
