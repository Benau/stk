#ifndef HEADER_GE_VULKAN_PREV_CAMERA_HPP
#define HEADER_GE_VULKAN_PREV_CAMERA_HPP

#include "ge_vulkan_camera_scene_node.hpp"

#include <cstring>

namespace GE
{
// The camera of a viewport in the last frame, which the history of a frame
// cache (GEConfig::m_frame_cache) was rendered with: the irradiance of
// GEVulkanSurfelGI and the reflection of GEVulkanStochasticReflection. Same as
// PrevCamera of utils/prev_camera.glsl (std430), which reprojects a position
// of this frame to the pixel of the history which saw it.
struct GEVulkanPrevCamera
{
    float m_projection_view_matrix[16];
    // A pixel of the history back to a position (with its distance)
    float m_inverse_projection_view_matrix[16];
    // The m_viewport of GEVulkanCameraUBO (the offset and the size)
    float m_viewport[4];
    float m_position[4];
    // ------------------------------------------------------------------------
    void set(const GEVulkanCameraUBO* ubo)
    {
        memcpy(m_projection_view_matrix,
            ubo->m_projection_view_matrix.pointer(),
            sizeof(m_projection_view_matrix));
        memcpy(m_inverse_projection_view_matrix,
            ubo->m_inverse_projection_view_matrix.pointer(),
            sizeof(m_inverse_projection_view_matrix));
        m_viewport[0] = ubo->m_viewport.UpperLeftCorner.X;
        m_viewport[1] = ubo->m_viewport.UpperLeftCorner.Y;
        m_viewport[2] = ubo->m_viewport.LowerRightCorner.X;
        m_viewport[3] = ubo->m_viewport.LowerRightCorner.Y;
        m_position[0] = ubo->m_inverse_view_matrix[12];
        m_position[1] = ubo->m_inverse_view_matrix[13];
        m_position[2] = ubo->m_inverse_view_matrix[14];
        m_position[3] = 1.0f;
    }
};   // GEVulkanPrevCamera

static_assert(sizeof(GEVulkanPrevCamera) == 160, "PrevCamera is 160 bytes in "
    "the shaders");

}

#endif
