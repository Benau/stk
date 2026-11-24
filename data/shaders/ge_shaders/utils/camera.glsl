#ifndef CAMERA_SET
#define CAMERA_SET 1
#endif
layout(std140, set = CAMERA_SET, binding = 0) uniform CameraBuffer
{
    mat4 m_view_matrix;
    mat4 m_projection_matrix;
    mat4 m_inverse_view_matrix;
    mat4 m_inverse_projection_matrix;
    mat4 m_projection_view_matrix;
    mat4 m_inverse_projection_view_matrix;
    vec4 m_viewport;
    vec4 m_screensize;
    // Used by GPU culling (see GECullingTool)
    vec4 m_frustum[6];
    vec4 m_view_min_edge;
    vec4 m_view_max_edge;
} u_camera;
