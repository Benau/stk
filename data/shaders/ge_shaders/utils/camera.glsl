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
    // Used by GPU culling (see GECullingTool), what the vectors are depends on
    // the view and is named by the culling shader (see static_draws_view.glsl)
    vec4 m_culling[8];
} u_camera;
