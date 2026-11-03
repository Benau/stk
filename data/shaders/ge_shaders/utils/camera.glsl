layout(std140, set = 1, binding = 0) uniform CameraBuffer
{
    mat4 m_view_matrix;
    mat4 m_projection_matrix;
    mat4 m_inverse_view_matrix;
    mat4 m_inverse_projection_matrix;
    mat4 m_projection_view_matrix;
    mat4 m_inverse_projection_view_matrix;
    vec4 m_viewport;
    vec4 m_screensize;
    // x != 0 renders the dual paraboloid shadow map (spm.vert)
    vec4 m_shadow_flags;
#ifdef GPU_CULLING
    vec4 m_frustum[6];
#endif
} u_camera;
