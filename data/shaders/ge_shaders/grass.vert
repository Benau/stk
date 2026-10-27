#include "utils/camera.glsl"
#include "utils/spm_data.glsl"
#include "utils/spm_layout.vert"
#include "utils/grass_wind.glsl"
#include "../utils/get_world_location.vert"

layout(push_constant) uniform Constants
{
    vec3 m_wind_direction;
} u_push_constants;

void main()
{
    const vec3 offset = getGrassWindOffset(u_push_constants.m_wind_direction,
        v_position.y);
#ifdef RAY_QUERY_ENABLED
    // The wind is applied in the model space (the translation is not moved
    // by it), so the BLAS of the grass for ray tracing can be skinned once for
    // all the instances of a mesh with the same offset (see grass_wind.glsl),
    // and the instance transform (the one of this node) puts it in the world
    vec4 v_world_position = getWorldPosition(
        u_object_buffer.m_objects[gl_InstanceIndex].m_translation,
        u_object_buffer.m_objects[gl_InstanceIndex].m_rotation,
        u_object_buffer.m_objects[gl_InstanceIndex].m_scale,
        v_position + offset * v_color.r);
#else
    vec4 v_world_position = getWorldPosition(
        u_object_buffer.m_objects[gl_InstanceIndex].m_translation + offset *
        v_color.r,
        u_object_buffer.m_objects[gl_InstanceIndex].m_rotation,
        u_object_buffer.m_objects[gl_InstanceIndex].m_scale, v_position);
#endif
    f_world_position = v_world_position;
    gl_Position = u_camera.m_projection_view_matrix * v_world_position;
    f_vertex_color = vec4(1.0);
    f_uv = v_uv;
    f_uv_two = v_uv_two;
    f_material_id = u_object_buffer.m_objects[gl_InstanceIndex].m_material_id;
// With MATERIAL_ID_UNIFORM the fragment shader uses a push constant instead
#if defined(BIND_MESH_TEXTURES_AT_ONCE) && defined(SHADER_DRAW_PARAMETERS) && !defined(MATERIAL_ID_UNIFORM)
    if (f_material_id < 0)
        f_material_id = u_material_ids.m_material_id[gl_DrawIDARB];
#endif
    f_hue_change = u_object_buffer.m_objects[gl_InstanceIndex].m_hue_change;
#ifdef PBR_ENABLED
    f_normal = rotateVector(u_object_buffer.m_objects[gl_InstanceIndex].m_rotation, v_normal.xyz);
#endif
}
