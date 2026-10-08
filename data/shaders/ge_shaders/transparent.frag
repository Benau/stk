layout(location = 0) in vec4 f_vertex_color;
layout(location = 1) in vec2 f_uv;
layout(location = 3) flat in int f_material_id;
layout(location = 8) in vec4 f_world_position;

layout(location = 0) out vec4 o_color;

#include "utils/constants_utils.glsl"
#include "utils/sample_mesh_texture.glsl"
#ifdef PBR_ENABLED
#include "utils/camera.glsl"
#include "utils/global_light_data.glsl"
#include "utils/unproject_position.glsl"
#endif

void main()
{
    vec4 color = sampleMeshTexture0(f_material_id, f_uv) * f_vertex_color;
    vec3 mixed_color = color.xyz;
    float alpha = color.w;
#ifdef PBR_ENABLED
    mixed_color = convertColor(mixed_color);
    // Linear fog, and the solid materials have exponential fog in
    // handle_pbr.glsl
    if (u_global_light.m_fog_density > 0.0)
    {
        vec3 xpos = (u_camera.m_view_matrix * f_world_position).xyz;
        float fog = smoothstep(u_global_light.m_fog_range.x,
            u_global_light.m_fog_range.y, length(xpos));
        fog = min(fog, u_global_light.m_fog_color.a);
        mixed_color = mix(mixed_color, u_global_light.m_fog_color.rgb, fog);
    }
#endif
    o_color = vec4(mixed_color * alpha, alpha);
}
