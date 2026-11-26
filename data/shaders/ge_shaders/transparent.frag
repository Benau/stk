layout(location = 0) in vec4 f_vertex_color;
layout(location = 1) in vec2 f_uv;
layout(location = 3) flat in int f_material_id;
layout(location = 8) in vec4 f_world_position;

layout(location = 0) out vec4 o_color;

#include "utils/constants_utils.glsl"
#include "utils/sample_mesh_texture.glsl"
#ifdef PBR_ENABLED
#include "utils/camera.glsl"
#if defined(RAY_QUERY_ENABLED) && defined(SPLIT)
layout(location = 5) in vec3 f_normal;
#define SURFEL_GI_FORWARD
#include "utils/surfel_gi_volume.glsl"
#endif
#include "utils/global_light_data.glsl"
#include "utils/unproject_position.glsl"
#endif

void main()
{
    vec4 color = sampleMeshTexture0(f_material_id, f_uv) * f_vertex_color;
    vec3 mixed_color = color.xyz;
    float alpha = color.w;
#ifdef PBR_ENABLED
    #ifdef SURFEL_GI_FORWARD
    if (u_surfel_gi)
    {
        vec4 indirect = lookupVolume(f_world_position.xyz, normalize(f_normal), true);
        mixed_color *= mix(vec3(1.0), indirect.rgb, indirect.a);
    }
    #endif
    mixed_color += 0.2 * color.rgb;
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
