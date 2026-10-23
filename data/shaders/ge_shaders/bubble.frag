// Ray-traced reflection for the translucent bubble shields
// Uses the same RT scene/layout and hit shading as solid_rt.frag and
// displace_mask_rt.frag. Select it only for the RT pipeline (set 0: RT scene,
// set 1: camera/global light, set 3: environment, set 4: bindless mesh textures)
#extension GL_EXT_ray_query : require
#define MESH_TEXTURES_SET 4

layout(location = 0) in vec4 f_vertex_color;
layout(location = 1) in vec2 f_uv;
layout(location = 3) flat in int f_material_id;
layout(location = 5) in vec3 f_normal;
layout(location = 8) in vec4 f_world_position;

layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"

layout(set = 3, binding = 0) uniform samplerCube u_diffuse;
layout(set = 3, binding = 1) uniform samplerCube u_specular;
layout(set = 3, binding = 2) uniform samplerCube u_skybox_texture;
layout(set = 3, binding = 3) uniform sampler2DArrayShadow u_shadow_map;

#include "utils/sample_mesh_texture.glsl"
#include "utils/pbr_utils.glsl"
#include "utils/get_vertex_color.glsl"
#include "../utils/rgb_conversion.frag"
#include "utils/global_light_data.glsl"
#include "utils/pbr_light.glsl"
#include "utils/rt_utils.glsl"
#include "utils/rt_scene.glsl"
#include "utils/rt_reflection.glsl"

// Only trace nearby shields. The fade is applied to the reflected contribution,
// and the branch skips the ray query completely beyond the end distance
const float BUBBLE_RT_FADE_START = 25.0;
const float BUBBLE_RT_FADE_END = 45.0;

void main()
{
    vec4 base = sampleMeshTexture0(f_material_id, f_uv) * f_vertex_color;
    if (base.a <= 0.0)
        discard;

    vec3 color = convertColor(base.rgb);
    vec3 normal = normalize(f_normal);
    vec3 camera_pos = u_camera.m_inverse_view_matrix[3].xyz;
    vec3 to_camera = normalize(camera_pos - f_world_position.xyz);
    float facing = dot(normal, to_camera);
    float NdotV = clamp(abs(facing), 0.0, 1.0);
    float grazing = 1.0 - NdotV;

    // Two-sided shield. Orient the normal toward the camera before reflecting
    if (facing < 0.0)
        normal = -normal;

    float camera_distance = distance(camera_pos, f_world_position.xyz);
    float reflection_fade = 1.0 - smoothstep(BUBBLE_RT_FADE_START,
        BUBBLE_RT_FADE_END, camera_distance);
    float reflection_weight = reflection_fade * grazing;

    if (reflection_weight > 0.001)
    {
        vec3 reflection_dir = normalize(reflect(-to_camera, normal));
        // A miss uses the skybox
        vec3 reflection_color = texture(u_skybox_texture, reflection_dir).rgb;
        vec3 hit_color;
        if (traceReflection(f_world_position.xyz, reflection_dir, hit_color))
            reflection_color = convertColor(hit_color);
        color = mix(color, reflection_color, reflection_weight);
    }

    // Keep the bubble's texture alpha, with the view-angle opacity
    float alpha = base.a * (grazing * 0.4 + 0.6);

    // Match transparent.frag's fog behavior
    if (u_global_light.m_fog_density > 0.0)
    {
        vec3 xpos = (u_camera.m_view_matrix * f_world_position).xyz;
        float fog = smoothstep(u_global_light.m_fog_range.x,
            u_global_light.m_fog_range.y, length(xpos));
        fog = min(fog, u_global_light.m_fog_color.a);
        color = mix(color, u_global_light.m_fog_color.rgb, fog);
    }

    // The alpha-blend pipeline expects premultiplied color
    o_color = vec4(color * alpha, alpha);
}
