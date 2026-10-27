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
#include "utils/bubble_utils.glsl"
#include "utils/pbr_light.glsl"
#include "utils/rt_utils.glsl"
#include "utils/rt_scene.glsl"
#include "utils/rt_reflection.glsl"

void main()
{
    vec4 base = sampleMeshTexture0(f_material_id, f_uv) * f_vertex_color;
    if (base.a <= 0.0)
        discard;

    BubbleData bubble = getBubbleData(base, f_normal, f_world_position);
    vec3 reflection_color = vec3(0.0);
    if (bubble.reflection_weight > 0.0)
    {
        vec3 reflection_dir = getBubbleReflectionDirection(bubble);
        // A miss uses the skybox
        // Approximate gamma 1/2.2 with sqrt to avoid pow()
        reflection_color = sqrt(texture(u_skybox_texture, reflection_dir).rgb);
        vec3 hit_color;
        if (traceReflection(f_world_position.xyz, reflection_dir, hit_color))
            reflection_color = convertColor(hit_color);
    }
    o_color = finishBubble(bubble, f_world_position, reflection_color);
}
