// Skybox-only reflection for translucent bubble shields when ray queries are
// not available. Uses m_skybox_layout (set 0: mesh textures, set 1: camera and
// global light, set 2: environment).
layout(location = 0) in vec4 f_vertex_color;
layout(location = 1) in vec2 f_uv;
layout(location = 3) flat in int f_material_id;
layout(location = 5) in vec3 f_normal;
layout(location = 8) in vec4 f_world_position;

layout(set = 2, binding = 2) uniform samplerCube u_skybox_texture;
layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/sample_mesh_texture.glsl"
#include "../utils/rgb_conversion.frag"
#include "utils/global_light_data.glsl"
#include "utils/bubble_utils.glsl"

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
        reflection_color = texture(u_skybox_texture, reflection_dir).rgb;
        // Approximate gamma 1/2.2 with sqrt to avoid pow()
        if (u_deferred)
            reflection_color = sqrt(reflection_color);
    }
    o_color = finishBubble(bubble, f_world_position, reflection_color);
}
