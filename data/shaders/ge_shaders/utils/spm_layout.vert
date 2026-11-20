layout(location = 0) in vec3 v_position;
#ifdef UNORM_PACKED_NORMAL
// The driver has no vertex support of VK_FORMAT_A2B10G10R10_SNORM_PACK32,
// the same bits are read as unorm and converted back to signed here
layout(location = 1) in vec4 v_normal_unorm;
#else
layout(location = 1) in vec4 v_normal;
#endif
layout(location = 2) in vec4 v_color;
layout(location = 3) in vec2 v_uv;
layout(location = 4) in vec2 v_uv_two;
#ifdef UNORM_PACKED_NORMAL
layout(location = 5) in vec4 v_tangent_unorm;
#else
layout(location = 5) in vec4 v_tangent;
#endif
layout(location = 6) in ivec4 v_joint;
layout(location = 7) in vec4 v_weight;

layout(location = 0) out vec4 f_vertex_color;
layout(location = 1) out vec2 f_uv;
layout(location = 2) out vec2 f_uv_two;
layout(location = 3) flat out int f_material_id;
layout(location = 4) out float f_hue_change;
layout(location = 5) out vec3 f_normal;
layout(location = 6) out vec3 f_tangent;
layout(location = 7) out vec3 f_bitangent;
layout(location = 8) out vec4 f_world_position;

#ifdef UNORM_PACKED_NORMAL
vec4 convertUnormToSnorm1010102(vec4 v)
{
    // Back to the integer value of each field, then two's complement
    vec4 i = floor(v * vec4(1023.0, 1023.0, 1023.0, 3.0) + 0.5);
    i -= step(vec4(512.0, 512.0, 512.0, 2.0), i) *
        vec4(1024.0, 1024.0, 1024.0, 4.0);
    return max(i / vec4(511.0, 511.0, 511.0, 1.0), vec4(-1.0));
}
#define v_normal convertUnormToSnorm1010102(v_normal_unorm)
#define v_tangent convertUnormToSnorm1010102(v_tangent_unorm)
#endif
