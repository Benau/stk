const float BUBBLE_REFLECTION_FADE_START = 25.0;
const float BUBBLE_REFLECTION_FADE_END = 45.0;

struct BubbleData
{
    vec3 color;
    vec3 normal;
    vec3 to_camera;
    float reflection_weight;
    float alpha;
};

BubbleData getBubbleData(vec4 base, vec3 normal, vec4 world_position)
{
    BubbleData bubble;
    bubble.color = convertColor(base.rgb);
    normal = normalize(normal);
    vec3 camera_pos = u_camera.m_inverse_view_matrix[3].xyz;
    bubble.to_camera = normalize(camera_pos - world_position.xyz);
    float facing = dot(normal, bubble.to_camera);
    float NdotV = clamp(abs(facing), 0.0, 1.0);
    float grazing = 1.0 - NdotV;

    // Two-sided shield. Orient the normal toward the camera before reflecting
    if (facing < 0.0)
        normal = -normal;
    bubble.normal = normal;
    float camera_distance = distance(camera_pos, world_position.xyz);
    float reflection_fade = 1.0 - smoothstep(BUBBLE_REFLECTION_FADE_START,
        BUBBLE_REFLECTION_FADE_END, camera_distance);
    bubble.reflection_weight = reflection_fade * grazing;
    // Match the RT shader's branch: do not blend a negligible reflection
    if (bubble.reflection_weight <= 0.001)
        bubble.reflection_weight = 0.0;
    bubble.alpha = base.a * (grazing * 0.4 + 0.6);
    return bubble;
}

vec3 getBubbleReflectionDirection(BubbleData bubble)
{
    return normalize(reflect(-bubble.to_camera, bubble.normal));
}

vec4 finishBubble(BubbleData bubble, vec4 world_position,
                  vec3 reflection_color)
{
    vec3 color = mix(bubble.color, reflection_color,
        bubble.reflection_weight);
    // Match transparent.frag's fog behavior
    if (u_global_light.m_fog_density > 0.0)
    {
        vec3 xpos = (u_camera.m_view_matrix * world_position).xyz;
        float fog = smoothstep(u_global_light.m_fog_range.x,
            u_global_light.m_fog_range.y, length(xpos));
        fog = min(fog, u_global_light.m_fog_color.a);
        color = mix(color, u_global_light.m_fog_color.rgb, fog);
    }

    // The alpha-blend pipeline expects premultiplied color
    return vec4(color * bubble.alpha, bubble.alpha);
}
