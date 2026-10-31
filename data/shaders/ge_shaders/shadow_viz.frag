// The depth layers of the sun shadow map (GEVulkanShadowFBO), drawn by the
// 2D renderer when the shadow viz of the debug menu is on. Same 2x2 layout as
// the OpenGL renderer: the first layer is the top left quadrant, the last one
// is the bottom right. The Vulkan shadow map has 3 sun cascades, the last
// quadrant is empty.
layout(binding = 0) uniform sampler2DArray f_shadow_map;

layout(location = 0) in vec2 f_uv;

layout(location = 0) out vec4 o_color;

void main()
{
    // f_uv.y is 0 at the top of the screen in Vulkan
    int layer = int(f_uv.x >= 0.5) + int(f_uv.y >= 0.5) * 2;
    if (layer >= 3)
    {
        o_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    vec2 uv = fract(f_uv * 2.0);
    float depth = texture(f_shadow_map, vec3(uv, float(layer))).r;
    o_color = vec4(vec3(depth), 1.0);
}
