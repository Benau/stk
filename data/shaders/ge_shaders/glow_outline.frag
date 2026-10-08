// Color of the glow of a batch of meshes (linear, see GEVulkanDrawCall::
// renderGlowOutline), pushed after the 16 bytes which materials may use at
// the start of the push constants (for example the wind direction of grass)
layout(push_constant) uniform Constants
{
    layout(offset = 16) vec3 m_glow_color;
} u_push_constants;

layout(location = 0) out vec4 o_color;

void main()
{
    // Alpha is the coverage of the meshes, it's used to hide the glow behind
    // them in lighting_composite.frag
    o_color = vec4(u_push_constants.m_glow_color, 1.0);
}
