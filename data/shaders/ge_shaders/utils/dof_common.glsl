// Shared by the compute shaders of GEVulkanDepthOfField (dof_downsample.comp,
// dof_blur.comp and dof_composite.comp)
layout(push_constant) uniform PushConstants
{
    // x0, y0, x1, y1 (exclusive) in pixels of the full size image: the
    // viewport of one camera. Nothing is read or written outside of it
    ivec4 m_rect;
    // Same in pixels of the half size images: only the pixels which are
    // entirely inside of m_rect, so the viewports never share a pixel
    ivec4 m_half_rect;
    // x: distance of the center of the zone in focus, y: half of its length,
    // z: length behind it where the blur grows to the maximum, w: same in
    // front of it (0 = no blur there), all in world units
    vec4 m_focus;
    // x: radius of the maximum blur in half size pixels (it has the height of
    // the viewport relative to the reference height in it, so the blur is the
    // same part of the screen at any resolution, render scale and
    // splitscreen), y and z: direction of the blur (dof_blur.comp)
    vec4 m_params;
} pc;

// Distance of the pixels without geometry (the sky)
const float DOF_SKY_DISTANCE = 1.0e6;

// The signed circle of confusion in [-1, 1] from the distance of a point to
// the camera: 0 in the zone in focus, > 0 behind it (far blur), < 0 in front
// of it (near blur). Its absolute value is the part of the maximum radius
float dofCoc(float dist)
{
    const float far_start = pc.m_focus.x + pc.m_focus.y;
    const float near_start = pc.m_focus.x - pc.m_focus.y;
    if (dist > far_start)
    {
        const float t = clamp((dist - far_start) / max(pc.m_focus.z, 1.0e-3),
            0.0, 1.0);
        return t * t * (3.0 - 2.0 * t);
    }
    if (pc.m_focus.w > 0.0 && dist < near_start)
    {
        const float t = clamp((near_start - dist) / pc.m_focus.w, 0.0, 1.0);
        return -(t * t * (3.0 - 2.0 * t));
    }
    return 0.0;
}
