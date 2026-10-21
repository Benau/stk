// The grid of the ray traced shadow (GEVulkanRTShadow), shared by the compute
// shaders which write it and the lighting pass which filters it. The image is
// the size of the render target times a scale in [0.25, 1.0]. The grid is the
// one of the whole render target (not of every viewport), the texel t samples
// a full resolution pixel near (t + 0.5) / scale, and a viewport owns the
// texels which have that position in it, so the viewports of a splitscreen
// never share a texel.

// 1: the ray of a texel starts from a pixel of its footprint which is not the
// center (a fixed low discrepancy offset per texel), so the filter of the
// lighting pass averages different positions and the edge of a shadow is
// known to better than a texel, like MSAA. It adds a bit of noise if the
// filter is small
#ifndef RT_SHADOW_JITTER
#define RT_SHADOW_JITTER 0
#endif

// x0, y0, x1, y1 (exclusive) in pixels of the viewport of the camera, as
// GEVulkanRTShadow gets it (the viewport clamped to the render target)
ivec4 rtShadowPixelRect(vec4 viewport, ivec2 fbo_size)
{
    const ivec2 start = ivec2(viewport.xy);
    return ivec4(max(start, ivec2(0)),
        min(fbo_size, start + ivec2(viewport.zw)));
}

// The full resolution pixel which the ray of the texel starts from (in the
// viewport, pixel_rect is rtShadowPixelRect)
ivec2 rtShadowTapPixel(ivec2 texel, float scale, ivec4 pixel_rect)
{
    vec2 position = vec2(texel) + 0.5;
#if RT_SHADOW_JITTER
    position += fract(vec2(texel) * vec2(0.7548776662, 0.5698402910)) - 0.5;
#endif
    return clamp(ivec2(floor(position / scale)), pixel_rect.xy,
        pixel_rect.zw - 1);
}

// x0, y0, x1, y1 (exclusive) of the texels which belong to a rect in pixels
// (x0, y0, x1, y1 exclusive), same as the one of the cpu
ivec4 rtShadowTexelRect(ivec4 rect, float scale)
{
    return ivec4(ceil(vec4(rect) * scale - 0.5));
}
