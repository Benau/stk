// Filters the visibility of a layer of GEVulkanRTShadow (layer 0 is the sun,
// layer 1 + i is the light i, a scaled grid, see rt_shadow_grid.glsl) to the
// full resolution pixel of the lighting pass: a tent of up to
// RT_SHADOW_FILTER_RADIUS texels (1 is bilinear, 2 is 4x4 texels, like a PCF
// kernel of a shadow map), times how far the surface of the texel is from the
// plane of the pixel (so the shadow of a kart is not spread to the ground
// behind it or the opposite), and the texels which belong to another viewport
// are ignored. The radius of the tent is RT_SHADOW_FILTER_WORLD_RADIUS in the
// world (so the softness is the same for a near and a far object, like a PCF
// kernel which is in the space of the shadow map) but at least 1 texel and at
// most RT_SHADOW_FILTER_RADIUS texels
//
// A texel of every layer has the visibility (x) and the linear depth (y, the
// distance along the view direction, 0 for the sky) of the pixel which its ray
// started from, written once by rt_shadow.comp, so a tap is one fetch and a few
// multiply-adds instead of the depth buffer and an unprojection
//
// Needs u_camera, u_depth (only the size), u_rt_shadow_texture,
// u_rt_shadow_scale and rt_shadow_grid.glsl
#ifndef RT_SHADOW_FILTER_RADIUS
#define RT_SHADOW_FILTER_RADIUS 4
#endif
// In meters (the size of a kart is around 1)
#ifndef RT_SHADOW_FILTER_WORLD_RADIUS
#define RT_SHADOW_FILTER_WORLD_RADIUS 0.03
#endif
#define RT_SHADOW_PLANE_TOLERANCE_SCALE 0.01
#define RT_SHADOW_PLANE_TOLERANCE_BIAS 0.02

float sampleRTShadow(vec3 view_position, vec3 view_normal, int layer)
{
    const float scale = u_rt_shadow_scale;
    const ivec4 rect = rtShadowPixelRect(u_camera.m_viewport,
        textureSize(u_depth, 0));
    const ivec4 texel_rect = rtShadowTexelRect(rect, scale);

    // Position of the pixel in texels, texel i is at i
    const vec2 p = gl_FragCoord.xy * scale - 0.5;
    // Texels in a meter at the depth of the pixel
    const float texels_per_meter = scale * 0.5 * u_camera.m_viewport.w *
        abs(u_camera.m_projection_matrix[1][1]) / abs(view_position.z);
    const float radius = clamp(RT_SHADOW_FILTER_WORLD_RADIUS *
        texels_per_meter, 1.0, float(RT_SHADOW_FILTER_RADIUS));
    const int radius_i = int(ceil(radius));
    // The view space position of a pixel is on the line through the origin and
    // (inverse_projection * (ndc, 1, 1)).xyz, which is linear in the ndc, so
    // the line of every tap is 2 multiply-adds and its position is the point
    // of the line at the linear depth of the tap
    const mat4 inverse_projection = u_camera.m_inverse_projection_matrix;
    const vec2 ndc_scale = 2.0 / u_camera.m_viewport.zw;
    const vec3 line_x = inverse_projection[0].xyz * ndc_scale.x;
    const vec3 line_y = inverse_projection[1].xyz * ndc_scale.y;
    const vec3 line_origin = inverse_projection[2].xyz +
        inverse_projection[3].xyz - inverse_projection[0].xyz -
        inverse_projection[1].xyz -
        line_x * u_camera.m_viewport.x - line_y * u_camera.m_viewport.y;
    const ivec2 first = ivec2(floor(p)) - (radius_i - 1);
    const float tolerance = RT_SHADOW_PLANE_TOLERANCE_SCALE *
        abs(view_position.z) + RT_SHADOW_PLANE_TOLERANCE_BIAS;

    float sum_visibility = 0.0;
    float sum_weight = 0.0;
    // If every texel is too far, the closest one
    float closest_distance = 1.0e30;
    float closest_visibility = 1.0;
    for (int y = 0; y < radius_i * 2; y++)
    {
        for (int x = 0; x < radius_i * 2; x++)
        {
            const ivec2 texel = first + ivec2(x, y);
            if (any(lessThan(texel, texel_rect.xy)) ||
                any(greaterThanEqual(texel, texel_rect.zw)))
                continue;
            const vec2 tent = max(1.0 - abs(vec2(texel) - p) /
                radius, vec2(0.0));
            if (tent.x * tent.y <= 0.0)
                continue;
            const vec2 tap = texelFetch(u_rt_shadow_texture,
                ivec3(texel, layer), 0).xy;
            // The linear depth, 0 is the sky
            if (tap.y <= 0.0)
                continue;
            const float visibility = tap.x;
            const vec2 pixel_center = vec2(rtShadowTapPixel(texel, scale,
                rect)) + 0.5;
            const vec3 tap_line = line_origin + line_x * pixel_center.x +
                line_y * pixel_center.y;
            // Irrlicht's view space looks along +Z
            const vec3 tap_position = tap_line * (tap.y / tap_line.z);
            const float distance_to_plane =
                abs(dot(tap_position - view_position, view_normal));
            const float weight = tent.x * tent.y *
                clamp(1.0 - distance_to_plane / tolerance, 0.0, 1.0);
            sum_visibility += visibility * weight;
            sum_weight += weight;
            if (distance_to_plane < closest_distance)
            {
                closest_distance = distance_to_plane;
                closest_visibility = visibility;
            }
        }
    }
    return sum_weight > 1.0e-4 ? sum_visibility / sum_weight :
        closest_visibility;
}
