// The visibility of a layer of GEVulkanRTShadow at a position: the sun (layer 0)
// or a point light or a spot light (layer 1 + the index in
// u_global_light.m_lights, the same as the lighting pass uses). Used by
// rt_shadow.comp, which declares the push constants (m_types: bits of
// GEShadowType which are traced, m_lights: number of layers after the sun) and
// includes utils/constants_utils.glsl, utils/global_light_data.glsl and
// utils/rt_scene.glsl before it. A layer is the z of a thread.
//
// A pixel which a layer doesn't light (facing away, out of the range or the
// cone of a light) is 1.0 without a ray, the lighting pass has no light there
// anyway

// The layer of a thread (the sun or a light)
int rtShadowLayer()
{
    return int(gl_GlobalInvocationID.z);
}

// Whether a layer has anything to do, the other ones write nothing (the
// lighting pass doesn't read them): the sun if it's traced, a light if the
// lights are traced and it exists. light_count is the number of lights of the
// camera (u_global_light.m_light_count)
bool rtShadowLayerActive(int layer, int light_count)
{
    if (layer == 0)
        return (pc.m_types & int(GST_SUN)) != 0;
    return (pc.m_types & int(GST_POINTLIGHT)) != 0 && layer <= pc.m_lights &&
        layer <= light_count;
}

// Whether the light reaches the position (world space), same tests as the
// lighting pass (accumulateLights of utils/pbr_light.glsl)
bool rtShadowLightReaches(int light, vec3 world_position, vec3 world_normal,
                          out vec3 light_position)
{
    light_position = u_global_light.m_lights[light].m_position_radius.xyz;
    vec3 to_light = light_position - world_position;
    float distance_sq = dot(to_light, to_light);
    float invrange =
        u_global_light.m_lights[light].m_color_inverse_square_range.w;
    if (distance_sq * invrange > 1.0)
        return false;
    vec3 direction = to_light * inversesqrt(max(distance_sq, 1.0e-8));
    if (dot(world_normal, direction) <= 0.0)
        return false;
    // Spot light, the direction is in world space (x and y, z is from them)
    float sscale = u_global_light.m_lights[light].m_direction_scale_offset.z;
    if (sscale != 0.0)
    {
        vec3 sdir = vec3(
            u_global_light.m_lights[light].m_direction_scale_offset.xy, 0.0);
        sdir.z = sqrt(max(1.0 - dot(sdir.xy, sdir.xy), 0.0)) * sign(sscale);
        float attenuation = clamp(dot(-sdir, direction) * abs(sscale) +
            u_global_light.m_lights[light].m_direction_scale_offset.w, 0.0,
            1.0);
        if (attenuation == 0.0)
            return false;
    }
    return true;
}

// view_depth is the one of the position in view space. Layers which are not
// active are never asked
float rtShadowLayerVisibility(int layer, vec3 world_position,
                              vec3 world_normal, float view_depth)
{
    if (layer == 0)
    {
        vec3 sun_direction = normalize(u_global_light.m_sun_direction);
        // A surface facing away from the sun has no sun light anyway
        if (dot(world_normal, sun_direction) <= 0.0)
            return 1.0;
        return getSunShadow(world_position, world_normal, sun_direction,
            view_depth);
    }
    vec3 light_position;
    if (!rtShadowLightReaches(layer - 1, world_position, world_normal,
        light_position))
        return 1.0;
    return getLightShadow(world_position, world_normal, light_position,
        view_depth);
}
