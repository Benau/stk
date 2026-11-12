layout (set = 2, binding = 0) uniform samplerCube u_diffuse;
layout (set = 2, binding = 1) uniform samplerCube u_specular;
layout (set = 2, binding = 3) uniform sampler2DArrayShadow u_shadow_map;

#include "camera.glsl"
#include "constants_utils.glsl"
#include "spm_data.glsl"
#include "pbr_utils.glsl"
#include "global_light_data.glsl"

#include "pbr_light.glsl"
#include "sun_direction.glsl"

vec3 gtaoMultiBounce(float visibility, vec3 albedo)
{
    // Jimenez et al. 2016, "Practical Realtime Strategies for Accurate Indirect Occlusion"
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;

    return max(vec3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}

// ao is the visibility of the ambient light (screen space ambient occlusion),
// 1.0 if there is none. Same as IBL.frag of the OpenGL renderer.
// rt_shadow is the visibility of the sun of GEVulkanRTShadow, it's only used if
// the sun is ray traced (u_rt_shadow_type)
// gi is the irradiance of GEVulkanSurfelGI and its coverage (alpha), it's only
// used with u_surfel_gi
// solid_reflection is the specular radiance of GEVulkanSolidReflection (the
// traced color mixed with the specular IBL), it replaces the specular IBL if
// has_solid_reflection. With u_stochastic_reflection it's the traced radiance
// premultiplied by its coverage (alpha), the rest is the specular IBL
// (alpha is unused otherwise)
vec3 handlePBRDeferred(vec3 diffuse_color, vec3 pbr, vec3 world_normal,
                       vec3 eyedir, vec3 normal, float perceptual_roughness,
                       vec4 world_position, float view_depth, float ao,
                       float rt_shadow, vec4 gi, bool has_solid_reflection,
                       vec4 solid_reflection)
{
    vec3 ao_diffuse = vec3(1.0);
    vec3 ao_specular = vec3(1.0);
    if (u_ssao && ao < 1.0)
    {
        float NdotV = max(dot(normal, eyedir), 0.0);
        // Lagarde and de Rousiers 2014, "Moving Frostbite to PBR"
        float ao_spec = clamp(pow(NdotV + ao,
            exp2(-16.0 * perceptual_roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
        ao_diffuse = gtaoMultiBounce(ao, diffuse_color);
        ao_specular = gtaoMultiBounce(ao_spec, diffuse_color);
    }

    float radiance_level = perceptual_roughness * u_specular_levels_minus_one;
    vec3 reflection = reflect(-eyedir, normal);

    vec3 irradiance = getAmbientIrradiance();
    vec3 radiance = vec3(0.0);
    if (u_ibl)
    {
        vec3 world_reflection = (u_camera.m_inverse_view_matrix *
            vec4(reflection, 0.0)).xyz;
        irradiance = texture(u_diffuse, world_normal).rgb;
        radiance = textureLod(u_specular, world_reflection, radiance_level).rgb;
    }
    // The traced reflection replaces the specular IBL (it's the IBL where
    // nothing is traced)
    if (has_solid_reflection)
    {
        if (u_stochastic_reflection)
        {
            radiance = solid_reflection.rgb + radiance *
                (1.0 - solid_reflection.a);
        }
        else
            radiance = solid_reflection.rgb;
    }
    // With or without image based lighting
    float diffuse_ambient_scale = 1.0;
    if (u_surfel_gi)
    {
        irradiance = mix(irradiance, gi.rgb, gi.a);
        // With or without image based lighting the ambient color is lit by
        // PBRSunAmbientEmitLight too (the rest of the sky), the surfels have
        // all of it where they cover the pixel (the rays of a surfel which
        // see the sky get it, the ones which hit a wall see it lit by it)
        diffuse_ambient_scale = 1.0 - gi.a;
    }

    vec3 lightdir = sunDirection(reflection,
        u_global_light.m_sun_direction, u_global_light.m_sun_angle_tan_half,
        u_camera.m_inverse_view_matrix);

    float shadow = 1.0;
    if ((u_rt_shadow_type & GST_SUN) != 0)
        shadow = rt_shadow;
    else if ((u_shadow_type & GST_SUN) != 0 && u_shadow_size != 0)
    {
        float NdotL = clamp(dot(normal, lightdir), 0.0, 1.0);
        shadow = getShadowFactor(u_shadow_map, world_position.xyz, view_depth, NdotL, world_normal, u_global_light.m_sun_direction);
    }

    vec3 mixed_color = PBRSunAmbientEmitLight(
        normal, eyedir, lightdir, diffuse_color,
        irradiance, radiance,
        u_global_light.m_sun_color, u_global_light.m_ambient_color,
        diffuse_ambient_scale,
        perceptual_roughness, pbr.y, pbr.z, shadow, ao_diffuse, ao_specular);

    // Without image based lighting PBRSunAmbientEmitLight has no specular
    // ambient light, so the reflection is added with the same scale as the
    // specular IBL would have (the Fresnel of environmentLight)
    if (has_solid_reflection && !u_ibl)
    {
        mixed_color += reflectionSpecularAlbedo(diffuse_color, pbr.y,
            perceptual_roughness, max(dot(normal, eyedir), 0.0001)) *
            solid_reflection.rgb * ao_specular;
    }
    return mixed_color;
}

// Exponential fog: the fog is added to the lit color (not mixed with it),
// before tonemap
vec3 handleFog(vec3 hdr, vec3 xpos)
{
    // xpos is the view space position, density is 0 if there is no fog
    float fog_density = u_global_light.m_fog_density;
    if (fog_density > 0.0)
    {
        float fog_factor = 1.0 - exp(-fog_density * length(xpos));
        hdr += u_global_light.m_fog_color.rgb * fog_factor;
    }
    return hdr;
}

vec3 handlePBR(vec3 diffuse_color, vec3 pbr, vec4 world_position,
               vec3 world_normal)
{
    vec3 xpos = (u_camera.m_view_matrix * world_position).xyz;
    vec3 eyedir = -normalize(xpos);
    vec3 normal = (u_camera.m_view_matrix * vec4(world_normal, 0.0)).xyz;
    float perceptual_roughness = 1.0 - pbr.x;

    vec3 mixed_color = handlePBRDeferred(diffuse_color, pbr, world_normal,
        eyedir, normal, perceptual_roughness, world_position, xpos.z, 1.0, 1.0,
        vec4(0.0), false, vec4(0.0));

    mixed_color += accumulateLights(u_global_light.m_light_count,
        diffuse_color, normal, xpos, eyedir, perceptual_roughness, pbr.y,
        world_position.xyz);

    mixed_color = handleFog(mixed_color, xpos);
    return convertColor(mixed_color);
}
