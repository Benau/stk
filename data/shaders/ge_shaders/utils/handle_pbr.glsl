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
// Surfel confidence blends the environment fallback into GI without adding both.
vec3 handlePBRDeferred(vec3 diffuse_color, vec3 pbr, vec3 world_normal,
                       vec3 eyedir, vec3 normal, float perceptual_roughness,
                       vec4 world_position, float camera_distance,
                       float view_depth, float ao,
                       float rt_shadow, vec4 gi_irradiance, bool gi_lighting)
{
    // The GI is the ambient of the deferred pixels alone
    const bool gi = gi_lighting && u_surfel_gi;
    // How much of the pixel the surfels cover (0 when the GI is disabled), only
    // to keep the screen space occlusion off the ambient of the GI
    const float coverage = gi ? clamp(gi_irradiance.w, 0.0, 1.0) : 0.0;

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
        // The surfels already have the occlusion of the ambient light (their
        // rays see the sky which reaches them), so the screen space occlusion
        // is not applied to the GI: it would darken it twice, and the noise of
        // the occlusion would flicker it
        ao_diffuse = mix(ao_diffuse, vec3(1.0), coverage);
    }

    float radiance_level = perceptual_roughness * u_specular_levels_minus_one;
    vec3 reflection = reflect(-eyedir, normal);

    vec3 irradiance = vec3(0.0);
    vec3 radiance = vec3(0.0);
    if (u_ibl)
    {
        vec3 world_reflection = (u_camera.m_inverse_view_matrix *
            vec4(reflection, 0.0)).xyz;
        // The specular of the prefiltered environment is always the one of the
        // environment: the surfels are a diffuse GI only
        radiance = textureLod(u_specular, world_reflection,
            radiance_level).rgb;
        if (!gi || coverage < 1.0)
            irradiance = texture(u_diffuse, world_normal).rgb;
    }
    if (gi)
    {
        irradiance = mix(irradiance + u_global_light.m_ambient_color * 0.25,
            gi_irradiance.xyz, coverage);
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
        shadow = getShadowFactor(u_shadow_map, world_position.xyz,
            camera_distance, view_depth, NdotL, world_normal,
            u_global_light.m_sun_direction);
    }

    vec3 mixed_color = PBRSunAmbientEmitLight(
        normal, eyedir, lightdir, diffuse_color,
        irradiance, radiance,
        u_global_light.m_sun_color,
        // The GI replaces the flat ambient light too: adding it would double
        // count the ambient which the surfels trace
        gi ? vec3(0.0) : u_global_light.m_ambient_color,
        perceptual_roughness, pbr.y, pbr.z, shadow, ao_diffuse, ao_specular);

    if (!u_ibl && gi)
        mixed_color += irradiance * diffuse_color * ao_diffuse;

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
        eyedir, normal, perceptual_roughness, world_position, length(xpos),
        xpos.z, 1.0, 1.0, vec4(0.0), false);

    mixed_color += accumulateLights(u_global_light.m_light_count,
        diffuse_color, normal, xpos, eyedir, perceptual_roughness, pbr.y,
        world_position.xyz);

    mixed_color = handleFog(mixed_color, xpos);
    return convertColor(mixed_color);
}
