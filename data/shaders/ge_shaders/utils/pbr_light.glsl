// Same BRDF as bevy's point_light: the representative point of the sphere
// area light, the exact Smith-GGX visibility, the multiscatter compensation
// and the Burley diffuse. light_to_frag is the vector from the fragment to the
// light in view space, light_radius is the radius of the sphere area light
// (0 for the point lights of the engine).
vec3 PBRLight(
    vec3 normal,
    vec3 eyedir,
    vec3 light_to_frag,
    vec3 color,
    float perceptual_roughness,
    float metallic,
    float light_radius)
{
    float distance = length(light_to_frag);
    vec3 lightdir = light_to_frag * (1.0 / max(distance, 0.0001));
    float NdotV = max(dot(normal, eyedir), 0.0001);
    float NdotL = clamp(dot(normal, lightdir), 0.0, 1.0);

    float roughness = perceptualRoughnessToRoughness(perceptual_roughness);
    vec2 F_ab = F_AB(perceptual_roughness, NdotV);

    vec3 diffuse_color = color * (1.0 - metallic);
    vec3 F0 = mix(vec3(0.04), color, metallic);
    // No real world material has specular values under 0.02, so we use this range as a
    // "pre-baked specular occlusion" that extinguishes the fresnel term, for artistic control.
    // See: https://google.github.io/filament/Filament.html#specularocclusion
    float F90 = clamp(dot(F0, vec3(50.0 * 0.33)), 0.0, 1.0);

    // Representative point of the sphere area light, Karis 2013, the
    // compute_specular_layer_values_for_point_light of bevy
    vec3 reflection = reflect(-eyedir, normal);
    float LtFdotR = max(0.0001, dot(light_to_frag, reflection));
    vec3 center_to_ray = LtFdotR * reflection - light_to_frag;
    float center_to_ray_len_sq = max(dot(center_to_ray, center_to_ray), 1e-8);
    vec3 closest_point = light_to_frag + center_to_ray *
        clamp(light_radius * inversesqrt(center_to_ray_len_sq), 0.0, 1.0);
    float closest_point_len_sq = max(dot(closest_point, closest_point), 1e-8);
    vec3 L_spec = closest_point * inversesqrt(closest_point_len_sq);
    float a_prime = clamp(roughness + light_radius /
        (2.0 * max(distance, 0.0001)), 0.0, 1.0);

    float normalization_factor = roughness / max(a_prime, 0.0001);
    float specular_intensity = normalization_factor * normalization_factor;
    // Lerp between the base roughness and the area light one so the highlights
    // of smooth materials don't become too rough and dim (bevy's
    // specular_fix_remap)
    float fix = 1.0 - pow(1.0 - roughness, 4.0);
    float brdf_roughness = mix(roughness, a_prime, fix);

    // The diffuse uses the light direction, the specular the representative
    // point
    vec3 H = normalize(eyedir + lightdir);
    float NdotH = clamp(dot(normal, H), 0.0, 1.0);
    float LdotH = clamp(dot(lightdir, H), 0.0, 1.0);
    vec3 diffuse = diffuse_color * Fd_Burley(roughness, NdotV, NdotL, LdotH);

    vec3 H_spec = normalize(eyedir + L_spec);
    float NdotL_spec = clamp(dot(normal, L_spec), 0.0, 1.0);
    float NdotH_spec = clamp(dot(normal, H_spec), 0.0, 1.0);
    float LdotH_spec = clamp(dot(L_spec, H_spec), 0.0, 1.0);

    float D = D_GGX(brdf_roughness, NdotH_spec);
    float V = V_Smith_GGX_Correlated(brdf_roughness, NdotV, NdotL_spec);
    vec3 F = fresnel(F0, F90, LdotH_spec);
    // Multiscatter energy compensation (bevy's specular_multiscatter)
    vec3 specular = specular_intensity * D * V * F *
        (1.0 + F0 * (1.0 / (F_ab.x + F_ab.y) - 1.0));

    // Sphere area light visibility (solid-angle attenuation)
    if (light_radius > 0.0)
    {
        float solid_angle = light_radius * light_radius /
            max(distance * distance, 1e-4);
        specular *= clamp(NdotL_spec /
            max(NdotL_spec + solid_angle, 1e-4), 0.0, 1.0);
    }

    return diffuse * NdotL + specular * NdotL_spec;
}

vec3 PBRSunAmbientEmitLight(
    vec3 normal,
    vec3 eyedir,
    vec3 sundir,
    vec3 color,
    vec3 irradiance,
    vec3 radiance,
    vec3 sun_color,
    vec3 ambient_color,
    float perceptual_roughness,
    float metallic,
    float emissive,
    float shadow,
    vec3 ao_diffuse,
    vec3 ao_specular)
{
    // Copied from PBRLight to use F_ab and F90 again
    float NdotV = max(dot(normal, eyedir), 0.0001);
    float NdotL = clamp(dot(normal, sundir), 0.0, 1.0);

    vec2 F_ab = F_AB(perceptual_roughness, NdotV);

    vec3 H = normalize(eyedir + sundir);
    float NdotH = clamp(dot(normal, H), 0.0, 1.0);
    float LdotH = clamp(dot(sundir, H), 0.0, 1.0);

    vec3 diffuse_color = color * (1.0 - metallic);
    vec3 F0 = mix(vec3(0.04), color, metallic);
    // No real world material has specular values under 0.02, so we use this range as a
    // "pre-baked specular occlusion" that extinguishes the fresnel term, for artistic control.
    // See: https://google.github.io/filament/Filament.html#specularocclusion
    float F90 = clamp(dot(F0, vec3(50.0 * 0.33)), 0.0, 1.0);

    float roughness = perceptualRoughnessToRoughness(perceptual_roughness);

    vec3 diffuse = diffuse_color * Fd_Burley(roughness, NdotV, NdotL, LdotH);

    float D = D_GGX(roughness, NdotH);
    float V = V_Smith_GGX_Correlated(roughness, NdotV, NdotL);
    vec3 F = fresnel(F0, F90, LdotH);
    vec3 specular = D * V * F *
        (1.0 + F0 * (1.0 / (F_ab.x + F_ab.y) - 1.0));

    vec3 sunlight = NdotL * (diffuse + specular) * shadow;

    vec3 diffuse_ambient = envBRDFApprox(diffuse_color, F_AB(1.0, NdotV));

    vec3 specular_ambient = F90 * envBRDFApprox(F0, F_ab);

    // Other 0.6 comes from skybox
    ambient_color *= 0.4;
    vec3 environment;
    if (u_ibl)
    {
        // The multiscatter of the dielectric and of the metallic parts are
        // compensated separately, see environmentLight
        environment = environmentLight(irradiance, radiance, diffuse_color,
            vec3(0.04), color, metallic, F_ab, ao_diffuse, ao_specular);
    }
    else
    {
        environment = u_global_light.m_skytop_color * ambient_color *
            diffuse_color * ao_diffuse;
    }

    vec3 emit = emissive * color * 4.0;

    return sun_color * sunlight
          + environment + emit
          + (diffuse_ambient * ao_diffuse + specular_ambient * ao_specular) *
          ambient_color;
}

float getShadowPCF(sampler2DArrayShadow map, vec2 shadowtexcoord, int layer, float depth)
{
    // Castaño, 2013, "Shadow Mapping Summary Part 1"
    float shadow_size = float(u_shadow_size);
    vec2 uv = shadowtexcoord * shadow_size + 0.5;
    vec2 base = (floor(uv) - 0.5) / shadow_size;
    vec2 st = fract(uv);

    vec2 uw = vec2(3.0 - 2.0 * st.x, 1.0 + 2.0 * st.x);
    vec2 vw = vec2(3.0 - 2.0 * st.y, 1.0 + 2.0 * st.y);

    vec2 u = vec2((2.0 - st.x) / uw.x - 1.0, st.x / uw.y + 1.0) / shadow_size;
    vec2 v = vec2((2.0 - st.y) / vw.x - 1.0, st.y / vw.y + 1.0) / shadow_size;

    float sum = 0.0;
    sum += uw.x * vw.x * texture(map, vec4(base + vec2(u.x, v.x), float(layer), depth));
    sum += uw.y * vw.x * texture(map, vec4(base + vec2(u.y, v.x), float(layer), depth));
    sum += uw.x * vw.y * texture(map, vec4(base + vec2(u.x, v.y), float(layer), depth));
    sum += uw.y * vw.y * texture(map, vec4(base + vec2(u.y, v.y), float(layer), depth));
    return sum * (1.0 / 16.0);
}

// ---------------------------------------------------------------------------
// sampleOmniShadow
//
// Point lights  :  the 6 faces of a cube map, the same as bevy's
//                  fetch_point_shadow: every face is a 90 degree perspective
//                  map and the face is picked by the largest absolute
//                  component of the direction from the light to the fragment.
//                  The camera matrix of a slot is the projection * view of its
//                  face (GEVulkanOmniShadowFBO::buildLightMatrices).
//
// Spot lights   :  a single perspective map of the cone, the camera matrix of
//                  the slot is the projection * view of the light.
//
// Every map is a tile of the omni shadow atlas, a single 2D depth map of 2 *
// u_shadow_size pixels: getOmniShadowSize is the size of a tile in pixels and
// getOmniShadowAtlasRect the position of the tile of a face from its slot.
// The PCF taps are clamped to the tile so they don't read the maps next to
// it.
// ---------------------------------------------------------------------------
// Same normal bias of bevy's fetch_point_shadow: in world units per world unit
// of distance from the light, scaled by the texel size at a distance of one
// (2 / face size for the 90 degree faces of the cube map), and the depth bias
// in world units along the ray to the light
const float POINT_SHADOW_NORMAL_BIAS = 2.0;
const float POINT_SHADOW_DEPTH_BIAS = 0.08;
// The spot light uses the same biases, with the texel size at a distance of
// one of its perspective map instead of a face of the cube map
const float SPOT_SHADOW_NORMAL_BIAS = 2.0;
const float SPOT_SHADOW_DEPTH_BIAS = 0.08;

// The 2 * shadow_size atlas is a square grid of the smallest number of rows
// which fits all the slots, the same choice as GEVulkanOmniShadowFBO
uint getOmniShadowTilesPerRow()
{
    uint slots = uint(OMNI_FACES_PER_LIGHT) * u_point_shadow_limit;
    uint tiles_per_row = 1u;
    while (tiles_per_row * tiles_per_row < slots)
        tiles_per_row++;
    return tiles_per_row;
}

// The size in pixels of a tile of the atlas
uint getOmniShadowSize()
{
    return (u_shadow_size * 2u) / getOmniShadowTilesPerRow();
}

// The tile in the atlas of a face of the cube map of a light (0 is the only
// face of a spot light), the same slot layout as GEVulkanOmniShadowFBO
vec4 getOmniShadowAtlasRect(int light_id, int face)
{
    uint atlas_size = u_shadow_size * 2u;
    uint size = getOmniShadowSize();
    uint tiles_per_row = getOmniShadowTilesPerRow();
    uint slot = uint(light_id) * uint(OMNI_FACES_PER_LIGHT) + uint(face);
    float inv_atlas_size = 1.0 / float(atlas_size);
    return vec4(float((slot % tiles_per_row) * size) * inv_atlas_size,
        float((slot / tiles_per_row) * size) * inv_atlas_size,
        float(size) * inv_atlas_size, float(size) * inv_atlas_size);
}

// The face, the uv (in the map of the face) and the depth of the shadow of a
// position, false if the position is outside of the map (it's lit then)
bool getOmniShadowCoord(int light_id, vec3 world_pos, vec3 world_normal,
                        out int face, out vec2 uv, out float depth)
{
    vec3 light_pos = u_global_light.m_lights[light_id].m_position_radius.xyz;
    float radius = max(u_global_light.m_lights[light_id]
        .m_position_radius.w, 0.0001);
    vec3 surface_to_light = light_pos - world_pos;
    float distance = length(surface_to_light);
    vec3 to_light = surface_to_light * (1.0 / max(distance, 0.0001));
    face = 0;
    float texel_size = 1.0 / float(getOmniShadowSize());

    float sscale = u_global_light.m_lights[light_id].m_direction_scale_offset.z;
    if (sscale != 0.0)
    {
        // Spot light: the world size of a texel at the distance of the
        // fragment is 2 * tan(half fov) * distance / size. cos_outer is
        // encoded in the scale and the offset of the cone attenuation, the
        // map pads the cone by 2 degrees (see buildLightMatrices)
        float cos_outer = clamp(u_global_light.m_lights[light_id]
            .m_direction_scale_offset.w / -abs(sscale), -1.0, 1.0);
        float tan_outer = sqrt(max(1.0 - cos_outer * cos_outer, 0.0)) /
            max(cos_outer, 0.0001);
        float tan_padded = tan_outer + 0.035 * (1.0 + tan_outer * tan_outer);
        vec3 offset_position = world_pos + world_normal *
            (SPOT_SHADOW_NORMAL_BIAS * 2.0 * texel_size * tan_padded *
            distance) + to_light * SPOT_SHADOW_DEPTH_BIAS;

        vec4 clip = u_global_light.m_lights[light_id]
            .m_shadow_projection_view_matrix[0] * vec4(offset_position, 1.0);
        if (clip.w <= 0.0)
            return false;
        vec3 ndc = clip.xyz / clip.w;
        uv = ndc.xy * 0.5 + 0.5;
        depth = ndc.z;
        return uv.x >= 0.0 && uv.x <= 1.0 &&
            uv.y >= 0.0 && uv.y <= 1.0 &&
            depth >= 0.0 && depth <= 1.0;
    }

    // Point light, bevy's fetch_point_shadow: the normal bias grows with the
    // world size of a texel at the distance of the fragment (2 * distance /
    // face size for a 90 degree face), the depth bias is along the ray to the
    // light. The texel size is taken at the largest absolute axis like bevy
    vec3 abs_surface_to_light = abs(surface_to_light);
    float distance_to_light = max(abs_surface_to_light.x,
        max(abs_surface_to_light.y, abs_surface_to_light.z));
    vec3 offset_position = world_pos + world_normal *
        (POINT_SHADOW_NORMAL_BIAS * 2.0 * texel_size * sqrt(2.0) *
        distance_to_light) + to_light * POINT_SHADOW_DEPTH_BIAS;

    // The face of the cube map: the largest absolute component of the
    // direction from the light to the biased position, in the +X, -X, +Y, -Y,
    // +Z, -Z order of GEVulkanOmniShadowFBO
    vec3 direction = offset_position - light_pos;
    direction *= 1.0 / max(length(direction), 0.0001);
    vec3 abs_direction = abs(direction);
    if (abs_direction.x >= abs_direction.y &&
        abs_direction.x >= abs_direction.z)
        face = direction.x >= 0.0 ? 0 : 1;
    else if (abs_direction.y >= abs_direction.z)
        face = direction.y >= 0.0 ? 2 : 3;
    else
        face = direction.z >= 0.0 ? 4 : 5;

    vec4 clip = u_global_light.m_lights[light_id]
        .m_shadow_projection_view_matrix[face] * vec4(offset_position, 1.0);
    if (clip.w <= 0.0)
        return false;
    vec3 ndc = clip.xyz / clip.w;
    // A fragment at the border of two faces can project a bit outside of the
    // face it picked after the bias, clamp it instead of fully lighting it
    uv = clamp(ndc.xy * 0.5 + 0.5, 0.0, 1.0);
    depth = ndc.z;
    return depth >= 0.0 && depth <= 1.0;
}

// The uv of the atlas of a uv in the map of a light, clamped to the tile
vec2 getOmniShadowAtlasUV(vec4 atlas_rect, vec2 uv)
{
    float half_texel = 1.0 / (2.0 * float(getOmniShadowSize()));
    vec2 min_uv = atlas_rect.xy + atlas_rect.zw * half_texel;
    vec2 max_uv = atlas_rect.xy + atlas_rect.zw * (1.0 - half_texel);
    return clamp(atlas_rect.xy + uv * atlas_rect.zw, min_uv, max_uv);
}

// Castaño 2013 PCF like getShadowPCF, on the tile of the face of the light
float getOmniShadowPCF(int light_id, int face, vec2 uv, float depth)
{
    vec4 atlas_rect = getOmniShadowAtlasRect(light_id, face);
    float size = float(getOmniShadowSize());
    vec2 uv_size = uv * size;
    vec2 base = (floor(uv_size) - 0.5) / size;
    vec2 st = fract(uv_size);

    vec2 uw = vec2(3.0 - 2.0 * st.x, 1.0 + 2.0 * st.x);
    vec2 vw = vec2(3.0 - 2.0 * st.y, 1.0 + 2.0 * st.y);

    vec2 u = vec2((2.0 - st.x) / uw.x - 1.0, st.x / uw.y + 1.0) / size;
    vec2 v = vec2((2.0 - st.y) / vw.x - 1.0, st.y / vw.y + 1.0) / size;

    float sum = 0.0;
    vec2 tap = getOmniShadowAtlasUV(atlas_rect, base + vec2(u.x, v.x));
    sum += uw.x * vw.x * texture(u_omni_shadow_map, vec3(tap, depth));
    tap = getOmniShadowAtlasUV(atlas_rect, base + vec2(u.y, v.x));
    sum += uw.y * vw.x * texture(u_omni_shadow_map, vec3(tap, depth));
    tap = getOmniShadowAtlasUV(atlas_rect, base + vec2(u.x, v.y));
    sum += uw.x * vw.y * texture(u_omni_shadow_map, vec3(tap, depth));
    tap = getOmniShadowAtlasUV(atlas_rect, base + vec2(u.y, v.y));
    sum += uw.y * vw.y * texture(u_omni_shadow_map, vec3(tap, depth));
    return sum * (1.0 / 16.0);
}

float sampleOmniShadow(int light_id, vec3 world_pos, vec3 world_normal)
{
    int face;
    vec2 uv;
    float depth;
    if (!getOmniShadowCoord(light_id, world_pos, world_normal,
        face, uv, depth))
        return 1.0;

    return getOmniShadowPCF(light_id, face, uv, depth);
}

// The same shadow with a single tap instead of the 4 of the PCF, for the
// volumetric fog of light_scatter.comp which samples it at every step of the
// march (the fog has no surface which needs the PCF)
float sampleOmniShadowSingleSample(int light_id, vec3 world_pos,
                                   vec3 world_normal)
{
    int face;
    vec2 uv;
    float depth;
    if (!getOmniShadowCoord(light_id, world_pos, world_normal,
        face, uv, depth))
        return 1.0;

    vec4 atlas_rect = getOmniShadowAtlasRect(light_id, face);
    return texture(u_omni_shadow_map,
        vec3(getOmniShadowAtlasUV(atlas_rect, uv), depth));
}

#ifdef HAS_RT_SHADOW_TEXTURE
#include "rt_shadow_grid.glsl"
#include "rt_shadow_filter.glsl"
// Visibility of a point light or a spot light with the ray traced shadow
// (GEVulkanRTShadow), layer 1 + the index of the light, filtered to the pixel
// (xpos and normal are in view space). 1.0 if the light has none (the lights
// are not ray traced, or it's not one of the first u_point_shadow_limit)
float getLightRTShadow(int light, vec3 xpos, vec3 normal)
{
    if ((u_rt_shadow_type & GST_POINTLIGHT) != 0 &&
        light < int(u_point_shadow_limit) &&
        light + 1 < textureSize(u_rt_shadow_texture, 0).z)
    {
        return sampleRTShadow(xpos, normal, light + 1);
    }
    return 1.0;
}
#else
float getLightRTShadow(int light, vec3 xpos, vec3 normal)
{
    return 1.0;
}
#endif

vec3 accumulateLights(int light_count, vec3 diffuse_color, vec3 normal,
                      vec3 xpos, vec3 eyedir, float perceptual_roughness,
                      float metallic, vec3 world_position, vec3 world_normal)
{
    vec3 accumulated_color = vec3(0.0);
    for (int i = 0; i < light_count; i++)
    {
        vec3 light_to_frag = (u_camera.m_view_matrix *
            vec4(u_global_light.m_lights[i].m_position_radius.xyz,
            1.0)).xyz - xpos;
        float invrange = u_global_light.m_lights[i].m_color_inverse_square_range.w;
        float distance_sq = dot(light_to_frag, light_to_frag);
        if (distance_sq * invrange > 1.)
            continue;
        // SpotLight
        float sattenuation = 1.;
        float sscale = u_global_light.m_lights[i].m_direction_scale_offset.z;
        float distance = sqrt(distance_sq);
        float distance_inverse = 1. / distance;
        vec3 L = light_to_frag * distance_inverse;
        if (sscale != 0.)
        {
            vec3 sdir =
                vec3(u_global_light.m_lights[i].m_direction_scale_offset.xy, 0.);
            sdir.z = sqrt(1. - dot(sdir, sdir)) * sign(sscale);
            sdir = (u_camera.m_view_matrix * vec4(sdir, 0.0)).xyz;
            sattenuation = clamp(dot(-sdir, L) *
                abs(sscale) +
                u_global_light.m_lights[i].m_direction_scale_offset.w, 0.0, 1.0);
#ifndef TILED_GPU
            // Reduce branching in tiled GPU
            if (sattenuation == 0.)
                continue;
#endif
        }
        vec3 diffuse_specular = PBRLight(normal, eyedir, L, diffuse_color,
            perceptual_roughness, metallic, 0.0);
        if ((u_shadow_type & GST_POINTLIGHT) != 0 &&
            u_shadow_size > 0 && i < u_point_shadow_limit)
        {
            float shadow = sampleOmniShadow(i, world_position, world_normal);
            diffuse_specular *= shadow;
        }
        else
            diffuse_specular *= getLightRTShadow(i, xpos, normal);
        float attenuation = 20. / (1. + distance_sq);
        float radius = u_global_light.m_lights[i].m_position_radius.w;
        attenuation *= (radius - distance) / radius;
        attenuation *= sattenuation * sattenuation;
        vec3 light_color =
            u_global_light.m_lights[i].m_color_inverse_square_range.xyz;
        accumulated_color += light_color * attenuation * diffuse_specular;
    }
    return accumulated_color;
}

// Copied because reusing in a loop will be slower
vec3 calculateLight(int i, vec3 diffuse_color, vec3 normal, vec3 xpos,
                    vec3 eyedir, float perceptual_roughness, float metallic,
                    vec3 world_position, vec3 world_normal)
{
    vec3 light_to_frag = (u_camera.m_view_matrix *
        vec4(u_global_light.m_lights[i].m_position_radius.xyz,
        1.0)).xyz - xpos;
    float invrange = u_global_light.m_lights[i].m_color_inverse_square_range.w;
    float distance_sq = dot(light_to_frag, light_to_frag);
    if (distance_sq * invrange > 1.)
        return vec3(0.0);
    // SpotLight
    float sattenuation = 1.;
    float sscale = u_global_light.m_lights[i].m_direction_scale_offset.z;
    float distance = sqrt(distance_sq);
    float distance_inverse = 1. / distance;
    vec3 L = light_to_frag * distance_inverse;
    if (sscale != 0.)
    {
        vec3 sdir =
            vec3(u_global_light.m_lights[i].m_direction_scale_offset.xy, 0.);
        sdir.z = sqrt(1. - dot(sdir, sdir)) * sign(sscale);
        sdir = (u_camera.m_view_matrix * vec4(sdir, 0.0)).xyz;
        sattenuation = clamp(dot(-sdir, L) *
            abs(sscale) +
            u_global_light.m_lights[i].m_direction_scale_offset.w, 0.0, 1.0);
        if (sattenuation == 0.)
            return vec3(0.0);
    }
    vec3 diffuse_specular = PBRLight(normal, eyedir, L, diffuse_color,
        perceptual_roughness, metallic, 0.0);
    if ((u_shadow_type & GST_POINTLIGHT) != 0 &&
        u_shadow_size > 0 && i < u_point_shadow_limit)
    {
        float shadow = sampleOmniShadow(i, world_position, world_normal);
        diffuse_specular *= shadow;
    }
    else
        diffuse_specular *= getLightRTShadow(i, xpos, normal);
    float attenuation = 20. / (1. + distance_sq);
    float radius = u_global_light.m_lights[i].m_position_radius.w;
    attenuation *= (radius - distance) / radius;
    attenuation *= sattenuation * sattenuation;
    vec3 light_color =
        u_global_light.m_lights[i].m_color_inverse_square_range.xyz;
    return light_color * attenuation * diffuse_specular;
}

// The sun cascades are split by view depth near and by distance far, see
// GEVulkanShadowFBO::prepare: a cascade starts at the depth where the far
// sphere of the previous one still covers the whole screen and ends at its
// far distance. The last cascade has no far distance of its own, its body
// ends at the bounding box of the scene. camera_distance is the distance of
// the position from the camera, view_depth its depth along the view direction
float getShadowFactor(sampler2DArrayShadow map, vec3 world_position,
                      float camera_distance, float view_depth, float NdotL,
                      vec3 normal, vec3 lightdir)
{
    if (NdotL <= 0.001)
        return 1.0;

    vec2 base_normal_bias = (u_global_light.m_shadow_view_matrix * vec4(normal, 0.)).xy;
    base_normal_bias *= (1.0 - max(0.0, dot(-normal, lightdir))) / float(u_shadow_size);

    // Near depth of each cascade and the cosine of the largest angle between
    // the view direction and a frustum ray
    vec4 split_near = u_global_light.m_shadow_split;
    float min_ray_cos = u_global_light.m_shadow_split_min_ray_cos.x;

    int level = view_depth >= split_near.y ?
        (view_depth >= split_near.z ?
            (view_depth >= split_near.w ? 3 : 2) : 1) : 0;

    vec4 light_view_position = u_global_light.m_shadow_projection_view_matrix[level] * vec4(world_position, 1.0);
    light_view_position.xyz /= light_view_position.w;
    light_view_position.xy = light_view_position.xy * 0.5 + 0.5 + base_normal_bias;

    float shadow = getShadowPCF(map, light_view_position.xy, level, light_view_position.z);

    if (level > 0)
    {
        // The previous cascade covers this pixel until the distance along
        // its ray reaches its far distance, at the view depth
        // near_depth / min_ray_cos * view_depth / camera_distance. Fade it
        // out over that overlap: at the corners of the screen it ends right
        // at the near depth, at the center it's the whole far radius of the
        // previous cascade
        float near_depth = level == 1 ? split_near.y :
            (level == 2 ? split_near.z : split_near.w);
        float prev_far_depth = near_depth / min_ray_cos *
            view_depth / camera_distance;
        float blend = clamp((view_depth - near_depth) /
            max(prev_far_depth - near_depth, 0.0001), 0.0, 1.0);

        vec4 prev_light_view_position = u_global_light.m_shadow_projection_view_matrix[level - 1] * vec4(world_position, 1.0);
        prev_light_view_position.xyz /= prev_light_view_position.w;
        prev_light_view_position.xy = prev_light_view_position.xy * 0.5 + 0.5 + base_normal_bias;

        shadow = mix(getShadowPCF(map, prev_light_view_position.xy, level - 1, prev_light_view_position.z), shadow, blend);
    }

    return shadow;
}

// The same cascades and normal bias as getShadowFactor, but a single sample of
// the map (no PCF) and no blend of the cascades: the GI injects the sun into
// every ray which it traces, so it shares the shadows of the lighting pass
// with one tap instead of the 16 of a pixel. A position outside of the
// cascades is lit (the map is clamped to its edge, which is not its shadow)
float getShadowFactorSingleSample(sampler2DArrayShadow map,
                                  vec3 world_position, float camera_distance,
                                  float view_depth, float NdotL, vec3 normal,
                                  vec3 lightdir)
{
    if (NdotL <= 0.001)
        return 1.0;

    vec2 base_normal_bias = (u_global_light.m_shadow_view_matrix * vec4(normal, 0.)).xy;
    base_normal_bias *= (1.0 - max(0.0, dot(-normal, lightdir))) / float(u_shadow_size);

    vec4 split_near = u_global_light.m_shadow_split;
    int level = view_depth >= split_near.y ?
        (view_depth >= split_near.z ?
            (view_depth >= split_near.w ? 3 : 2) : 1) : 0;

    vec4 light_view_position = u_global_light.m_shadow_projection_view_matrix[level] * vec4(world_position, 1.0);
    light_view_position.xyz /= light_view_position.w;
    light_view_position.xy = light_view_position.xy * 0.5 + 0.5 + base_normal_bias;
    if (light_view_position.x < 0.0 || light_view_position.x > 1.0 ||
        light_view_position.y < 0.0 || light_view_position.y > 1.0)
        return 1.0;

    return texture(map, vec4(light_view_position.xy, float(level),
        light_view_position.z));
}
