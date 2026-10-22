// The mask of the displace materials (the shift of the distortion, like water)
// and, if its reflection is on (u_displace_reflection), the reflection color
// which displace_transparent.frag blends over them: screen space (SSR) or
// ray traced.
//
// displace_mask_rt.frag is this shader with DISPLACE_RT defined, the reflected
// ray is traced in the ray traced scene (GEVulkanRTScene) with the shading of
// the hits shared with the reflection of the solid materials (see
// utils/rt_reflection.glsl) instead of the screen, which has no history and no
// screen edges. It uses the pipeline layout of solid_rt.frag (GEVulkanDrawCall::
// m_solid_rt_layout: the scene is set 0, the environment is set 3 and the mesh
// textures, also the one of the mask of this material, are the bindless set 4),
// so displace_mask_rt.frag defines DISPLACE_ENV_SET and MESH_TEXTURES_SET. The
// environment is the fallback of both
layout(location = 1) in vec2 f_uv;
layout(location = 5) in vec3 f_normal;
layout(location = 8) in vec4 f_world_position;

layout(location = 3) flat in int f_material_id;

layout(location = 0) out vec2 o_displace_mask;
layout(location = 1) out vec4 o_displace_ssr;

layout(push_constant) uniform Constants
{
    vec4 m_displace_direction;
} u_push_constants;

// The set of the environment (skybox, and the cubemaps of the hits) in the
// pipeline layout, the mesh textures are MESH_TEXTURES_SET of
// utils/sample_mesh_texture.glsl
#ifndef DISPLACE_ENV_SET
#define DISPLACE_ENV_SET 2
#endif
#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/sample_mesh_texture.glsl"
#include "../utils/displace_utils.frag"

layout (set = DISPLACE_ENV_SET, binding = 2) uniform samplerCube u_skybox_texture;
#ifdef DISPLACE_RT
// The environment for the shading of a hit (the same set as the skybox), and
// u_shadow_map is only to share pbr_light.glsl
layout (set = DISPLACE_ENV_SET, binding = 0) uniform samplerCube u_diffuse;
layout (set = DISPLACE_ENV_SET, binding = 1) uniform samplerCube u_specular;
layout (set = DISPLACE_ENV_SET, binding = 3) uniform sampler2DArrayShadow
    u_shadow_map;
#include "utils/pbr_utils.glsl"
#include "utils/get_vertex_color.glsl"
#include "../utils/rgb_conversion.frag"
#include "utils/global_light_data.glsl"
#include "utils/pbr_light.glsl"
#include "utils/rt_utils.glsl"
#include "utils/rt_scene.glsl"
#include "utils/rt_reflection.glsl"
#else
layout (set = 3, binding = 0) uniform sampler2D u_displace_color;
layout (set = 3, binding = 1) uniform sampler2DShadow u_depth;
layout (set = 3, binding = 2) uniform sampler2D u_hiz_depth;

// Iteration limit of the hiz trace (0 uses the depth buffer)
const uint u_hiz_iterations =
    u_displace_reflection == DISPLACE_REFLECTION_SSR_HIZ400 ? 400u :
    u_displace_reflection == DISPLACE_REFLECTION_SSR_HIZ200 ? 200u :
    u_displace_reflection == DISPLACE_REFLECTION_SSR_HIZ100 ? 100u : 0u;
#ifdef PBR_ENABLED
#define SSR_HIZ
#endif
#include "../utils/ssr.glsl"
#endif

void main()
{
#ifdef PBR_ENABLED
    float horiz = sampleMeshTexture2(f_material_id, f_uv + u_push_constants.m_displace_direction.xy * 150.).x;
    float vert = sampleMeshTexture2(f_material_id, (f_uv.yx + u_push_constants.m_displace_direction.zw * 150.) * vec2(0.9)).x;
    vec2 mask = getDisplaceShift(horiz, vert);
    mask = (mask + 1.0) * 0.5;
    o_displace_mask = mask;
    if (u_displace_reflection != DISPLACE_REFLECTION_DISABLED)
    {
        float alpha = sampleMeshTexture0(f_material_id, f_uv).a;
        if (alpha == 0.0)
        {
            o_displace_ssr = vec4(0.0);
            return;
        }
        // eye-space position
        vec3 xpos = (u_camera.m_view_matrix * f_world_position).xyz;
        // eye-space view direction (points from surface toward eye at origin)
        vec3 eyedir = -normalize(xpos);
        // eye-space normal
        vec3 normal = (u_camera.m_view_matrix * vec4(normalize(f_normal), 0)).xyz;

        // bail out immediately if normal is facing away from the camera,
        // dot(normal, eyedir) <= 0 means back-facing
        float NdotV = dot(normal, eyedir);
        if (NdotV <= 0.0)
        {
#if SSR_DEBUG_MODE == 1
            o_displace_ssr = vec4(1.0, 0.0, 1.0, 1.0);
#else
            o_displace_ssr = vec4(0.0);
#endif
            return;
        }

        // compute reflection in eye-space
        vec3 reflected = reflect(-eyedir, normal);
        // bring it back into world-space
        vec3 world_reflection = (u_camera.m_inverse_view_matrix *
            vec4(reflected, 0.0)).xyz;

        // fallback to skybox
        vec4 fallback = texture(u_skybox_texture, world_reflection);

        // early exit if normal is facing camera too directly (no meaningful reflection)
        if (normal.z < -0.75)
        {
#if SSR_DEBUG_MODE == 1
            o_displace_ssr = vec4(1.0, 1.0, 0.0, 1.0);
#else
            o_displace_ssr = fallback;
#endif
            return;
        }

#ifdef DISPLACE_RT
        vec4 result;
        // The hit is lit in hdr, the displace color is after the tonemap
        vec3 hdr;
        if (traceReflection(f_world_position.xyz, normalize(world_reflection),
            hdr))
        {
            float fresnel = (1.0 - NdotV) * (1.0 - NdotV);
            result = mix(fallback, vec4(convertColor(hdr), 1.0), fresnel);
        }
        else
            result = fallback;
#else
        vec4 result;
        vec2 viewport_scale = u_camera.m_viewport.zw / u_camera.m_screensize.xy;
        vec2 viewport_offset = u_camera.m_viewport.xy / u_camera.m_screensize.xy;
        bool hit = true;
        vec2 coords;
#if SSR_DEBUG_MODE != 0
        vec3 dbgStart = vec3(0.0);
#endif
        if (u_hiz_iterations == 0)
        {
            coords = RayCast(reflected, xpos, u_camera.m_projection_matrix,
                viewport_scale, viewport_offset, u_depth);
        }
        else
        {
            // Trace HiZ to find the hit point.
#if SSR_DEBUG_MODE != 0
            hit = traceHiZView(xpos, reflected, u_camera.m_projection_matrix,
                coords, dbgStart);
#else
            vec3 startSS;
            hit = traceHiZView(xpos, reflected, u_camera.m_projection_matrix,
                coords, startSS);
#endif
            coords = coords * viewport_scale + viewport_offset;
        }
        vec2 viewport_coords = (coords - viewport_offset) / viewport_scale;
        if (!hit || viewport_coords.x < 0. || viewport_coords.x > 1. ||
            viewport_coords.y < 0. || viewport_coords.y > 1.)
        {
            result = fallback;
        }
        else
        {
            result = texture(u_displace_color, coords);
            float edge = GetEdgeFade(coords, viewport_scale, viewport_offset);
            //float fresnel = pow(1.0 - NdotV, 2.0);
            float fresnel = (1.0 - NdotV) * (1.0 - NdotV);
            float blend_weight = edge * fresnel;
            result = mix(fallback, result, blend_weight);
        }
#if SSR_DEBUG_MODE != 0
        if (u_hiz_iterations != 0)
        {
            bool outside = hit && (viewport_coords.x < 0. || viewport_coords.x > 1. ||
                viewport_coords.y < 0. || viewport_coords.y > 1.);
            result = dbgColor(hit, outside, viewport_coords, dbgStart);
        }
#endif
#endif
        o_displace_ssr = result;
    }
#endif
}
