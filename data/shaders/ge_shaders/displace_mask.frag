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

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/sample_mesh_texture.glsl"
#include "../utils/displace_utils.frag"

layout (set = 2, binding = 2) uniform samplerCube u_skybox_texture;
layout (set = 3, binding = 0) uniform sampler2D u_displace_color;
layout (set = 3, binding = 1) uniform sampler2DShadow u_depth;
layout (set = 3, binding = 2) uniform sampler2D u_hiz_depth;

#ifdef PBR_ENABLED
#define SSR_HIZ
#endif
#include "../utils/ssr.glsl"

void main()
{
#ifdef PBR_ENABLED
    float horiz = sampleMeshTexture2(f_material_id, f_uv + u_push_constants.m_displace_direction.xy * 150.).x;
    float vert = sampleMeshTexture2(f_material_id, (f_uv.yx + u_push_constants.m_displace_direction.zw * 150.) * vec2(0.9)).x;
    vec2 mask = getDisplaceShift(horiz, vert);
    mask = (mask + 1.0) * 0.5;
    o_displace_mask = mask;
    if (u_ssr)
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
        o_displace_ssr = result;
    }
#endif
}
