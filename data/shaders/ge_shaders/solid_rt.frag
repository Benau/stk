// Ray traced reflection of the solid materials (GEVulkanDrawCall::
// renderSolidRT), a fullscreen draw at the end of the lighting pass like
// solid_ssr.frag, but the reflected ray is traced in the ray traced scene
// (GEVulkanRTScene, an acceleration structure of the static nodes in world
// space and instances of other ones) instead of the screen, so it has no
// history and no screen edges.
//
// The hit is shaded by utils/rt_reflection.glsl (shared with the reflection of
// the displace materials). A ray without a hit gets the specular IBL (same as
// solid_ssr.frag), a not glossy pixel is discarded, so the specular IBL which
// is already in the lit color stays (the output is premultiplied, blended by
// m_alphablend: src + dst * (1 - alpha)).
//
// Set 0 is the one of GEVulkanRTScene (acceleration structure, data buffer,
// geometry buffer, mesh cache), set 1 the camera and the global light, set 2 GVDFP_HDR
// (g-buffer, same as lighting), set 3 the environment and set 4 the mesh
// textures
#extension GL_EXT_ray_query : require

#define MESH_TEXTURES_SET 4

layout(set = 2, binding = 0) uniform sampler2D u_color;
layout(set = 2, binding = 1) uniform sampler2D u_normal;
layout(set = 2, binding = 2) uniform sampler2D u_depth;
layout(set = 3, binding = 0) uniform samplerCube u_diffuse;
layout(set = 3, binding = 1) uniform samplerCube u_specular;
// Only to share pbr_light.glsl, the shadow of a hit is a ray
layout(set = 3, binding = 3) uniform sampler2DArrayShadow u_shadow_map;

layout(location = 0) out vec4 o_color;

#include "utils/camera.glsl"
#include "utils/constants_utils.glsl"
#include "utils/pbr_utils.glsl"
#include "utils/unproject_position.glsl"
#include "../utils/decodeNormal.frag"
#include "utils/get_vertex_color.glsl"
#include "utils/sample_mesh_texture.glsl"
#include "../utils/rgb_conversion.frag"
#include "utils/global_light_data.glsl"
#include "utils/pbr_light.glsl"
#include "utils/rt_utils.glsl"
#include "utils/rt_scene.glsl"

#include "utils/rt_reflection.glsl"

void main()
{
    ivec2 pixel = ivec2(gl_FragCoord.xy);
    vec4 normal_data = texelFetch(u_normal, pixel, 0);
    // Gloss (specval of the OpenGL renderer), the cheapest rejection first
    float specval = normal_data.z;
    float depth = texelFetch(u_depth, pixel, 0).x;
    if (specval < 0.5 || depth == 0.0)
        discard;

    vec3 xpos = getPosFromUVDepth(vec3(gl_FragCoord.xy, depth),
        u_camera.m_viewport, u_camera.m_inverse_projection_matrix);
    vec3 eyedir = -normalize(xpos);
    vec3 view_normal = DecodeNormal(normal_data.xy);
    vec3 normal = (u_camera.m_view_matrix * vec4(view_normal, 0.0)).xyz;
    vec3 reflected = reflect(-eyedir, normal);

    // The scene is in world space
    vec3 world_position = (u_camera.m_inverse_view_matrix *
        vec4(xpos, 1.0)).xyz;
    vec3 world_reflection = normalize((u_camera.m_inverse_view_matrix *
        vec4(reflected, 0.0)).xyz);
    float cosine = dot(reflected, eyedir);

    vec3 hit_color;
    // A ray without a hit keeps the specular IBL of the lit color
    if (!traceReflection(world_position, world_reflection, hit_color))
        discard;

    // Replace the specular IBL of the lit color by the hit, with the same
    // Fresnel and F0 (metallic) of it, the diffuse is not touched. The
    // output is added (m_additive) and can be negative
    float perceptual_roughness = 1.0 - specval;
    float metallic = normal_data.w;
    vec3 albedo = texelFetch(u_color, pixel, 0).rgb;
    float NdotV = max(dot(normal, eyedir), 0.0001);
    vec3 spec_albedo = reflectionSpecularAlbedo(albedo, metallic,
        perceptual_roughness, NdotV);
    vec3 ibl = vec3(0.0);
    if (u_ibl)
    {
        ibl = textureLod(u_specular, world_reflection,
            perceptual_roughness * u_specular_levels_minus_one).rgb;
    }
    // Glossy 0.5 - 1.0 to 0.0 - 1.0 (the ray is sharp, a rougher surface
    // gets less of it)
    float weight = (specval - 0.5) * 2.0;
    o_color = vec4(weight * spec_albedo * (hit_color - ibl), 0.0);
}
