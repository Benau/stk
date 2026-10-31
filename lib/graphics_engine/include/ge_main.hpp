#ifndef HEADER_GE_MAIN_HPP
#define HEADER_GE_MAIN_HPP

#include <IVideoDriver.h>
#include <SColor.h>
#include <matrix4.h>
#include <quaternion.h>
#include <vector3d.h>

#include <array>
#include <cstdint>
#include <string>
#include <unordered_set>

namespace irr
{
    namespace scene
    {
        class IMesh; class IAnimatedMesh;
    }
}

namespace GE
{
class GEOcclusionCulling;
class GESPMBuffer;
class GEVulkanDriver;
enum GELogLevel : unsigned
{
    GLL_VERBOSE,
    GLL_INFO,
    GLL_WARNING,
    GLL_ERROR
};
// The anti-aliasing of the final image of the FBOs (GEVulkanFinalChain). The
// game saves the name of it (see getAntiAliasingFromName), not the value, so
// the order doesn't matter. Needs compute in the main queue (see
// isAntiAliasingEnabled), and renders the scene to a FBO even without render
// scaling and deferred rendering
enum GEAntiAliasing : unsigned
{
    GAA_NONE = 0,
    // Fast approximate anti-aliasing, one compute kernel with a few samples
    // per pixel, the fallback for the low-end devices (see GEVulkanFXAA)
    GAA_FXAA,
    // Conservative Morphological Anti-Aliasing 2 (compute, see
    // GEVulkanAntiAliasing)
    GAA_CMAA2,
    // Reserved for the next ones: GAA_SMAA, GAA_TAA
    GAA_COUNT
};
enum GEAutoDeferredType : unsigned
{
    GADT_DISABLED = 0,
    GADT_SINGLE_PASS,
    GADT_DISPLACE
};
// Reflection of the solid (opaque) materials, shared numeric values with
// SOLID_REFLECTION_* in constants_utils.glsl
enum GESolidReflectionType : unsigned
{
    GSRT_DISABLED = 0,
    // Screen space reflection of the previous frame (needs the split deferred
    // FBO), FAST uses the depth buffer, HIZ* use the hierarchical depth
    GSRT_SSR_FAST,
    GSRT_SSR_HIZ100,
    GSRT_SSR_HIZ200,
    GSRT_SSR_HIZ400,
    GSRT_COUNT,
};
// Reflection of the displace (water) materials, shared numeric values with
// DISPLACE_REFLECTION_* in constants_utils.glsl
enum GEDisplaceReflectionType : unsigned
{
    GDRT_DISABLED = 0,
    GDRT_SSR_FAST,
    GDRT_SSR_HIZ100,
    GDRT_SSR_HIZ200,
    GDRT_SSR_HIZ400,
    // Number of the settings, which can't be set to GEConfig::
    // m_displace_reflection
    GDRT_COUNT,
    // Reflection traced in the ray traced scene, it's what getDisplaceReflection()
    // returns (never a value of the setting) when isDisplaceRayTracing()
    GDRT_RT = GDRT_COUNT,
};
inline bool isSolidReflectionSSR(GESolidReflectionType t)
                { return t >= GSRT_SSR_FAST && t <= GSRT_SSR_HIZ400; }
inline bool isSolidReflectionHiZ(GESolidReflectionType t)
                { return t >= GSRT_SSR_HIZ100 && t <= GSRT_SSR_HIZ400; }
inline bool isDisplaceReflectionHiZ(GEDisplaceReflectionType t)
                { return t >= GDRT_SSR_HIZ100 && t <= GDRT_SSR_HIZ400; }
// Settings of ray tracing (needs VK_KHR_ray_query, see isRayTracing()), the
// features below do nothing if m_enabled is false
struct GERayTracingConfig
{
    // Global switch
    bool m_enabled;
    // Reflection of the solid materials is traced, it overrides
    // GEConfig::m_solid_reflection (SSR is not used when it's on)
    bool m_solid_reflection;
    // Reflection of the displace materials (water, or anything which uses the
    // displace mask) is traced, it overrides GEConfig::m_displace_reflection
    // (the setting is not changed, it is used again if this is false)
    bool m_displace_reflection;
    // Moving nodes (GRTT_DYNAMIC) are in the traced scene too, including the
    // skinned (animated) meshes
    bool m_dynamic_objects;
    // Materials which discard by the alpha of their texture (alphatest and
    // unlit shaders) are in the traced scene, otherwise the rays pass through
    // them
    bool m_alphatest;
    // Grass (the grass material has a wind animation in its vertex shader) is
    // in the traced scene, it's baked every frame by a compute shader
    bool m_grass;
    // The shadows which are ray traced instead of using shadow maps: all the
    // ones of GEConfig::m_shadow_type, the sun and the point lights and spot
    // lights if they are on there (the first GEConfig::m_point_shadow_limit
    // lights of the rendered ones have a shadow, as with shadow maps), so the
    // setting of the point light shadows is the one of GEConfig (see
    // getRTShadowType()). No shadow map is created then, whatever
    // GEConfig::m_shadow_size is, and they are used again (with
    // GEConfig::m_shadow_size) if this is false
    bool m_shadow;
    // Resolution of the ray traced shadow relative to the size of the render
    // target (like the render scale), in [0.25, 1.0] (see getRTShadowScale()).
    // One ray is traced per texel, the lighting pass filters it (depth and
    // normal aware, never across the viewports of the splitscreen) to the full
    // resolution. Needs the render target to be created again if it changes
    float m_shadow_scale;
    // Surfel based global illumination (GEVulkanSurfelGI): the indirect diffuse
    // of the visible surfaces is ray traced and added to the ambient light of
    // the lighting pass, see isSurfelGIEnabled()
    bool m_surfel_gi;
    // ------------------------------------------------------------------------
    GERayTracingConfig()
    {
        m_enabled = false;
        m_solid_reflection = false;
        m_displace_reflection = false;
        m_dynamic_objects = false;
        m_alphatest = false;
        m_grass = false;
        m_shadow = false;
        m_shadow_scale = 1.0f;
        m_surfel_gi = false;
    }
};
enum GEShadowType : unsigned
{
    GST_NONE = 0,
    GST_SUN = 1,
    GST_POINTLIGHT = 1 << 1,
    GST_COMBINED = GST_SUN | GST_POINTLIGHT
};

struct GEConfig
{
bool m_disable_npot_texture;
bool m_convert_irrlicht_mesh;
bool m_texture_compression;
bool m_fullscreen_desktop;
bool m_pbr;
bool m_ibl;
unsigned m_deferred_split;
bool m_glow_outline;
// Light scattering of point lights in fog (needs the split deferred FBO and
// compute in the main queue, see GEVulkanLightScatter)
bool m_light_scatter;
// Light shafts (god rays) of the sun of a track which has them (see
// GEVulkanDriver::setGodRays), compute shaders in the split deferred FBO (see
// GEVulkanGodRays)
bool m_god_rays;
// Bloom of the finished hdr color (needs the split deferred FBO and compute in
// the main queue, see GEVulkanBloom)
bool m_bloom;
// See GERayTracingConfig, use isRayTracing() and isSolidRayTracing() to know
// if the device and the other settings allow it
GERayTracingConfig m_ray_tracing;
// Full path of an optional texture (lens dust for example) which the widest
// blur of the bloom is multiplied by and added to it, mapped on the whole
// viewport. Empty if there is none. It's loaded (and owned) by the deferred FBO
// so it must be set before the driver (or updateDriver) creates it
std::string m_bloom_blend_texture;
// Screen space ambient occlusion of the ambient light (needs the split deferred
// FBO and compute in the main queue, see GEVulkanSSAO)
bool m_ssao;
GEAutoDeferredType m_auto_deferred_type;
GESolidReflectionType m_solid_reflection;
GEDisplaceReflectionType m_displace_reflection;
std::unordered_set<std::string> m_ondemand_load_texture_paths;
float m_render_scale;
unsigned m_shadow_size;
GEShadowType m_shadow_type;
unsigned m_point_shadow_limit;
unsigned m_max_pointlights;
// Ambient occlusion, can be changed any time: the radius in world units (of
// GEVulkanSSAO, which has no other parameter with VBAO, and of the OpenGL
// renderer), and the power and intensity of the OpenGL renderer only
float m_ssao_radius;
float m_ssao_k;
float m_ssao_sigma;
// Debug view of GEVulkanSSAO (the "SSAO viz" of the debug menu, like the one
// of the OpenGL renderer): deferred_convert_color.frag writes the ambient
// occlusion to the final image instead of the tonemapped lighting. It's a push
// constant of that shader, so it can be changed any time
bool m_ssao_debug;
// Same for GEVulkanSurfelGI (the "GI viz" of the debug menu): the irradiance of
// the surfels and the indirect light of the ambient occlusion (if it's
// enabled) are written to the final image, without bloom and tonemap, so it
// shows exactly what is added to the ambient light. The images are half size,
// so every 2x2 block of pixels has the same value
bool m_gi_debug;
// See GEAntiAliasing, use isAntiAliasingEnabled() to know if it's used
GEAntiAliasing m_anti_aliasing;
// Depth of field of the final image (a stage of GEVulkanFinalChain which runs
// before the anti-aliasing, see GEVulkanDepthOfField), use
// isDepthOfFieldEnabled() to know if it's used. Needs the split deferred FBO
// (it's selected by refreshDeferredSplit()) because it reads the depth of it.
// Only the main FBO has it, not the separate render targets. Changing it needs
// the swapchain to be created again, like the anti-aliasing
bool m_dof;
// The next ones can be changed any time. Distances are in world units, from
// the position of the camera (the distance of the point, not the depth along
// the view direction, so it doesn't change when the camera rotates):
// distance of the center of the zone which is in focus
float m_dof_focus_distance;
// Half of the length of the zone which is in focus: nothing is blurred in
// [focus_distance - focus_range, focus_distance + focus_range]
float m_dof_focus_range;
// Length behind the zone in focus where the blur grows to the maximum
float m_dof_far_transition;
// Same in front of the zone in focus, 0 (default) disables the blur of the
// objects close to the camera
float m_dof_near_transition;
// Radius of the maximum blur in pixels of an image 1080 pixels high (it's
// scaled with the height of the viewport, so the blur is the same part of the
// screen at any resolution, render scale and splitscreen). Up to ~16
float m_dof_max_radius;
};

// True if GEConfig::m_anti_aliasing is not GAA_NONE and the device can run it
bool isAntiAliasingEnabled();
// True if GEConfig::m_dof is on, the device can run it and the pbr shaders are
// used (the split deferred FBO is needed)
bool isDepthOfFieldEnabled();
// True if the scene is rendered to a FBO which has the stages of
// GEVulkanFinalChain: the anti-aliasing, the depth of field, or a pass added
// with addFinalPass() (ge_final_pass.hpp) which is enabled
bool isFinalChainEnabled();
// True if an enabled pass of the final chain reads the depth of the FBO (the
// depth of field is one), which needs the split deferred FBO
bool needsFinalPassDepth();
// The anti-aliasing of the lower case name the game saves in its config: "fxaa"
// or "cmaa2", GAA_NONE for any other name (the game resolves the names of the
// other renderers itself)
GEAntiAliasing getAntiAliasingFromName(const std::string& name);
// True if ray tracing is on and the device supports it
bool isRayTracing();
// True if the reflection of the solid materials is traced (which overrides
// GEConfig::m_solid_reflection), needs the pbr shaders
bool isSolidRayTracing();
// Same for the reflection of the displace materials (GERayTracingConfig::
// m_displace_reflection)
bool isDisplaceRayTracing();
// True if the surfel based global illumination is on and the device supports it
// (GERayTracingConfig::m_surfel_gi), needs the pbr shaders and the split
// deferred FBO (which is used with ray tracing)
bool isSurfelGIEnabled();
// What the displace reflection is with isDisplaceRayTracing() taken into
// account: GDRT_RT if it's ray traced, GEConfig::m_displace_reflection
// otherwise. Everything which needs the displace reflection (the attachments
// of the FBO, the HiZ depth, the shaders) uses it instead of the setting
GEDisplaceReflectionType getDisplaceReflection();
// What the solid reflection is with isSolidRayTracing() taken into account,
// only use these (not GEConfig::m_solid_reflection) to know if SSR is used
bool isSolidSSR();
bool isSolidHiZ();
GESolidReflectionType getSolidReflection();
// The bits of GEConfig::m_shadow_type which are ray traced (all of them with
// GERayTracingConfig::m_shadow), GST_NONE if ray tracing or the pbr shaders are
// off. A track without shadow of the sun (no GST_SUN in m_shadow_type) has no
// ray traced shadow of it either, and the same for the point lights
GEShadowType getRTShadowType();
// GERayTracingConfig::m_shadow_scale clamped to [0.25, 1.0]
float getRTShadowScale();
// The shadows which use shadow maps, GEConfig::m_shadow_type without the ones
// of getRTShadowType(), use it (not GEConfig::m_shadow_type) to create the
// shadow maps and in the shaders
GEShadowType getShadowType();
// GEConfig::m_shadow_size if there is any shadow map (getShadowType() isn't
// GST_NONE), otherwise 0
unsigned getShadowSize();
void setVideoDriver(irr::video::IVideoDriver* driver);
void setShaderFolder(const std::string& path);
irr::video::IVideoDriver* getDriver();
GE::GEVulkanDriver* getVKDriver();
const std::string& getShaderFolder();
GEConfig* getGEConfig();
void deinit();
uint64_t getMonoTimeMs();
void mathPlaneFrustumf(float* out, const irr::core::matrix4& pvm);
inline size_t getPadding(size_t in, size_t alignment)
{
    if (in == 0 || alignment == 0)
        return 0;
    size_t mod = in % alignment;
    if (mod == 0)
        return 0;
    else
        return alignment - mod;
}
inline int get4x4CompressedTextureSize(int width, int height)
{
    int blockcount = ((width + 3) / 4) * ((height + 3) / 4);
    int blocksize = 4 * 4;
    return blockcount * blocksize;
}
irr::scene::IAnimatedMesh* convertIrrlichtMeshToSPM(irr::scene::IMesh* mesh);
inline uint8_t srgb255ToLinear(unsigned color_srgb_255)
{
    static unsigned srgb_linear_map[256] =
    {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
        2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4,
        4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7,
        7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11,
        11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16,
        16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 22, 22,
        23, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29,
        30, 31, 31, 32, 33, 33, 34, 35, 36, 36, 37, 38,
        38, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 55, 56, 57, 58,
        59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71,
        72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84,
        85, 87, 88, 89, 90, 92, 93, 94, 95, 97, 98, 99,
        101, 102, 103, 105, 106, 107, 109, 110, 112, 113, 114, 116,
        117, 119, 120, 122, 123, 125, 126, 128, 129, 131, 132, 134,
        135, 137, 139, 140, 142, 144, 145, 147, 148, 150, 152, 153,
        155, 157, 159, 160, 162, 164, 166, 167, 169, 171, 173, 175,
        176, 178, 180, 182, 184, 186, 188, 190, 192, 193, 195, 197,
        199, 201, 203, 205, 207, 209, 211, 213, 215, 218, 220, 222,
        224, 226, 228, 230, 232, 235, 237, 239, 241, 243, 245, 248,
        250, 252, 255
    };
    return uint8_t(srgb_linear_map[color_srgb_255]);
}
inline irr::video::SColor srgb255ToLinearFromSColor(irr::video::SColor scolor_srgb)
{
    irr::video::SColor out = scolor_srgb;
    out.setRed(srgb255ToLinear(scolor_srgb.getRed()));
    out.setGreen(srgb255ToLinear(scolor_srgb.getGreen()));
    out.setBlue(srgb255ToLinear(scolor_srgb.getBlue()));
    return out;
}
void copyToMappedBuffer(uint32_t* mapped, GESPMBuffer* spmb, size_t offset = 0);
GEOcclusionCulling* getOcclusionCulling();
void resetOcclusionCulling();
bool hasOcclusionCulling();
bool needsDeferredRendering();
std::array<float, 4>& getDisplaceDirection();
void setAbsoluteRotationScale(const irr::core::matrix4& model_matrix,
                              irr::core::quaternion& rotation,
                              irr::core::vector3df& scale);
// Decides (and stores in GEConfig::m_deferred_split) whether the deferred
// pipeline is split into separate render passes (GEVulkanDeferredFBOSplit,
// no input attachments, SPLIT shader define) or uses the single render pass
// with subpasses (GEVulkanDeferredFBO). It's stored in the config because
// the constructor of GEVulkanDeferredFBO (which GEVulkanDeferredFBOSplit
// derives from) needs to know it too before a virtual call is possible
void refreshDeferredSplit();
inline const char* getLightningIdent() { return "lightning_ident"; }
typedef void (*GELogCallback)(GELogLevel level, const char* message);
void setLogCallback(GELogCallback callback);
void logMessage(GELogLevel level, const char* format, ...);

}
#endif
