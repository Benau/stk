#ifndef HEADER_GE_VULKAN_LIGHT_HANDLER_HPP
#define HEADER_GE_VULKAN_LIGHT_HANDLER_HPP

#include "matrix4.h"
#include "vector2d.h"
#include "vector3d.h"
#include "SColor.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace irr
{
    namespace scene
    {
        class ILightSceneNode;
    }
}

namespace GE
{
class GEVulkanDriver;
class GEVulkanSkyBoxRenderer;
struct GEVulkanCameraUBO;
enum GEVulkanShadowCameraCascade : unsigned;
struct GEGlobalLightBuffer;
struct GELight;
class GEVulkanLightHandler
{
private:
    GEVulkanDriver* m_vk;

    std::vector<uint8_t> m_buffer;

    struct GELightStorage;
    std::unique_ptr<GELightStorage> m_lights;

    unsigned m_fullscreen_light_count;

    unsigned m_non_occluded_lights;

    // ------------------------------------------------------------------------
    // Typed pointers into m_buffer (private helpers, defined in .cpp).
    GEGlobalLightBuffer* getGlobalLightPtr() const;
    // ------------------------------------------------------------------------
    GELight* getRenderingLightsPtr() const;
    // ------------------------------------------------------------------------
    // Picks the lights to render from the submitted ones, with fading
    void selectLights(const irr::core::vector3df& cam_pos);
    // ------------------------------------------------------------------------
    static unsigned getFadeExtraSlot();

public:
    // ------------------------------------------------------------------------
    GEVulkanLightHandler(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    ~GEVulkanLightHandler();
    // ------------------------------------------------------------------------
    void prepare();
    // ------------------------------------------------------------------------
    void generate(const irr::core::vector3df& cam_pos,
                  GEVulkanSkyBoxRenderer* skybox);
    // ------------------------------------------------------------------------
    void addLightNode(irr::scene::ILightSceneNode* node);
    // ------------------------------------------------------------------------
    // For lights that exist but are not submitted this frame (frustum culled),
    // so their fade state is kept and keeps advancing
    void touchLightNode(irr::scene::ILightSceneNode* node);
    // ------------------------------------------------------------------------
    // Raw GPU buffer pointer – only the first getSize() bytes are valid data.
    void* getData()                                 { return m_buffer.data(); }
    // ------------------------------------------------------------------------
    // Bytes of active data to upload (header + active lights only).
    size_t getSize() const;
    // ------------------------------------------------------------------------
    // Total allocated buffer size – use this for Vulkan descriptor ranges and
    // dynamic-buffer allocation so the full max-light slot is always reserved.
    static size_t getMaxSize();
    // ------------------------------------------------------------------------
    // Size of the light array in the GPU buffer and MAX_RENDERING_LIGHT in
    // the shaders: m_max_pointlights plus a few extra slots so lights that
    // are fading out can stay in while the selected set is already full.
    static unsigned getMaxRenderingLights();
    // ------------------------------------------------------------------------
    unsigned getLightCount() const;
    // ------------------------------------------------------------------------
    // 0 if there is no fog, which is also when there is no light scattering
    float getFogDensity() const;
    // ------------------------------------------------------------------------
    unsigned getFullscreenLightCount() const
                                           { return m_fullscreen_light_count; }
    // ------------------------------------------------------------------------
    void setShadowMatrices(GEVulkanCameraUBO* ubo,
                           GEVulkanShadowCameraCascade cc);
    // ------------------------------------------------------------------------
    // View depth where each sun cascade starts and the cosine of the largest
    // angle between the view direction and a frustum ray, see
    // GEVulkanShadowFBO::prepare
    void setShadowSplit(const irr::core::vector3df& near_depths,
                        float min_ray_cos);
    // ------------------------------------------------------------------------
    void setLightShadowMatrices(GEVulkanCameraUBO* ubo, unsigned light,
                                unsigned face);
    // ------------------------------------------------------------------------
    // Typed accessors for individual rendered lights (used by omni-shadow FBO).
    const irr::core::vector3df& getLightPosition(unsigned light_id) const;
    // ------------------------------------------------------------------------
    float getLightRadius(unsigned light_id) const;
    // ------------------------------------------------------------------------
    // Returns true if light_id is a spotlight (m_scale != 0).
    bool getLightIsSpot(unsigned light_id) const;
    // ------------------------------------------------------------------------
    // Returns the normalised spotlight direction, reconstructed from the
    // packed (X, Y) components and the sign encoded in m_scale.
    // Result is undefined for point lights.
    irr::core::vector3df getLightDirection(unsigned light_id) const;
    // ------------------------------------------------------------------------
    // Returns the outer-cone half-angle in radians (same convention as
    // Irrlicht's SLight::OuterCone).  Returns 0 for point lights.
    float getLightOuterCone(unsigned light_id) const;
    // ------------------------------------------------------------------------
    // Number of lights (in render order) that are not occluded.  The
    // rendered-light array is stable-partitioned so these come first.
    // Only meaningful when isDeferredFBO() + hasOcclusionCulling(); otherwise
    // equals getLightCount() (all lights are considered non-occluded).
    unsigned getNonOccludedLightCount() const    { return m_non_occluded_lights; }

};   // GEVulkanLightHandler

}

#endif
