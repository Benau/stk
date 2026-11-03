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
class GEVulkanCamera;
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

    // Snapshot of the camera for the light clusters (GEClusterDataGenerator of
    // the upstream commit), taken in prepare so generate only needs the buffer
    irr::core::matrix4 m_cluster_view_matrix;
    irr::core::vector3df m_cluster_far_left_down;
    irr::core::vector3df m_cluster_far_right_up;
    irr::core::vector2df m_cluster_screen_size;
    std::vector<irr::core::vector3df> m_cluster_x_planes;
    std::vector<irr::core::vector3df> m_cluster_y_planes;
    float m_cluster_near;
    bool m_cluster_camera_valid;
    // Offset of the cluster block in m_buffer and its size in the last
    // generated frame
    size_t m_cluster_offset;
    size_t m_cluster_size;

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
    // ------------------------------------------------------------------------
    void initClusterCamera(GEVulkanCamera* cam);
    // ------------------------------------------------------------------------
    void generateClusters();

public:
    // ------------------------------------------------------------------------
    GEVulkanLightHandler(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    ~GEVulkanLightHandler();
    // ------------------------------------------------------------------------
    // cam is the camera of the view that collects the UBO, it is used to
    // generate the light clusters (NULL only from the constructor)
    void prepare(GEVulkanCamera* cam = NULL);
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
    // Raw GPU buffer pointer – only the first getUploadSize() bytes are valid
    // data.
    void* getData()                                 { return m_buffer.data(); }
    // ------------------------------------------------------------------------
    // Bytes to upload for the current frame: the light block and the cluster
    // block after it
    size_t getUploadSize() const
                          { return m_cluster_offset + m_cluster_size; }
    // ------------------------------------------------------------------------
    // Total allocated buffer size – use this for the dynamic-buffer allocation
    // so the cluster block always fits
    size_t getMaxSize() const                             { return m_buffer.size(); }
    // ------------------------------------------------------------------------
    // Size of the light block only (header + MAX_RENDERING_LIGHT slots), the
    // range of the uniform buffer binding 3
    static size_t getMaxLightBlockSize();
    // ------------------------------------------------------------------------
    // Offset of the light cluster block in the buffer of the light handler,
    // used as the dynamic offset of the storage buffer binding 5
    static size_t getClusterOffset();
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
    // View depth where each sun cascade starts (an array of one value per
    // cascade) and the cosine of the largest angle between the view direction
    // and a frustum ray, see GEVulkanShadowFBO::prepare
    void setShadowSplit(const float* near_depths, float min_ray_cos);
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
