#ifndef HEADER_GE_VULKAN_LIGHT_SCATTER_HPP
#define HEADER_GE_VULKAN_LIGHT_SCATTER_HPP

#include "ge_vulkan_low_res_effect.hpp"

namespace GE
{

// Volumetric fog and light scattering of the sun, the point lights and the
// spot lights, which replaces the light scattering of the OpenGL renderer
// (see light_scatter.comp), owned by GEVulkanDeferredFBOSplit. All the compute
// shaders and the images of it are in GEVulkanLowResEffect:
//
// 1. For each viewport (draw call) at half the size of it, light_scatter.comp
//    marches the ray of every pixel in the fog, using the camera and light
//    data of the draw call, the depth of the g-buffer (the ray ends at the
//    geometry) and the shadow map of the environment: the sun has the shadows
//    of its cascades and the point lights and spot lights have the ones of
//    their cube maps (the tiles of the omni atlas). The fog has the density of
//    the exponential fog of the track, faded by its height
//    (fog-start-height/fog-end-height).
//    The light shaft of the track (GEVulkanDriver::setLightShaft) is the sun
//    which the camera sees: it colors the scattering of the sun and gives the
//    direction of its phase. The disc of the sun and its rays are rendered by
//    GEVulkanGodRays.
// 2. The blur (never reading outside of the rect of the viewport).
// 3. The lighting pass adds it to hdr together with the glow outline (see
//    lighting_composite.frag and GEVulkanDrawCall::renderLightingComposite),
//    after the normal fog of the geometry and the skybox
class GEVulkanLightScatter : public GEVulkanLowResEffect
{
protected:
    // ------------------------------------------------------------------------
    virtual bool hasEffect(const GEVulkanDrawCall* dc) const;
    // ------------------------------------------------------------------------
    virtual void pushConstants(VkCommandBuffer cmd, const Viewport& vp);
public:
    // ------------------------------------------------------------------------
    GEVulkanLightScatter(GEVulkanDriver* vk,
                         const irr::core::dimension2d<irr::u32>& size,
                         GEVulkanAttachmentTexture* depth);
};   // GEVulkanLightScatter

}

#endif
