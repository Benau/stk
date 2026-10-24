#ifndef HEADER_GE_VULKAN_GOD_RAYS_HPP
#define HEADER_GE_VULKAN_GOD_RAYS_HPP

#include "ge_vulkan_low_res_effect.hpp"

namespace GE
{

// Light shafts (god rays) of the sun of the track (GEVulkanDriver::setGodRays),
// the same look as the OpenGL renderer, owned by GEVulkanDeferredFBOSplit. The
// OpenGL renderer draws a sphere for the sun, here there is no mesh, graphics
// pipeline or render pass of its own, it's all compute shaders (see
// GEVulkanLowResEffect for the images, the blur and the lighting pass):
//
// 1. For each viewport (draw call) at a quarter of the size of it,
//    god_rays.comp marches from every pixel to the sun, which is a disc of the
//    screen space (using the camera data of the draw call) hidden by what is
//    closer in the depth of the g-buffer, and writes the sum of the sun mask
//    on the way (so the sun is in the image with its rays).
// 2. The blur, with a width which is the same fraction of the viewport as the
//    light scattering.
// 3. The lighting pass adds it to hdr (before the bloom, like the OpenGL
//    renderer)
class GEVulkanGodRays : public GEVulkanLowResEffect
{
protected:
    // ------------------------------------------------------------------------
    virtual bool hasEffect(const GEVulkanDrawCall* dc) const;
    // ------------------------------------------------------------------------
    virtual void pushConstants(VkCommandBuffer cmd, const Viewport& vp);
public:
    // ------------------------------------------------------------------------
    GEVulkanGodRays(GEVulkanDriver* vk,
                    const irr::core::dimension2d<irr::u32>& size,
                    GEVulkanAttachmentTexture* depth);
};   // GEVulkanGodRays

}

#endif
