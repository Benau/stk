#ifndef HEADER_GE_VULKAN_LIGHT_SCATTER_HPP
#define HEADER_GE_VULKAN_LIGHT_SCATTER_HPP

#include "ge_vulkan_low_res_effect.hpp"

namespace GE
{

// Light scattering of the point lights in the fog (the same look as the
// OpenGL renderer, see light_scatter.comp), owned by GEVulkanDeferredFBOSplit.
// All the compute shaders and the images of it are in GEVulkanLowResEffect:
//
// 1. For each viewport (draw call) at half the size of it, light_scatter.comp
//    marches the ray of every pixel in the fog lit by the lights, using the
//    camera and light data of the draw call and the depth of the g-buffer.
// 2. The blur (never reading outside of the rect of the viewport).
// 3. The lighting pass adds it to hdr together with the glow outline (see
//    lighting_composite.frag and GEVulkanDrawCall::renderLightingComposite),
//    so it's above the fog of the lit geometry and the sky
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
