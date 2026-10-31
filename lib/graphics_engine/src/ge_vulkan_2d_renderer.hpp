#ifndef HEADER_GE_VULKAN_2D_RENDERER_HPP
#define HEADER_GE_VULKAN_2D_RENDERER_HPP

#include "vulkan_wrapper.h"

#include "ITexture.h"
#include "S3DVertex.h"

namespace GE
{
class GEVulkanDriver;
namespace GEVulkan2dRenderer
{
// ----------------------------------------------------------------------------
void init(GEVulkanDriver*);
// ----------------------------------------------------------------------------
void destroy();
// ----------------------------------------------------------------------------
void createPipelineLayout();
// ----------------------------------------------------------------------------
void createGraphicsPipeline();
// ----------------------------------------------------------------------------
void createTrisBuffers();
// ----------------------------------------------------------------------------
void uploadTrisBuffers();
// ----------------------------------------------------------------------------
// shadow_map_view is the 2D array of the sun shadow map when the shadow viz
// of the debug menu is on, it's drawn before the queued 2D triangles
void render(VkImageView shadow_map_view = VK_NULL_HANDLE);
// ----------------------------------------------------------------------------
void clear();
// ----------------------------------------------------------------------------
void addVerticesIndices(irr::video::S3DVertex* vertices,
                        unsigned vertices_count, uint16_t* indices,
                        unsigned indices_count,
                        const irr::video::ITexture* t);
// ----------------------------------------------------------------------------
void updateTextureDescriptor();
};   // GEVulkanRenderer

}

#endif
