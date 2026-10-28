#ifndef HEADER_GE_VULKAN_FEATURES_HPP
#define HEADER_GE_VULKAN_FEATURES_HPP

#include "vulkan_wrapper.h"

namespace GE
{
class GEVulkanDriver;
namespace GEVulkanFeatures
{
// ----------------------------------------------------------------------------
void init(GEVulkanDriver*);
// ----------------------------------------------------------------------------
void printStats();
// ----------------------------------------------------------------------------
bool supportsBindTexturesAtOnce();
// ----------------------------------------------------------------------------
bool supportsRGBA8Blit();
// ----------------------------------------------------------------------------
bool supportsR8Blit();
// ----------------------------------------------------------------------------
bool supportsDescriptorIndexing();
// ----------------------------------------------------------------------------
bool supportsPartiallyBound();
// ----------------------------------------------------------------------------
// For the raster passes, false if the mesh textures are bound per material,
// which is also used when the driver is faster this way (see
// g_slow_bind_mesh_textures)
bool supportsBindMeshTexturesAtOnce();
// ----------------------------------------------------------------------------
// Ray query is supported but the mesh textures are per material for the raster
// passes, so GEVulkanTextureDescriptor has to provide an array for ray query
bool needsRTMeshTextureMirror();
// ----------------------------------------------------------------------------
// VkDriverId of VkPhysicalDeviceDriverProperties, 0 if unknown
int getDriverID();
// ----------------------------------------------------------------------------
bool supportsMultiDrawIndirect();
// ----------------------------------------------------------------------------
bool supportsBaseVertexRendering();
// ----------------------------------------------------------------------------
bool supportsComputeInMainQueue();
 // ----------------------------------------------------------------------------
bool supportsShaderDrawParameters();
// ----------------------------------------------------------------------------
bool supportsS3TCBC3();
// ----------------------------------------------------------------------------
bool supportsBPTCBC7();
// ----------------------------------------------------------------------------
bool supportsASTC4x4();
// ----------------------------------------------------------------------------
bool supportsShaderStorageImageExtendedFormats();
// ----------------------------------------------------------------------------
bool supportsDynamicRendering();
// ----------------------------------------------------------------------------
bool supportsDynamicRenderingLocalRead();
// ----------------------------------------------------------------------------
// VK_KHR_ray_query with acceleration structures and buffer device address
// (needs Vulkan 1.2 and bind mesh textures at once), used by the ray traced
// reflections
bool supportsRayQuery();
// ----------------------------------------------------------------------------
uint32_t getAccelerationStructureMaxGeometryCount();
// ----------------------------------------------------------------------------
uint32_t getAccelerationStructureScratchAlignment();
};   // GEVulkanFeatures

}

#endif
