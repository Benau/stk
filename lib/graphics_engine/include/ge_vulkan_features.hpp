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
bool supportsBindMeshTexturesAtOnce();
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
