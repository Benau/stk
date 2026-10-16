#ifndef HEADER_VULKAN_WRAPPER_HPP
#define HEADER_VULKAN_WRAPPER_HPP

#if !defined(__APPLE__) || defined(DLOPEN_MOLTENVK)
#include <glad/vulkan.h>

#ifdef DLOPEN_MOLTENVK
#define VK_NO_PROTOTYPES 1
// We copy mvk_private_api.h with #include <vulkan/vulkan.h>
// removed
#include <mvk_private_api.h>
extern PFN_vkGetPhysicalDeviceMetalFeaturesMVK vkGetPhysicalDeviceMetalFeaturesMVK;
#endif

#else
#include <vulkan/vulkan.h>

#if defined(__APPLE__)
#include <MoltenVK/mvk_private_api.h>
#if defined(MVK_VERSION) && MVK_VERSION >= MVK_MAKE_VERSION(1, 4, 0)
#include <MoltenVK/mvk_deprecated_api.h>
#endif

// Link-only stubs, remove after ray tracing PR in MoltenVK is merged
inline VKAPI_ATTR VkResult VKAPI_CALL
vkCreateAccelerationStructureKHR(
    VkDevice,
    const VkAccelerationStructureCreateInfoKHR*,
    const VkAllocationCallbacks*,
    VkAccelerationStructureKHR* pAccelerationStructure)
{
    if (pAccelerationStructure)
        *pAccelerationStructure = VK_NULL_HANDLE;
    return VK_ERROR_EXTENSION_NOT_PRESENT;
}

inline VKAPI_ATTR void VKAPI_CALL
vkDestroyAccelerationStructureKHR(
    VkDevice, VkAccelerationStructureKHR,
    const VkAllocationCallbacks*)
{
}

inline VKAPI_ATTR void VKAPI_CALL
vkCmdBuildAccelerationStructuresKHR(
    VkCommandBuffer, uint32_t,
    const VkAccelerationStructureBuildGeometryInfoKHR*,
    const VkAccelerationStructureBuildRangeInfoKHR* const*)
{
}

inline VKAPI_ATTR void VKAPI_CALL
vkGetAccelerationStructureBuildSizesKHR(
    VkDevice, VkAccelerationStructureBuildTypeKHR,
    const VkAccelerationStructureBuildGeometryInfoKHR*,
    const uint32_t*,
    VkAccelerationStructureBuildSizesInfoKHR* pSizeInfo)
{
    if (pSizeInfo)
    {
        pSizeInfo->accelerationStructureSize = 0;
        pSizeInfo->updateScratchSize = 0;
        pSizeInfo->buildScratchSize = 0;
    }
}

inline VKAPI_ATTR VkDeviceAddress VKAPI_CALL
vkGetAccelerationStructureDeviceAddressKHR(
    VkDevice, const VkAccelerationStructureDeviceAddressInfoKHR*)
{
    return 0;
}

#endif

#endif

#endif

#ifndef TILED_GPU
#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined (_M_ARM64)
#define TILED_GPU 1
#endif
#endif
