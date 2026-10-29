#include "ge_vulkan_features.hpp"

#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "../source/Irrlicht/os.h"
#include <SDL_cpuinfo.h>

extern "C" const char* Android_Custom_Vulkan_Driver_In_Use();

namespace GE
{
namespace GEVulkanFeatures
{
// ============================================================================
bool g_supports_bind_textures_at_once = false;
bool g_supports_rgba8_blit = false;
bool g_supports_r8_blit = false;
// https://chunkstories.xyz/blog/a-note-on-descriptor-indexing
bool g_supports_descriptor_indexing = false;
bool g_supports_partially_bound = false;
uint32_t g_max_sampler_supported = 0;
bool g_supports_multi_draw_indirect = false;
bool g_supports_base_vertex_rendering = true;
bool g_supports_compute_in_main_queue = false;
bool g_supports_shader_draw_parameters = false;
bool g_supports_s3tc_bc3 = false;
bool g_supports_bptc_bc7 = false;
bool g_supports_astc_4x4 = false;
bool g_supports_shader_storage_image_extended_format = false;
bool g_supports_dynamic_rendering = false;
bool g_supports_dynamic_rendering_local_read = false;
bool g_supports_ray_query = false;
uint32_t g_as_max_geometry_count = 0;
uint32_t g_as_scratch_alignment = 256;
// Textures can be an array in shader (before the overrides which make the mesh
// textures per material, those are for the raster passes)
bool g_bind_textures_hw = false;
// 0 is not a valid VkDriverId, it means unknown
int g_driver_id = 0;
std::string g_driver_name;
std::string g_driver_info;
// The drivers (Turnip) which are much slower when the textures are indexed in
// the shader (no prefetch of the texture as the index isn't a constant), the
// textures are bound per material then (the same as when the textures can't be
// bound at once)
bool g_slow_bind_all_textures = false;
// Can force g_slow_bind_all_textures, 1 is per material, 0 is bound at once
const char* SLOW_BIND_ALL_TEXTURES_ENV = "GE_NO_BIND_ALL_TEXTURES";
}   // GEVulkanFeatures

extern "C" int enableRayQuery();
// ============================================================================
void GEVulkanFeatures::init(GEVulkanDriver* vk)
{
    g_supports_bind_textures_at_once = true;
    bool dynamic_indexing = true;
    VkPhysicalDeviceLimits limit = vk->getPhysicalDeviceProperties().limits;
    // https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxDescriptorSetSamplers&platform=all
    // https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxDescriptorSetSampledImages&platform=all
    // https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPerStageDescriptorSamplers&platform=all
    // https://vulkan.gpuinfo.org/displaydevicelimit.php?name=maxPerStageDescriptorSampledImages&platform=all
    // We decide 512 (GEVulkanShaderManager::getSamplerSize()) based on those infos
    g_max_sampler_supported = std::min(
    {
        limit.maxDescriptorSetSamplers,
        limit.maxDescriptorSetSampledImages,
        limit.maxPerStageDescriptorSamplers,
        limit.maxPerStageDescriptorSampledImages
    });
    const unsigned max_sampler_size = GEVulkanShaderManager::getSamplerSize();
    if (max_sampler_size > g_max_sampler_supported)
        g_supports_bind_textures_at_once = false;
    if (vk->getPhysicalDeviceFeatures().shaderSampledImageArrayDynamicIndexing == VK_FALSE)
        dynamic_indexing = false;
    g_supports_multi_draw_indirect = vk->getPhysicalDeviceFeatures().multiDrawIndirect &&
         vk->getPhysicalDeviceFeatures().drawIndirectFirstInstance;

    VkFormatProperties format_properties = {};
    vkGetPhysicalDeviceFormatProperties(vk->getPhysicalDevice(),
        VK_FORMAT_R8G8B8A8_UNORM, &format_properties);
    g_supports_rgba8_blit = format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    format_properties = {};
    vkGetPhysicalDeviceFormatProperties(vk->getPhysicalDevice(),
        VK_FORMAT_R8_UNORM, &format_properties);
    g_supports_r8_blit = format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    format_properties = {};
    vkGetPhysicalDeviceFormatProperties(vk->getPhysicalDevice(),
        VK_FORMAT_BC3_UNORM_BLOCK, &format_properties);
    g_supports_s3tc_bc3 = format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
#ifdef BASISU_TC
    format_properties = {};
    vkGetPhysicalDeviceFormatProperties(vk->getPhysicalDevice(),
        VK_FORMAT_BC7_UNORM_BLOCK, &format_properties);
    g_supports_bptc_bc7 = format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    format_properties = {};
    vkGetPhysicalDeviceFormatProperties(vk->getPhysicalDevice(),
        VK_FORMAT_ASTC_4x4_UNORM_BLOCK, &format_properties);
    g_supports_astc_4x4 = format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
#endif
    g_supports_shader_storage_image_extended_format = vk->getPhysicalDeviceFeatures().shaderStorageImageExtendedFormats;
    if (g_supports_shader_storage_image_extended_format)
    {
        // iOS simulator doesn't support writing to VK_FORMAT_A2B10G10R10_UNORM_PACK32
        try
        {
            std::vector<VkFormat> a2b10g10r10 =
            {
                VK_FORMAT_A2B10G10R10_UNORM_PACK32,
            };
            VkFormat format = vk->findSupportedFormat(a2b10g10r10,
                VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
                VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
                VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT);
        }
        catch (std::runtime_error& e)
        {
            g_supports_shader_storage_image_extended_format = false;
        }
    }

    uint32_t extension_count;
    vkEnumerateDeviceExtensionProperties(vk->getPhysicalDevice(), NULL,
        &extension_count, NULL);
    std::vector<VkExtensionProperties> extensions(extension_count);
    vkEnumerateDeviceExtensionProperties(vk->getPhysicalDevice(), NULL,
        &extension_count, &extensions[0]);

    for (VkExtensionProperties& prop : extensions)
    {
        if (strcmp(prop.extensionName,
            VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME) == 0 &&
            dynamic_indexing)
            g_supports_descriptor_indexing = true;
    }

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(vk->getPhysicalDevice(),
        &queue_family_count, NULL);
    if (queue_family_count != 0)
    {
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(vk->getPhysicalDevice(),
            &queue_family_count, &queue_families[0]);
        uint32_t main_family = vk->getGraphicsFamily();
        if (main_family < queue_families.size())
        {
            g_supports_compute_in_main_queue =
                (queue_families[main_family].queueFlags & VK_QUEUE_COMPUTE_BIT)
                != 0;
        }
    }

    VkPhysicalDeviceFeatures2 supported_features = {};
    supported_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing_features = {};
    descriptor_indexing_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    supported_features.pNext = &descriptor_indexing_features;

    VkPhysicalDeviceShaderDrawParametersFeatures shader_draw = {};
    shader_draw.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES;
    descriptor_indexing_features.pNext = &shader_draw;

    VkPhysicalDeviceDynamicRenderingFeatures dynamic_rendering_features = {};
    dynamic_rendering_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    shader_draw.pNext = &dynamic_rendering_features;

    VkPhysicalDeviceDynamicRenderingLocalReadFeaturesKHR
        dynamic_rendering_local_read_features = {};
    dynamic_rendering_local_read_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_LOCAL_READ_FEATURES_KHR;
    dynamic_rendering_features.pNext =
        &dynamic_rendering_local_read_features;

    PFN_vkGetPhysicalDeviceFeatures2 get_features = vkGetPhysicalDeviceFeatures2;
    if (vk->getPhysicalDeviceProperties().apiVersion < VK_API_VERSION_1_1 ||
        !get_features)
    {
        get_features = (PFN_vkGetPhysicalDeviceFeatures2)
            vkGetPhysicalDeviceFeatures2KHR;
    }
    if (!get_features)
        return;
    get_features(vk->getPhysicalDevice(), &supported_features);
    if (supported_features.sType !=
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2)
        return;

    if (descriptor_indexing_features.shaderSampledImageArrayNonUniformIndexing
        != VK_TRUE)
        g_supports_descriptor_indexing = false;
    if (descriptor_indexing_features.runtimeDescriptorArray != VK_TRUE)
        g_supports_descriptor_indexing = false;
    g_supports_partially_bound = (descriptor_indexing_features
        .descriptorBindingPartiallyBound == VK_TRUE);
    g_supports_shader_draw_parameters = (shader_draw
        .shaderDrawParameters == VK_TRUE);
    g_supports_dynamic_rendering =
        (dynamic_rendering_features.dynamicRendering == VK_TRUE);
    g_supports_dynamic_rendering_local_read =
        (dynamic_rendering_local_read_features.dynamicRenderingLocalRead ==
        VK_TRUE);

    // Vulkan 1.4 implementations only have to support local read for storage
    // resources and single sampled color attachments
    if (vk->getPhysicalDeviceProperties().apiVersion >= VK_MAKE_VERSION(1, 4, 0))
    {
        VkPhysicalDeviceProperties2 properties = {};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;

        VkPhysicalDeviceVulkan14Properties vk_14_prop = {};
        vk_14_prop.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
        properties.pNext = &vk_14_prop;

        vkGetPhysicalDeviceProperties2(vk->getPhysicalDevice(), &properties);
        g_supports_dynamic_rendering_local_read = vk_14_prop
            .dynamicRenderingLocalReadDepthStencilAttachments == VK_TRUE;
    }

#if defined(__APPLE__)
    if (enableRayQuery() == 1)
    {
        // We checked for MTLArgumentBuffersTier2 support
        g_supports_bind_textures_at_once = true;
        g_max_sampler_supported = 1e6;
    }

    MVKPhysicalDeviceMetalFeatures mvk_features = {};
    size_t mvk_features_size = sizeof(MVKPhysicalDeviceMetalFeatures);
    vkGetPhysicalDeviceMetalFeaturesMVK(vk->getPhysicalDevice(), &mvk_features,
        &mvk_features_size);
    g_supports_base_vertex_rendering = mvk_features.baseVertexInstanceDrawing;
    if (!g_supports_base_vertex_rendering)
        g_supports_bind_textures_at_once = false;

    // https://github.com/KhronosGroup/MoltenVK/issues/1743
    g_supports_shader_draw_parameters = false;
#endif

    const bool original_bind_mesh_textures = supportsBindMeshTexturesAtOnce();
    g_driver_id = 0;
    g_driver_name.clear();
    g_driver_info.clear();
    bool has_driver_properties = vk->getPhysicalDeviceProperties().apiVersion >=
        VK_API_VERSION_1_2;
    for (VkExtensionProperties& prop : extensions)
    {
        if (strcmp(prop.extensionName,
            VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME) == 0)
            has_driver_properties = true;
    }
    if (has_driver_properties && vkGetPhysicalDeviceProperties2 &&
        vk->getPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_1)
    {
        VkPhysicalDeviceProperties2 properties = {};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        VkPhysicalDeviceDriverProperties driver_properties = {};
        driver_properties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
        properties.pNext = &driver_properties;
        vkGetPhysicalDeviceProperties2(vk->getPhysicalDevice(), &properties);
        g_driver_id = (int)driver_properties.driverID;
        g_driver_name = driver_properties.driverName;
        g_driver_info = driver_properties.driverInfo;
    }
    // Only Turnip is known to need it, add the other drivers here when they
    // are tested, they can be forced by the environment variable meanwhile
    g_slow_bind_all_textures = g_driver_id == VK_DRIVER_ID_MESA_TURNIP;
    const char* force_per_material = std::getenv(SLOW_BIND_ALL_TEXTURES_ENV);
    if (force_per_material != NULL && force_per_material[0] != '\0')
        g_slow_bind_all_textures = atoi(force_per_material) != 0;

    // Ray query needs Vulkan 1.2 (buffer device address and SPIR-V 1.4 are
    // core), the mesh textures (as an array) for hit shading, and a few
    // descriptor sets for the ray traced pass
    g_supports_ray_query = false;
    if (vk->getPhysicalDeviceProperties().apiVersion >= VK_API_VERSION_1_2 &&
        vkGetPhysicalDeviceFeatures2 && original_bind_mesh_textures &&
        g_supports_dynamic_rendering && supportsComputeInMainQueue() &&
        vk->getPhysicalDeviceProperties().limits.maxBoundDescriptorSets >= 8)
    {
        bool has_as = false;
        bool has_ray_query = false;
        bool has_deferred_host_ops = false;
        for (VkExtensionProperties& prop : extensions)
        {
            if (strcmp(prop.extensionName,
                VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0)
                has_as = true;
            else if (strcmp(prop.extensionName,
                VK_KHR_RAY_QUERY_EXTENSION_NAME) == 0)
                has_ray_query = true;
            else if (strcmp(prop.extensionName,
                VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME) == 0)
                has_deferred_host_ops = true;
        }
        if (has_as && has_ray_query && has_deferred_host_ops)
        {
            VkPhysicalDeviceFeatures2 rt_features = {};
            rt_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
            VkPhysicalDeviceBufferDeviceAddressFeatures bda = {};
            bda.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
            VkPhysicalDeviceAccelerationStructureFeaturesKHR as_features = {};
            as_features.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
            VkPhysicalDeviceRayQueryFeaturesKHR ray_query_features = {};
            ray_query_features.sType =
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
            rt_features.pNext = &bda;
            bda.pNext = &as_features;
            as_features.pNext = &ray_query_features;
            vkGetPhysicalDeviceFeatures2(vk->getPhysicalDevice(), &rt_features);
            g_supports_ray_query = bda.bufferDeviceAddress == VK_TRUE &&
                as_features.accelerationStructure == VK_TRUE &&
                ray_query_features.rayQuery == VK_TRUE;
        }
    }
    if (g_supports_ray_query)
    {
        VkPhysicalDeviceProperties2 properties = {};
        properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        VkPhysicalDeviceAccelerationStructurePropertiesKHR as_properties = {};
        as_properties.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
        properties.pNext = &as_properties;
        vkGetPhysicalDeviceProperties2(vk->getPhysicalDevice(), &properties);
        g_as_max_geometry_count = (uint32_t)std::min<uint64_t>(
            as_properties.maxGeometryCount, 0xffffffffu);
        g_as_scratch_alignment = std::max(1u,
            as_properties.minAccelerationStructureScratchOffsetAlignment);
    }
}   // init

// ----------------------------------------------------------------------------
void GEVulkanFeatures::printStats()
{
    std::string driver_id;
    if (!g_driver_name.empty())
        driver_id += std::string(" (name: ") + g_driver_name + ")";
    if (!g_driver_info.empty())
        driver_id += std::string(" (info: ") + g_driver_info + ")";
    if (!driver_id.empty())
    {
        driver_id.erase(driver_id.begin());
        os::Printer::log("Vulkan driver id", driver_id.c_str());
    }
    os::Printer::log(
        "Vulkan has slow bind all textures at once (per material)",
        g_slow_bind_all_textures ? "true" : "false");
    os::Printer::log(
        "Vulkan can bind textures at once in shader",
        supportsBindTexturesAtOnce() ? "true" : "false");
    os::Printer::log(
        "Vulkan can bind mesh textures at once in shader",
        supportsBindMeshTexturesAtOnce() ? "true" : "false");
    os::Printer::log(
        "Vulkan supports linear blitting for rgba8",
        g_supports_rgba8_blit ? "true" : "false");
    os::Printer::log(
        "Vulkan supports linear blitting for r8",
        g_supports_r8_blit ? "true" : "false");
    os::Printer::log(
        "Vulkan supports VK_EXT_descriptor_indexing",
        g_supports_descriptor_indexing ? "true" : "false");
    os::Printer::log(
        "Vulkan supports multi-draw indirect",
        g_supports_multi_draw_indirect ? "true" : "false");
    os::Printer::log(
        "Vulkan supports base vertex rendering",
        g_supports_base_vertex_rendering ? "true" : "false");
    os::Printer::log(
        "Vulkan supports compute in main queue",
        g_supports_compute_in_main_queue ? "true" : "false");
    os::Printer::log(
        "Vulkan supports shader draw parameters",
        g_supports_shader_draw_parameters ? "true" : "false");
    os::Printer::log(
        "Vulkan supports s3 texture compression (bc3, dxt5)",
        g_supports_s3tc_bc3 ? "true" : "false");
    os::Printer::log(
        "Vulkan supports BPTC texture compression (bc7)",
        g_supports_bptc_bc7 ? "true" : "false");
    os::Printer::log(
        "Vulkan supports adaptive scalable texture compression (4x4 block)",
        supportsASTC4x4() ? "true" : "false");
    os::Printer::log("Vulkan supports shader storage image extended formats",
        supportsShaderStorageImageExtendedFormats() ? "true" : "false");
    os::Printer::log(
        "Vulkan supports dynamic rendering (VK_KHR_dynamic_rendering)",
        g_supports_dynamic_rendering ? "true" : "false");
    os::Printer::log(
        "Vulkan supports dynamic rendering local read",
        g_supports_dynamic_rendering_local_read ? "true" : "false");
    os::Printer::log(
        "Vulkan supports ray query (VK_KHR_ray_query)",
        g_supports_ray_query ? "true" : "false");
    os::Printer::log(
        "Vulkan descriptor can be partially bound",
        g_supports_partially_bound ? "true" : "false");
}   // printStats

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsBindTexturesAtOnce()
{
    return supportsDescriptorIndexing() && g_supports_bind_textures_at_once &&
        !g_slow_bind_all_textures;
}   // supportsBindTexturesAtOnce

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsRGBA8Blit()
{
    return g_supports_rgba8_blit;
}   // supportsRGBA8Blit

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsR8Blit()
{
    return g_supports_r8_blit;
}   // supportsR8Blit

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsDescriptorIndexing()
{
    return g_supports_descriptor_indexing;
}   // supportsDescriptorIndexing

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsPartiallyBound()
{
    return g_supports_partially_bound;
}   // supportsPartiallyBound

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsBindMeshTexturesAtOnce()
{
    if (!supportsBindTexturesAtOnce())
        return false;
    const unsigned sampler_count = GEVulkanShaderManager::getSamplerSize() *
        GEVulkanShaderManager::getMeshTextureLayer();
    return g_max_sampler_supported >= sampler_count;
}   // supportsBindMeshTexturesAtOnce

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::needsRTMeshTextureMirror()
{
    return supportsRayQuery() && !supportsBindMeshTexturesAtOnce();
}   // needsRTMeshTextureMirror

// ----------------------------------------------------------------------------
int GEVulkanFeatures::getDriverID()
{
    return g_driver_id;
}   // getDriverID

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsMultiDrawIndirect()
{
    return g_supports_multi_draw_indirect;
}   // supportsMultiDrawIndirect

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsBaseVertexRendering()
{
    return g_supports_base_vertex_rendering;
}   // supportsBaseVertexRendering

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsComputeInMainQueue()
{
    return g_supports_compute_in_main_queue;
}   // supportsComputeInMainQueue

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsShaderDrawParameters()
{
    return g_supports_shader_draw_parameters;
}   // supportsShaderDrawParameters

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsS3TCBC3()
{
    return g_supports_s3tc_bc3;
}   // supportsS3TCBC3

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsBPTCBC7()
{
    return g_supports_bptc_bc7;
}   // supportsBPTCBC7

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsASTC4x4()
{
    return g_supports_astc_4x4;
}   // supportsASTC4x4

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsShaderStorageImageExtendedFormats()
{
    return g_supports_shader_storage_image_extended_format;
}   // supportsShaderStorageImageExtendedFormats

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsDynamicRendering()
{
    return g_supports_dynamic_rendering;
}   // supportsDynamicRendering

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsDynamicRenderingLocalRead()
{
    return g_supports_dynamic_rendering_local_read;
} // supportsDynamicRenderingLocalRead

// ----------------------------------------------------------------------------
bool GEVulkanFeatures::supportsRayQuery()
{
    return g_supports_ray_query;
}   // supportsRayQuery

// ----------------------------------------------------------------------------
uint32_t GEVulkanFeatures::getAccelerationStructureMaxGeometryCount()
{
    return g_as_max_geometry_count;
}   // getAccelerationStructureMaxGeometryCount

// ----------------------------------------------------------------------------
uint32_t GEVulkanFeatures::getAccelerationStructureScratchAlignment()
{
    return g_as_scratch_alignment;
}   // getAccelerationStructureScratchAlignment

}
