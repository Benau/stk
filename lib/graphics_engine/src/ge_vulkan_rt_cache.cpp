#include "ge_vulkan_rt_scene.hpp"
#include "ge_main.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_driver.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace GE
{
namespace
{
const uint64_t CACHE_VERSION = 0x53544b4153000001ull;
const uint64_t MAX_CACHE_BYTES = 256ull << 20;

uint64_t hashBytes(const void* source, size_t count, uint64_t seed)
{
    const uint8_t* bytes = static_cast<const uint8_t*>(source);
    for (size_t index = 0; index < count; ++index)
        seed = (seed ^ bytes[index]) * 1099511628211ull;
    return seed;
}

struct CacheTransfer
{
    GEVulkanDriver* driver;
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation memory = VK_NULL_HANDLE;
    VkDeviceAddress address = 0;
    VkDeviceSize padding = 0, bytes = 0;
    uint8_t* mapped = NULL;
    std::vector<VkDeviceSize> offsets;

    explicit CacheTransfer(GEVulkanDriver* vk) : driver(vk) {}
    ~CacheTransfer()
    {
        if (buffer)
            vmaDestroyBuffer(driver->getVmaAllocator(), buffer, memory);
    }

    bool allocate(const std::vector<uint64_t>& sizes, bool reading_from_gpu)
    {
        for (uint64_t size : sizes)
        {
            if (size < 56 || size > MAX_CACHE_BYTES - bytes)
                return false;
            offsets.push_back(bytes);
            bytes = (bytes + size + 255u) & ~uint64_t(255u);
        }
        VmaAllocationCreateInfo allocation = {};
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        allocation.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | (reading_from_gpu ?
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT : VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
        if (!driver->createBuffer(bytes + 255, VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, allocation, buffer, memory))
            return false;
        VkBufferDeviceAddressInfo buffer_info = {};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
        buffer_info.buffer = buffer;
        address = vkGetBufferDeviceAddress(driver->getDevice(), &buffer_info);
        padding = (0 - address) & 255u;
        address += padding;
        VmaAllocationInfo info;
        vmaGetAllocationInfo(driver->getVmaAllocator(), memory, &info);
        if (!info.pMappedData)
            return false;
        mapped = static_cast<uint8_t*>(info.pMappedData) + padding;
        return true;
    }
};

void cacheBarrier(VkCommandBuffer cmd, bool to_host)
{
    VkMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = to_host ? VK_ACCESS_HOST_READ_BIT : VK_ACCESS_MEMORY_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        to_host ? VK_PIPELINE_STAGE_HOST_BIT : VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        0, 1, &barrier, 0, NULL, 0, NULL);
}
}

std::string GEVulkanRTScene::staticCachePath(const std::vector<GERTVertex>& vertices,
    const std::vector<uint32_t>& indices, const std::vector<GERTGeometry>& infos,
    const std::vector<std::pair<uint32_t, uint32_t> >& counts) const
{
    const char* override_dir = std::getenv("STK_RT_BLAS_CACHE_DIR");
    const std::string directory = override_dir ? override_dir : getGEConfig()->m_rt_cache_directory;
    if (directory.empty())
        return {};
    const auto& device = m_vk->getPhysicalDeviceProperties();
    uint64_t hash = hashBytes(device.pipelineCacheUUID, VK_UUID_SIZE, CACHE_VERSION);
    hash = hashBytes(&device.driverVersion, sizeof(device.driverVersion), hash);
    hash = hashBytes(indices.data(), indices.size() * sizeof(uint32_t), hash);
    for (const GERTVertex& vertex : vertices)
        hash = hashBytes(vertex.m_position, sizeof(vertex.m_position), hash);
    for (size_t index = 0; index < infos.size(); ++index)
    {
        uint32_t geometry[] = { counts[index].first, counts[index].second, infos[index].m_shading_type };
        hash = hashBytes(geometry, sizeof(geometry), hash);
    }
    for (const StaticPart& part : m_static_parts)
        hash = hashBytes(&part.m_count, sizeof(part.m_count), hash);
    std::ostringstream path;
    path << directory << "/rt-blas-" << std::hex << hash << ".bin";
    return path.str();
}

bool GEVulkanRTScene::loadStaticCache(const std::string& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        return false;
    const uint64_t length = uint64_t(file.tellg());
    uint64_t header[3];
    std::vector<uint64_t> sizes(m_static_parts.size());
    uint64_t total = sizeof(header) + sizes.size() * sizeof(uint64_t);
    if (length < total || length > total + MAX_CACHE_BYTES)
        return false;
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(header), sizeof(header)) || header[0] != CACHE_VERSION ||
        header[1] != sizes.size() || !file.read(reinterpret_cast<char*>(sizes.data()), sizes.size() * sizeof(uint64_t)))
        return false;
    for (uint64_t size : sizes)
    {
        if (size < 56 || size > length - total)
            return false;
        total += size;
    }
    CacheTransfer transfer(m_vk);
    if (total != length || !transfer.allocate(sizes, false))
        return false;
    uint64_t hash = CACHE_VERSION;
    for (size_t index = 0; index < sizes.size(); ++index)
    {
        uint8_t* blob = transfer.mapped + transfer.offsets[index];
        if (!file.read(reinterpret_cast<char*>(blob), sizes[index]))
            return false;
        hash = hashBytes(blob, size_t(sizes[index]), hash);
        VkAccelerationStructureVersionInfoKHR version = {};
        version.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_VERSION_INFO_KHR;
        version.pVersionData = blob;
        VkAccelerationStructureCompatibilityKHR compatible;
        vkGetDeviceAccelerationStructureCompatibilityKHR(m_vk->getDevice(), &version, &compatible);
        uint64_t layout[3];
        memcpy(layout, blob + 2 * VK_UUID_SIZE, sizeof(layout));
        if (compatible != VK_ACCELERATION_STRUCTURE_COMPATIBILITY_COMPATIBLE_KHR || layout[0] < 56 ||
            layout[0] > sizes[index] || !layout[1] || layout[1] > m_static_parts[index].m_bytes || layout[2] != 0)
            return false;
    }
    if (hash != header[2])
        return false;
    vmaFlushAllocation(m_vk->getVmaAllocator(), transfer.memory, transfer.padding, transfer.bytes);
    VkCommandBuffer cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
    for (size_t index = 0; index < sizes.size(); ++index)
    {
        VkCopyMemoryToAccelerationStructureInfoKHR copy = {};
        copy.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_ACCELERATION_STRUCTURE_INFO_KHR;
        copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_DESERIALIZE_KHR;
        copy.src.deviceAddress = transfer.address + transfer.offsets[index];
        copy.dst = m_static_parts[index].m_handle;
        vkCmdCopyMemoryToAccelerationStructureKHR(cmd, &copy);
    }
    cacheBarrier(cmd, false);
    GEVulkanCommandLoader::endSingleTimeCommands(cmd);
    GE::logMessage(GLL_INFO, "Loaded %u prebuilt static BLAS", (unsigned)sizes.size());
    return true;
}

void GEVulkanRTScene::saveStaticCache(const std::string& path)
{
    VkDevice device = m_vk->getDevice();
    std::vector<VkAccelerationStructureKHR> handles;
    for (const StaticPart& part : m_static_parts)
        handles.push_back(part.m_handle);
    VkQueryPoolCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_SERIALIZATION_SIZE_KHR;
    info.queryCount = uint32_t(handles.size());
    VkQueryPool queries;
    if (vkCreateQueryPool(device, &info, NULL, &queries) != VK_SUCCESS)
        return;
    VkCommandBuffer cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
    cacheBarrier(cmd, false);
    vkCmdResetQueryPool(cmd, queries, 0, info.queryCount);
    vkCmdWriteAccelerationStructuresPropertiesKHR(cmd, info.queryCount, handles.data(), info.queryType, queries, 0);
    GEVulkanCommandLoader::endSingleTimeCommands(cmd);
    std::vector<uint64_t> sizes(handles.size());
    VkResult result = vkGetQueryPoolResults(device, queries, 0, info.queryCount,
        sizes.size() * sizeof(uint64_t), sizes.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    vkDestroyQueryPool(device, queries, NULL);
    CacheTransfer transfer(m_vk);
    if (result != VK_SUCCESS || !transfer.allocate(sizes, true))
        return;
    memset(transfer.mapped, 0, size_t(transfer.bytes));
    vmaFlushAllocation(m_vk->getVmaAllocator(), transfer.memory, transfer.padding, transfer.bytes);
    cmd = GEVulkanCommandLoader::beginSingleTimeCommands();
    for (size_t index = 0; index < handles.size(); ++index)
    {
        VkCopyAccelerationStructureToMemoryInfoKHR copy = {};
        copy.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_TO_MEMORY_INFO_KHR;
        copy.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_SERIALIZE_KHR;
        copy.src = handles[index];
        copy.dst.deviceAddress = transfer.address + transfer.offsets[index];
        vkCmdCopyAccelerationStructureToMemoryKHR(cmd, &copy);
    }
    cacheBarrier(cmd, true);
    GEVulkanCommandLoader::endSingleTimeCommands(cmd);
    vmaInvalidateAllocation(m_vk->getVmaAllocator(), transfer.memory, transfer.padding, transfer.bytes);
    uint64_t hash = CACHE_VERSION;
    for (size_t index = 0; index < sizes.size(); ++index)
        hash = hashBytes(transfer.mapped + transfer.offsets[index], size_t(sizes[index]), hash);
    const uint64_t header[] = { CACHE_VERSION, sizes.size(), hash };
    const std::string temp = path + "." + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::ofstream file(temp, std::ios::binary);
    if (!file)
        return;
    file.write(reinterpret_cast<const char*>(header), sizeof(header));
    file.write(reinterpret_cast<const char*>(sizes.data()), sizes.size() * sizeof(uint64_t));
    for (size_t index = 0; index < sizes.size(); ++index)
        file.write(reinterpret_cast<const char*>(transfer.mapped + transfer.offsets[index]), sizes[index]);
    file.close();
    if (file && std::rename(temp.c_str(), path.c_str()) == 0)
        GE::logMessage(GLL_INFO, "Saved %u prebuilt static BLAS", info.queryCount);
    else
        std::remove(temp.c_str());
}
}
