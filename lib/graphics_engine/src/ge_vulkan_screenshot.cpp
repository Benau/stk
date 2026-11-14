#include "ge_vulkan_screenshot.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_command_loader.hpp"
#include "ge_vulkan_driver.hpp"

#include "IImage.h"

#include <cstring>

namespace GE
{
// ----------------------------------------------------------------------------
GEVulkanScreenShot::GEVulkanScreenShot(GEVulkanDriver* vk)
                  : m_vk(vk), m_finished(new Finished())
{
    m_in_flight.resize(GEVulkanDriver::getMaxFrameInFlight());
}   // GEVulkanScreenShot

// ----------------------------------------------------------------------------
GEVulkanScreenShot::~GEVulkanScreenShot()
{
    destroy();
}   // ~GEVulkanScreenShot

// ----------------------------------------------------------------------------
/** Saves the next frame which is presented to path (a png file), callback is
 *  called by the main thread with true if it was written. */
void GEVulkanScreenShot::add(const std::string& path,
                             std::function<void(bool)> callback)
{
    m_requests.push_back({ path, callback });
}   // add

// ----------------------------------------------------------------------------
/** Copies the swapchain image of this frame (in the present layout after the
 *  last render pass) into a host visible buffer, if something was requested.
 *  frame is the frame in flight of the command buffer. */
void GEVulkanScreenShot::record(VkCommandBuffer cmd, VkImage image,
                                unsigned frame)
{
    if (m_requests.empty())
        return;
    std::vector<Request> requests;
    std::swap(requests, m_requests);
    const auto fail = [this](std::vector<Request>& requests)
    {
        std::lock_guard<std::mutex> lock(m_finished->m_mutex);
        for (Request& r : requests)
            m_finished->m_results.emplace_back(r.m_callback, false);
    };
    // Some surfaces can't be the source of a copy (the swapchain is created
    // with it when they can)
    if (!m_vk->isSwapChainTransferSrc())
    {
        GE::logMessage(GLL_ERROR, "GEVulkanScreenShot: the swapchain images "
            "can't be copied");
        fail(requests);
        return;
    }

    Capture capture;
    VkExtent2D extent = m_vk->getSwapChainExtent();
    capture.m_width = extent.width;
    capture.m_height = extent.height;
    capture.m_format = m_vk->getSwapChainImageFormat();
    capture.m_transform = m_vk->getSurfaceTransform();
    // All the formats which are saved have 4 bytes per pixel
    VkDeviceSize size = (VkDeviceSize)capture.m_width * capture.m_height * 4;
    VmaAllocationCreateInfo alloc_info = {};
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    if (!m_vk->createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        alloc_info, capture.m_buffer, capture.m_memory))
    {
        GE::logMessage(GLL_ERROR, "GEVulkanScreenShot: createBuffer failed");
        fail(requests);
        return;
    }
    capture.m_requests = std::move(requests);

    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &barrier);

    VkBufferImageCopy region = {};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = { capture.m_width, capture.m_height, 1 };
    vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        capture.m_buffer, 1, &region);

    // Back to the layout of the present, and the copy is visible to the host
    // once the fence of the frame is signaled
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    VkMemoryBarrier host = {};
    host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0,
        1, &host, 0, NULL, 1, &barrier);
    m_in_flight[frame].push_back(std::move(capture));
}   // record

// ----------------------------------------------------------------------------
/** Called after the fence of frame was waited: its copies are complete, a
 *  thread saves them. The callbacks of the saved screenshots are called. */
void GEVulkanScreenShot::handleFrame(unsigned frame)
{
    for (Capture& capture : m_in_flight[frame])
    {
        if (!GEVulkanCommandLoader::multiThreadingEnabled())
        {
            save(m_vk, capture, m_finished);
            continue;
        }
        GEVulkanDriver* vk = m_vk;
        std::shared_ptr<Finished> finished = m_finished;
        std::shared_ptr<Capture> copied(new Capture(std::move(capture)));
        GEVulkanCommandLoader::addMultiThreadingCommand(
            [vk, copied, finished]()
            {
                save(vk, *copied, finished);
            });
    }
    m_in_flight[frame].clear();

    std::vector<std::pair<std::function<void(bool)>, bool> > results;
    {
        std::lock_guard<std::mutex> lock(m_finished->m_mutex);
        std::swap(results, m_finished->m_results);
    }
    for (auto& r : results)
    {
        if (r.first)
            r.first(r.second);
    }
}   // handleFrame

// ----------------------------------------------------------------------------
/** Converts the copy of a capture to an image (the orientation the user sees,
 *  without the pre-rotation of the swapchain), writes it to the files of its
 *  requests and destroys its buffer, on any thread. */
void GEVulkanScreenShot::save(GEVulkanDriver* vk, Capture& capture,
                              std::shared_ptr<Finished> finished)
{
    VmaAllocator allocator = vk->getVmaAllocator();
    VmaAllocationInfo info;
    vmaGetAllocationInfo(allocator, capture.m_memory, &info);
    vmaInvalidateAllocation(allocator, capture.m_memory, 0, VK_WHOLE_SIZE);
    const uint8_t* src = (const uint8_t*)info.pMappedData;

    // The byte order of the channels of the 8 bit formats, or the bit
    // offsets of the 10 bit ones
    int r = 0, g = 1, b = 2;
    bool packed_10_bit = false;
    bool supported = true;
    switch (capture.m_format)
    {
    case VK_FORMAT_B8G8R8A8_UNORM:
    case VK_FORMAT_B8G8R8A8_SRGB:
        r = 2;
        b = 0;
        break;
    case VK_FORMAT_R8G8B8A8_UNORM:
    case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_A8B8G8R8_UNORM_PACK32:
    case VK_FORMAT_A8B8G8R8_SRGB_PACK32:
        break;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
        packed_10_bit = true;
        r = 0;
        g = 10;
        b = 20;
        break;
    case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        packed_10_bit = true;
        r = 20;
        g = 10;
        b = 0;
        break;
    default:
        supported = false;
        break;
    }

    // With pre-rotation the swapchain image is rotated (see
    // GEVulkanDriver::getRotatedRect2D, a point of the screen at x, y is at
    // the same place of the image as below)
    const uint32_t w = capture.m_width;
    const uint32_t h = capture.m_height;
    const bool swap = capture.m_transform ==
        VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
        capture.m_transform == VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR;
    core::dimension2d<u32> size(swap ? h : w, swap ? w : h);
    video::IImage* image = NULL;
    if (supported)
        image = vk->createImage(video::ECF_R8G8B8, size);
    if (image)
    {
        uint8_t* dst = (uint8_t*)image->lock();
        for (uint32_t y = 0; y < size.Height; y++)
        {
            for (uint32_t x = 0; x < size.Width; x++)
            {
                uint32_t sx = x, sy = y;
                switch (capture.m_transform)
                {
                case VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR:
                    sx = w - 1 - y;
                    sy = x;
                    break;
                case VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR:
                    sx = w - 1 - x;
                    sy = h - 1 - y;
                    break;
                case VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR:
                    sx = y;
                    sy = h - 1 - x;
                    break;
                default:
                    break;
                }
                const uint8_t* p = src + ((size_t)sy * w + sx) * 4;
                uint8_t* q = dst + ((size_t)y * size.Width + x) * 3;
                if (packed_10_bit)
                {
                    uint32_t v;
                    memcpy(&v, p, 4);
                    q[0] = (uint8_t)(((v >> r) & 1023) >> 2);
                    q[1] = (uint8_t)(((v >> g) & 1023) >> 2);
                    q[2] = (uint8_t)(((v >> b) & 1023) >> 2);
                }
                else
                {
                    q[0] = p[r];
                    q[1] = p[g];
                    q[2] = p[b];
                }
            }
        }
        image->unlock();
    }
    else
    {
        GE::logMessage(GLL_ERROR, "GEVulkanScreenShot: swapchain format %d is "
            "not supported", (int)capture.m_format);
    }
    vmaDestroyBuffer(allocator, capture.m_buffer, capture.m_memory);

    std::vector<std::pair<std::function<void(bool)>, bool> > results;
    for (Request& request : capture.m_requests)
    {
        bool written = image &&
            vk->writeImageToFile(image, request.m_path.c_str(), 0);
        results.emplace_back(request.m_callback, written);
    }
    if (image)
        image->drop();
    std::lock_guard<std::mutex> lock(finished->m_mutex);
    for (auto& r : results)
        finished->m_results.push_back(std::move(r));
}   // save

// ----------------------------------------------------------------------------
/** At shutdown: the copies in flight, the requests
 *  which were not recorded and the callbacks are dropped. */
void GEVulkanScreenShot::destroy()
{
    bool in_flight = false;
    for (std::vector<Capture>& captures : m_in_flight)
        in_flight |= !captures.empty();
    if (in_flight)
        vkDeviceWaitIdle(m_vk->getDevice());
    for (std::vector<Capture>& captures : m_in_flight)
    {
        for (Capture& capture : captures)
        {
            capture.m_requests.clear();
            vmaDestroyBuffer(m_vk->getVmaAllocator(), capture.m_buffer,
                capture.m_memory);
        }
        captures.clear();
    }
    m_requests.clear();
    std::lock_guard<std::mutex> lock(m_finished->m_mutex);
    m_finished->m_results.clear();
}   // destroy

}
