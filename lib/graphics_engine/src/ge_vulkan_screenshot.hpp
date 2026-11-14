#ifndef HEADER_GE_VULKAN_SCREENSHOT_HPP
#define HEADER_GE_VULKAN_SCREENSHOT_HPP

#include "vulkan_wrapper.h"
#include "ge_vma.hpp"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace GE
{
class GEVulkanDriver;

// Screenshots of the swapchain without stalling: the image of the frame is
// copied into a host visible buffer at the end of its command buffer, and
// when the fence of that frame was waited (which the driver does anyway two
// frames later, see GEVulkanDriver::endScene) a thread of
// GEVulkanCommandLoader converts it and writes the png file. The callback of
// a screenshot is called by the main thread in a later frame.
class GEVulkanScreenShot
{
private:
    struct Request
    {
        std::string m_path;
        std::function<void(bool)> m_callback;
    };

    // The copy of a frame and the requests which it saves
    struct Capture
    {
        VkBuffer m_buffer;
        VmaAllocation m_memory;
        uint32_t m_width, m_height;
        VkFormat m_format;
        VkSurfaceTransformFlagBitsKHR m_transform;
        std::vector<Request> m_requests;
    };

    struct Finished
    {
        std::mutex m_mutex;
        std::vector<std::pair<std::function<void(bool)>, bool> > m_results;
    };

    GEVulkanDriver* m_vk;

    // Requested, not recorded yet
    std::vector<Request> m_requests;

    // Recorded in the command buffer of a frame in flight
    std::vector<std::vector<Capture> > m_in_flight;

    // Written (or failed) by a thread, their callbacks are called by the
    // main thread. Shared with the threads, which can outlive this
    std::shared_ptr<Finished> m_finished;

    // ------------------------------------------------------------------------
    static void save(GEVulkanDriver* vk, Capture& capture,
                     std::shared_ptr<Finished> finished);
public:
    // ------------------------------------------------------------------------
    GEVulkanScreenShot(GEVulkanDriver* vk);
    // ------------------------------------------------------------------------
    ~GEVulkanScreenShot();
    // ------------------------------------------------------------------------
    void add(const std::string& path, std::function<void(bool)> callback);
    // ------------------------------------------------------------------------
    void record(VkCommandBuffer cmd, VkImage image, unsigned frame);
    // ------------------------------------------------------------------------
    void handleFrame(unsigned frame);
    // ------------------------------------------------------------------------
    void destroy();
};   // GEVulkanScreenShot

}

#endif
