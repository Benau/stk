#ifndef HEADER_GE_VULKAN_DYNAMIC_BUFFER_HPP
#define HEADER_GE_VULKAN_DYNAMIC_BUFFER_HPP

#include "vulkan_wrapper.h"
#include "ge_vma.hpp"

#include <vector>
#include <memory>
#include <utility>

namespace GE
{

class GEVulkanDynamicBuffer
{
private:
    std::vector<VkBuffer> m_host_buffer, m_local_buffer;

    std::vector<VmaAllocation> m_host_memory, m_local_memory;

    std::vector<void*> m_mapped_addr;

    size_t m_size, m_real_size;

    std::shared_ptr<bool> m_buffer_observer;

    const VkBufferUsageFlags m_usage;

    const bool m_enable_host_transfer;

    static int m_supports_host_transfer;
    // ------------------------------------------------------------------------
    void initHostBuffer(unsigned frame, bool with_transfer);
    // ------------------------------------------------------------------------
    void initLocalBuffer(unsigned frame);
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    /** Destroys everything and creates all host and local buffers again with
     *  the given size, all contents are lost. */
    void recreate(size_t size);
public:
    // ------------------------------------------------------------------------
    GEVulkanDynamicBuffer(VkBufferUsageFlags usage, size_t initial_size,
                          unsigned host_buffer_size,
                          unsigned local_buffer_size,
                          bool enable_host_transfer = true);
    // ------------------------------------------------------------------------
    ~GEVulkanDynamicBuffer();
    // ------------------------------------------------------------------------
    bool setCurrentData(const std::vector<std::pair<void*, size_t> >& data,
                        VkCommandBuffer custom_cmd = VK_NULL_HANDLE,
                        unsigned cur_frame = -1);
    // ------------------------------------------------------------------------
    bool setCurrentData(void* data, size_t size,
                        VkCommandBuffer custom_cmd = VK_NULL_HANDLE,
                        unsigned cur_frame = -1)
            { return setCurrentData({{ data, size }}, custom_cmd, cur_frame); }
    // ------------------------------------------------------------------------
    VkBuffer getCurrentBuffer() const;
    // ------------------------------------------------------------------------
    bool resizeIfNeeded(size_t new_size);
    // ------------------------------------------------------------------------
    /** Same as resizeIfNeeded, but keeps the content of one host buffer.
     *  \param slot Index into the host buffers (not a frame number), it must
     *  be the index the caller is writing into right now, ie.
     *  getCurrentBufferIdx() for a buffer written through the (frames in
     *  flight + 1) host buffers directly, or getCurrentFrame() for a buffer
     *  filled by the default setCurrentData.
     *  \param keep_size Number of bytes to keep (clamped to the old size),
     *  the whole old buffer by default.
     *  The other host buffers and all local buffers are recreated empty, so
     *  this is only for buffers whose every slot is rewritten before it is
     *  used again by the gpu (the usual per-frame data). If there are local
     *  buffers, the kept data is also copied to the local buffer used by the
     *  current frame. */
    bool resizeKeepData(size_t new_size, unsigned slot,
                        size_t keep_size = (size_t)-1);
    // ------------------------------------------------------------------------
    size_t getSize() const                                   { return m_size; }
    // ------------------------------------------------------------------------
    size_t getRealSize() const                          { return m_real_size; }
    // ------------------------------------------------------------------------
    std::vector<VkBuffer>& getHostBuffer()            { return m_host_buffer; }
    // ------------------------------------------------------------------------
    std::vector<VkBuffer>& getLocalBuffer()          { return m_local_buffer; }
    // ------------------------------------------------------------------------
    std::vector<VmaAllocation>& getHostMemory()       { return m_host_memory; }
    // ------------------------------------------------------------------------
    std::vector<VmaAllocation>& getLocalMemory()     { return m_local_memory; }
    // ------------------------------------------------------------------------
    std::vector<void*>& getMappedAddr()               { return m_mapped_addr; }
    // ------------------------------------------------------------------------
    std::shared_ptr<bool> getBufferObserver() const
                                                  { return m_buffer_observer; }
    // ------------------------------------------------------------------------
    /** This can only be called after creating an instance with
     *  local_buffer_size == 0 first, which is always done in
     *  GEVulkan2dRenderer::createTrisBuffers. */
    static bool supportsHostTransfer()
                                      { return m_supports_host_transfer == 1; }

};   // GEVulkanDynamicBuffer

}

#endif
