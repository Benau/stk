#ifndef HEADER_GE_VULKAN_TARGET_GROUP_HPP
#define HEADER_GE_VULKAN_TARGET_GROUP_HPP

#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"

#include "SColor.h"

#include <deque>
#include <memory>
#include <utility>
#include <vector>

namespace GE
{
class GEVulkanFBOTexture;

// One drawAll() call: the draw call is generated (and its data is written) at
// the time of drawAll() and is rendered later in
// GEVulkanDriver::buildCommandBuffers() with the snapshot of the camera.
// Both are owned by value, the draw call is referenced by its shadow fbo and
// by the m_dcs list of the group, so an entry must never be moved or copied
// (GEVulkanTargetGroup::m_entries is a deque so it never relocates them)
struct GEVulkanDrawEntry
{
    GEVulkanDrawCall m_draw_call;
    GEVulkanCamera m_camera;
    // ------------------------------------------------------------------------
    GEVulkanDrawEntry() {}
    // ------------------------------------------------------------------------
    GEVulkanDrawEntry(const GEVulkanDrawEntry&) = delete;
    // ------------------------------------------------------------------------
    GEVulkanDrawEntry& operator=(const GEVulkanDrawEntry&) = delete;
};

// All drawAll() calls which have the same render target (the same
// GEVulkanDriver::getRTTTexture() at the time of calling drawAll), the draw
// calls of the entries are only used for this target, so the pipelines
// created by them always have the shape of this target. The entries are kept
// between frames, only the first m_used of them are used in the current frame
// (splitscreen needs more than 1)
struct GEVulkanTargetGroup
{
    // NULL if forward rendering without any render target
    GEVulkanFBOTexture* m_target;
    // Observer of m_target (not used if it's NULL), it's the user's
    // responsibility to not delete the target before endScene
    std::weak_ptr<bool> m_observer;
    irr::video::SColor m_clear_color;
    // A deque is used because GEVulkanDrawEntry can't be moved, and the
    // pointers in m_dcs (and m_current_entry of the scene manager) must stay
    // valid when more entries are added
    std::deque<GEVulkanDrawEntry> m_entries;
    // The first m_used entries of m_entries (draw call, camera), kept here so
    // it can be passed to the rendering code without creating a vector every
    // frame (the capacity is kept)
    std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCamera*> > m_dcs;
    unsigned m_used;
    unsigned m_unused_frames;
    // ------------------------------------------------------------------------
    GEVulkanTargetGroup() : m_target(NULL), m_used(0), m_unused_frames(0) {}
    // ------------------------------------------------------------------------
    GEVulkanDrawEntry& acquireEntry()
    {
        if (m_used == m_entries.size())
            m_entries.emplace_back();
        GEVulkanDrawEntry& e = m_entries[m_used];
        m_used = m_used + 1;
        m_dcs.emplace_back(&e.m_draw_call, &e.m_camera);
        return e;
    }
    // ------------------------------------------------------------------------
    // Only the last acquired entry can be released
    void releaseLastEntry()
    {
        m_used--;
        m_dcs.pop_back();
    }
    // ------------------------------------------------------------------------
    // The entries are kept (with their draw calls and pipelines)
    void resetUsed()
    {
        m_used = 0;
        m_dcs.clear();
    }
};

}

#endif
