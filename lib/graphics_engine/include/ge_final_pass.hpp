#ifndef HEADER_GE_FINAL_PASS_HPP
#define HEADER_GE_FINAL_PASS_HPP

#include "vulkan_wrapper.h"

#include "dimension2d.h"

#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace GE
{
class GEVulkanCamera;
class GEVulkanDrawCall;
class GEVulkanDriver;

// One step of the final chain (see GEVulkanFinalChain), rendered with compute
// shaders after the FBO is rendered and before it is sampled by anything (the
// 2D renderer, or the users of a separate render target): the depth of field
// and the anti-aliasing of GE are two of them, and the game can add its own
// with addFinalPass()
class GEVulkanFinalStage
{
public:
    // ------------------------------------------------------------------------
    virtual ~GEVulkanFinalStage()                                            {}
    // ------------------------------------------------------------------------
    // True if the stage reads the depth of the FBO (binding 2 of io_set, in
    // shader read only layout), so it only runs for the FBO which keeps it
    // (GEVulkanFBOTexture::hasStoredDepth()), and only for the main one (not
    // the separate render targets, see GEVulkanFinalChain::initTarget). It has
    // to be the same as GEFinalPassInfo::m_needs_depth
    virtual bool needsDepth() const                           { return false; }
    // ------------------------------------------------------------------------
    // Outside of render pass. Reads binding 0 (and binding 2, the depth, if
    // needsDepth()) and writes binding 1 of io_set (set 0 of the pipeline
    // layout of the stage, which must be created with
    // GEVulkanFinalChain::getIOLayout()) and has to write all pixels of the
    // image, also the ones which are not in any viewport. The first one of
    // the chain is called with the layouts (shader read only for the sampled
    // image, general for the storage image) and the memory barriers for the
    // compute shaders of the chain already done, the chain does the barriers
    // between the stages and after the last one
    virtual void apply(VkCommandBuffer cmd, VkDescriptorSet io_set,
        const irr::core::dimension2d<irr::u32>& size,
        const std::vector<std::pair<GEVulkanDrawCall*,
        GEVulkanCamera*> >& p) = 0;
};   // GEVulkanFinalStage

// The order of the passes in the chain, lower runs first. The passes of the
// same order run in the order they were added. The depth of field is first, so
// the anti-aliasing (last) is applied on the edges of the blur too. The passes
// of the game are between them by default, they read what is displayed
// (tonemapped, display referred) and can't be anti-aliased by anything else
enum GEFinalPassOrder : int
{
    GFPO_DEPTH_OF_FIELD = 100,
    GFPO_DEFAULT = 500,
    GFPO_ANTI_ALIASING = 1000
};

// What the chain needs to know about a pass before the pass exists: the
// settings of the game decide if the scene is rendered to a FBO (instead of
// the swapchain image) and if the split deferred FBO (which keeps the depth)
// is used, and both are known before any stage is created
struct GEFinalPassInfo
{
    // For the log
    std::string m_name;

    // See GEFinalPassOrder
    int m_order;

    // True if the pass reads the depth of the FBO, which needs the split
    // deferred FBO (so the pbr shaders), the same as GEVulkanFinalStage::
    // needsDepth()
    bool m_needs_depth;

    // True (default) if the pass only runs for the main FBO of the game, not
    // for the separate render targets (the GUI shows them: the preview of a
    // kart for example)
    bool m_main_target_only;

    // Called when the chain is created, which is when the swapchain is
    // (re)created: false if the pass is not used. A change of what it returns
    // needs the swapchain to be created again, like the anti-aliasing and the
    // depth of field. NULL means always enabled. Passes need compute in the
    // main queue, if the device can't they are never enabled
    std::function<bool()> m_enabled;

    // Creates the stage when the chain is created (with the size of the
    // biggest FBO), the chain owns it. It can return NULL (nothing is added)
    // or throw (it's logged, the other passes are created anyway)
    std::function<GEVulkanFinalStage*(GEVulkanDriver* vk,
        VkDescriptorSetLayout io_layout,
        const irr::core::dimension2d<irr::u32>& capacity)> m_create;
    // ------------------------------------------------------------------------
    GEFinalPassInfo()
    {
        m_order = GFPO_DEFAULT;
        m_needs_depth = false;
        m_main_target_only = true;
    }
};   // GEFinalPassInfo

// Adds a pass to the final chain, returns the handle to remove it. It has to
// be called before the swapchain is created (or the game has to create it
// again, and call refreshDeferredSplit() if m_needs_depth is true) for it to
// be used. It doesn't create anything by itself
unsigned addFinalPass(const GEFinalPassInfo& info);
// Removes a pass added with addFinalPass(), it's not used anymore when the
// chain is created the next time
void removeFinalPass(unsigned handle);

}

#endif
