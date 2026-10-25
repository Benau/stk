#ifndef HEADER_GE_VULKAN_FINAL_CHAIN_HPP
#define HEADER_GE_VULKAN_FINAL_CHAIN_HPP

#include "vulkan_wrapper.h"

#include "dimension2d.h"

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace GE
{
class GEVulkanAttachmentTexture;
class GEVulkanCamera;
class GEVulkanDrawCall;
class GEVulkanDriver;
class GEVulkanFBOTexture;

// Destroys the descriptor pool of the final chain when the chain and all the
// targets which have a set of it are gone
struct GEVulkanFinalPool
{
    VkDevice m_device;

    VkDescriptorPool m_pool;
    // ------------------------------------------------------------------------
    GEVulkanFinalPool()
    {
        m_device = VK_NULL_HANDLE;
        m_pool = VK_NULL_HANDLE;
    }
    // ------------------------------------------------------------------------
    ~GEVulkanFinalPool()
    {
        if (m_pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(m_device, m_pool, NULL);
    }
};   // GEVulkanFinalPool

// What an FBO (GEVulkanFBOTexture) stores when the final chain is used: the
// one more image which the stages write to (the other image is the one of the
// FBO), and the descriptor sets of them
class GEVulkanFinalTarget
{
public:
    // Same size as the FBO, rgba8, storage image (always in general layout
    // while the chain uses it)
    GEVulkanAttachmentTexture* m_image;

    // The sets of the layout of GEVulkanFinalChain::getIOLayout():
    // [0] = image of the FBO (sampled) -> m_image (storage)
    // [1] = m_image (sampled) -> image of the FBO (storage). RESERVED for the
    // second stage of the chain (see GEVulkanFinalChain::render), it's
    // VK_NULL_HANDLE (not created) until the image of the FBO can be a
    // storage image, so only one stage can run now
    std::array<VkDescriptorSet, 2> m_sets;

    std::shared_ptr<GEVulkanFinalPool> m_pool;
    // ------------------------------------------------------------------------
    GEVulkanFinalTarget()
    {
        m_image = NULL;
        m_sets = {{ VK_NULL_HANDLE, VK_NULL_HANDLE }};
    }
    // ------------------------------------------------------------------------
    ~GEVulkanFinalTarget();
};   // GEVulkanFinalTarget

// One step of the chain, rendered with compute shaders after the FBO is
// rendered and before it is sampled by anything (the 2D renderer, or the
// users of a separate render target): anti-aliasing now, and for example
// depth of field, or an upscaler, later
class GEVulkanFinalStage
{
public:
    // ------------------------------------------------------------------------
    virtual ~GEVulkanFinalStage()                                            {}
    // ------------------------------------------------------------------------
    // Outside of render pass. Reads binding 0 and writes binding 1 of
    // io_set (set 0 of the pipeline layout of the stage, which must be
    // created with GEVulkanFinalChain::getIOLayout()) and has to write all
    // pixels of the image, also the ones which are not in any viewport. The
    // first one of the chain is called with the layouts (shader read only
    // for the sampled image, general for the storage image) and the memory
    // barriers for the compute shaders of the chain already done, the
    // chain does the barriers between the stages and after the last one
    virtual void apply(VkCommandBuffer cmd, VkDescriptorSet io_set,
        const irr::core::dimension2d<irr::u32>& size,
        const std::vector<std::pair<GEVulkanDrawCall*,
        GEVulkanCamera*> >& p) = 0;
};   // GEVulkanFinalStage

// The last step of the rendering of a FBO (the main one, and the separate
// render targets which have a GEVulkanFinalTarget), after its last render pass:
//
// 1. The stages of GEConfig (GEVulkanAntiAliasing for now) run in order, one
//    reads what the previous one wrote. They ping-pong between the image of
//    the FBO and GEVulkanFinalTarget::m_image: the first reads the FBO and
//    writes m_image, the second would read m_image and write the FBO
//    (GEVulkanFinalTarget::m_sets[1])...
// 2. If the last stage wrote m_image, it is blitted back to the image of the
//    FBO (vkCmdBlitImage converts rgba8 to the bgra8 of it), so everything
//    which samples the FBO (the 2D renderer when it presents the main one,
//    and the users of a separate target) gets the final image, in
//    shader read only layout like before
//
// To add a stage (DEPTH OF FIELD for example) that is not the last one:
// implement GEVulkanFinalStage, add it in the constructor in the order it has
// to run, and make m_sets[1] of the targets (initTarget) with the image of the
// FBO created with VK_IMAGE_USAGE_STORAGE_BIT (a rgba8 FBO image, or the stages
// can't write it) - the loop of render() already does the layout changes of
// the ping-pong. It's not tested because there is no second stage yet
class GEVulkanFinalChain
{
private:
    GEVulkanDriver* m_vk;

    // Targets bigger than this have no GEVulkanFinalTarget (the working
    // buffers of the stages are this size)
    irr::core::dimension2d<irr::u32> m_capacity;

    // Binding 0 = combined image sampler (compute), 1 = storage image
    // (compute)
    VkDescriptorSetLayout m_io_layout;

    std::shared_ptr<GEVulkanFinalPool> m_pool;

    std::vector<GEVulkanFinalStage*> m_stages;
    // ------------------------------------------------------------------------
    void createIOLayout();
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    // The stages are created with the setting of GEConfig, it has no stage
    // (isEmpty()) if there is none or it can't be created. capacity is the
    // size of the biggest FBO
    GEVulkanFinalChain(GEVulkanDriver* vk,
                       const irr::core::dimension2d<irr::u32>& capacity);
    // ------------------------------------------------------------------------
    ~GEVulkanFinalChain();
    // ------------------------------------------------------------------------
    bool isEmpty() const                           { return m_stages.empty(); }
    // ------------------------------------------------------------------------
    VkDescriptorSetLayout getIOLayout() const           { return m_io_layout; }
    // ------------------------------------------------------------------------
    const irr::core::dimension2d<irr::u32>& getCapacity() const
                                                         { return m_capacity; }
    // ------------------------------------------------------------------------
    // Gives the FBO the one more image and the sets, after its createRTT(). It
    // does nothing (so render() does nothing for it, and the FBO is as it was)
    // if there is no stage, or it's bigger than the capacity
    void initTarget(GEVulkanFBOTexture* fbo);
    // ------------------------------------------------------------------------
    // After the last render pass of the FBO has ended (outside of render
    // pass), p are the draw calls (viewports) which were rendered to it
    void render(VkCommandBuffer cmd, GEVulkanFBOTexture* fbo,
                const std::vector<std::pair<GEVulkanDrawCall*,
                GEVulkanCamera*> >& p);
};   // GEVulkanFinalChain

}

#endif
