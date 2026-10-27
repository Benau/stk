#ifndef HEADER_GE_VULKAN_FINAL_CHAIN_HPP
#define HEADER_GE_VULKAN_FINAL_CHAIN_HPP

#include "vulkan_wrapper.h"

#include "dimension2d.h"

#include <array>
#include <cstdint>
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

    // False after the chain is destroyed (the targets of the separate render
    // targets can live longer): nothing writes m_image of them anymore, so
    // the FBO is sampled directly again
    bool m_chain_alive;
    // ------------------------------------------------------------------------
    GEVulkanFinalPool()
    {
        m_device = VK_NULL_HANDLE;
        m_pool = VK_NULL_HANDLE;
        m_chain_alive = true;
    }
    // ------------------------------------------------------------------------
    ~GEVulkanFinalPool()
    {
        if (m_pool != VK_NULL_HANDLE)
            vkDestroyDescriptorPool(m_device, m_pool, NULL);
    }
};   // GEVulkanFinalPool

// What an FBO (GEVulkanFBOTexture) stores when the final chain is used: the
// two images which the stages write to (the image of the FBO itself is only
// read, by the first stage, so it never has to be a storage image: its format
// is bgra8 which not all the devices support as storage image), and the
// descriptor sets of them
class GEVulkanFinalTarget
{
public:
    // Same size as the FBO, rgba8, storage images (always in general layout
    // while the chain uses it). The stage k (of the ones which run for this
    // target, in order) writes m_images[k % 2] and reads the FBO if k is 0,
    // m_images[(k - 1) % 2] otherwise. m_images[1] is NULL (not created) if
    // only one stage runs
    std::array<GEVulkanAttachmentTexture*, 2> m_images;

    // Which of the stages of the chain run for this target (bit i = stage i),
    // the stages which need the depth don't run if the FBO doesn't keep it.
    // Set once, when the target is created
    uint32_t m_stage_mask;

    // How many bits of m_stage_mask are set, the last stage writes
    // m_images[(m_stage_count - 1) % 2]: it's the final image, in shader read
    // only layout after the chain, and what samples the FBO uses it
    // (GEVulkanFBOTexture::getImageView), so nothing is copied back to the
    // image of the FBO
    unsigned m_stage_count;

    // The sets of the layout of GEVulkanFinalChain::getIOLayout():
    // [0] = image of the FBO (sampled) -> m_images[0] (storage), for the first
    //       stage
    // [1] = m_images[0] (sampled) -> m_images[1] (storage), for the stages 1,
    //       3...
    // [2] = m_images[1] (sampled) -> m_images[0] (storage), for the stages 2,
    //       4... [1] and [2] are VK_NULL_HANDLE if there is one stage only
    // Binding 2 of all of them is the depth of the FBO (if it keeps it)
    std::array<VkDescriptorSet, 3> m_sets;

    std::shared_ptr<GEVulkanFinalPool> m_pool;
    // ------------------------------------------------------------------------
    GEVulkanFinalTarget()
    {
        m_images = {{ NULL, NULL }};
        m_stage_mask = 0;
        m_stage_count = 0;
        m_sets = {{ VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE }};
    }
    // ------------------------------------------------------------------------
    ~GEVulkanFinalTarget();
    // ------------------------------------------------------------------------
    // The image the last stage writes
    GEVulkanAttachmentTexture* getResultImage() const
    {
        return m_stage_count == 0 ? NULL :
            m_images[(m_stage_count - 1) % 2];
    }
    // ------------------------------------------------------------------------
    // True if what samples the FBO has to use getResultImage()
    bool isResultInImage() const
    {
        return getResultImage() && m_pool && m_pool->m_chain_alive;
    }
};   // GEVulkanFinalTarget

// One step of the chain, rendered with compute shaders after the FBO is
// rendered and before it is sampled by anything (the 2D renderer, or the
// users of a separate render target): the depth of field and the
// anti-aliasing now, and for example an upscaler later
class GEVulkanFinalStage
{
public:
    // ------------------------------------------------------------------------
    virtual ~GEVulkanFinalStage()                                            {}
    // ------------------------------------------------------------------------
    // True if the stage reads the depth of the FBO (binding 2 of io_set, in
    // shader read only layout), so it only runs for the FBO which keeps it
    // (GEVulkanFBOTexture::hasStoredDepth()), and only for the main one (not
    // the separate render targets, see GEVulkanFinalChain::initTarget)
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

// The last step of the rendering of a FBO (the main one, and the separate
// render targets which have a GEVulkanFinalTarget), after its last render pass:
//
// 1. The stages of GEConfig run in order: the depth of field
//    (GEVulkanDepthOfField, main FBO only), then the anti-aliasing
//    (GEVulkanFXAA or GEVulkanAntiAliasing), so the edges between the sharp
//    and the blurred areas are anti-aliased too. Both can be used at the same
//    time, or only one of them. One reads what the previous one wrote: the
//    first reads the FBO and writes GEVulkanFinalTarget::m_images[0], the
//    second reads it and writes m_images[1], the third reads that and writes
//    m_images[0]...
// 2. The final image is the one the last stage wrote, in shader read only
//    layout. GEVulkanFBOTexture::getImageView returns the view of it, so
//    everything which samples the FBO (the 2D renderer when it presents the
//    main one, and the users of a separate target) reads it directly: no copy
//    back to the image of the FBO
//
// To add a stage: implement GEVulkanFinalStage and add it in the constructor in
// the order it has to run
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
    // Gives the FBO the more images and the sets, after its createRTT(). It
    // does nothing (so render() does nothing for it, and the FBO is as it was)
    // if there is no stage which can run for it, or it's bigger than the
    // capacity. The stages which need the depth only run for the main FBO
    // (main_target), which keeps it (GEVulkanFBOTexture::hasStoredDepth()):
    // the separate render targets are what the GUI shows (the preview of a
    // kart for example), their distances are not the ones of a race
    void initTarget(GEVulkanFBOTexture* fbo, bool main_target);
    // ------------------------------------------------------------------------
    // After the last render pass of the FBO has ended (outside of render
    // pass), p are the draw calls (viewports) which were rendered to it
    void render(VkCommandBuffer cmd, GEVulkanFBOTexture* fbo,
                const std::vector<std::pair<GEVulkanDrawCall*,
                GEVulkanCamera*> >& p);
};   // GEVulkanFinalChain

}

#endif
