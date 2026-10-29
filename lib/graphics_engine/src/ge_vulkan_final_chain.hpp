#ifndef HEADER_GE_VULKAN_FINAL_CHAIN_HPP
#define HEADER_GE_VULKAN_FINAL_CHAIN_HPP

#include "ge_final_pass.hpp"

#include "vulkan_wrapper.h"

#include "dimension2d.h"

#include <array>
#include <cstdint>
#include <memory>
#include <string>
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

    // How many bits of m_stage_mask are set: the most stages which can run
    // for this target, so it decides how many images and sets it has
    unsigned m_stage_count;

    // The stages (bits of m_stage_mask) which run this frame, and how many of
    // them there are, set by GEVulkanFinalChain::prepare() every frame. The
    // last active stage writes m_images[(m_active_count - 1) % 2]: it's the
    // final image, in shader read only layout after the chain, and what
    // samples the FBO uses it (GEVulkanFBOTexture::getImageView), so nothing
    // is copied back to the image of the FBO. With no active stage (also
    // before the first prepare) the FBO is sampled directly
    uint32_t m_active_mask;

    unsigned m_active_count;

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
        m_active_mask = 0;
        m_active_count = 0;
        m_sets = {{ VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE }};
    }
    // ------------------------------------------------------------------------
    ~GEVulkanFinalTarget();
    // ------------------------------------------------------------------------
    // The image the last active stage writes, NULL if no stage is active
    GEVulkanAttachmentTexture* getResultImage() const
    {
        return m_active_count == 0 ? NULL :
            m_images[(m_active_count - 1) % 2];
    }
    // ------------------------------------------------------------------------
    // True if what samples the FBO has to use getResultImage()
    bool isResultInImage() const
    {
        return getResultImage() && m_pool && m_pool->m_chain_alive;
    }
};   // GEVulkanFinalTarget

// The passes which are enabled now (GEFinalPassInfo::m_enabled, and the
// device can run them), sorted by GEFinalPassOrder
std::vector<GEFinalPassInfo> getEnabledFinalPasses();

// The last step of the rendering of a FBO (the main one, and the separate
// render targets which have a GEVulkanFinalTarget), after its last render pass:
//
// 1. The enabled passes (see GEFinalPassInfo) run in the order of
//    GEFinalPassOrder: the depth of field (GEVulkanDepthOfField, main FBO
//    only), the passes of the game, then the anti-aliasing (GEVulkanFXAA or
//    GEVulkanAntiAliasing), so the edges between the sharp and the blurred
//    areas are anti-aliased too. Any of them can be used at the same time, or
//    none. One reads what the previous one wrote: the first reads the FBO
//    and writes GEVulkanFinalTarget::m_images[0], the second reads it and
//    writes m_images[1], the third reads that and writes m_images[0]...
// 2. The final image is the one the last stage wrote, in shader read only
//    layout. GEVulkanFBOTexture::getImageView returns the view of it, so
//    everything which samples the FBO (the 2D renderer when it presents the
//    main one, and the users of a separate target) reads it directly: no copy
//    back to the image of the FBO
//
// To add a stage: implement GEVulkanFinalStage and register it with
// addFinalPass() (ge_final_pass.hpp), the passes of GE are registered the same
// way
class GEVulkanFinalChain
{
private:
    struct Stage
    {
        GEVulkanFinalStage* m_stage;

        // See GEFinalPassInfo::m_main_target_only
        bool m_main_only;

        std::string m_name;
        // --------------------------------------------------------------------
        Stage() : m_stage(NULL), m_main_only(true)                           {}
    };

    GEVulkanDriver* m_vk;

    // Targets bigger than this have no GEVulkanFinalTarget (the working
    // buffers of the stages are this size)
    irr::core::dimension2d<irr::u32> m_capacity;

    // Binding 0 = combined image sampler (compute), 1 = storage image
    // (compute)
    VkDescriptorSetLayout m_io_layout;

    std::shared_ptr<GEVulkanFinalPool> m_pool;

    // In the order they run, at most 32 (GEVulkanFinalTarget::m_stage_mask)
    std::vector<Stage> m_stages;
    // ------------------------------------------------------------------------
    void createIOLayout();
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    // The stages of the enabled passes (getEnabledFinalPasses()) are created,
    // it has no stage (isEmpty()) if there is none or they can't be created.
    // capacity is the size of the biggest FBO
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
    // Decides which stages run for the FBO this frame (GEVulkanFinalStage::
    // isActive), it does nothing if the FBO has no GEVulkanFinalTarget. It has
    // to be called for every target which is rendered in the frame, with the
    // same draw calls as render(), before the 2D texture descriptors are
    // updated (GEVulkanDriver::endScene): the image which the users of the FBO
    // sample depends on it, and the descriptor update waits for the device when
    // an image view changed, which is only when a stage starts or stops being
    // active, not every frame
    void prepare(GEVulkanFBOTexture* fbo,
                 const std::vector<std::pair<GEVulkanDrawCall*,
                 GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    // After the last render pass of the FBO has ended (outside of render
    // pass), p are the draw calls (viewports) which were rendered to it
    void render(VkCommandBuffer cmd, GEVulkanFBOTexture* fbo,
                const std::vector<std::pair<GEVulkanDrawCall*,
                GEVulkanCamera*> >& p);
};   // GEVulkanFinalChain

}

#endif
