#ifndef HEADER_GE_VULKAN_POST_PROCESS_HPP
#define HEADER_GE_VULKAN_POST_PROCESS_HPP

#include "vulkan_wrapper.h"

#include "dimension2d.h"

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

namespace GE
{
class GEVulkanAttachmentTexture;
class GEVulkanCameraSceneNode;
class GEVulkanDrawCall;
class GEVulkanDriver;

// Common Vulkan helpers for the effects of the split deferred FBO (between the
// g-buffer and the lighting pass, or the lighting and the tonemap pass). The
// derived classes own their actual images/pipelines
class GEVulkanPostProcessing
{
protected:
    GEVulkanDriver* m_vk;

    // Size of the g-buffer/FBO
    irr::core::dimension2d<irr::u32> m_size;

    // Reference height used by effects whose blur width scales with the
    // viewport height
    static const float REFERENCE_HEIGHT;

    // ------------------------------------------------------------------------
    // Modes of utils/blur.glsl (compute shaders blur_hdr.comp or
    // blur_unorm.comp, which every effect uses)
    enum BlurMode : uint32_t
    {
        // Dual filter, the destination is half the size of the source
        BM_DOWN = 0,
        // Dual filter, the destination is twice the size of the source
        BM_UP,
        // Same as BM_UP, but mixed with what the destination has already
        BM_UP_ADD,
        // One direction of a gaussian blur, the same size
        BM_GAUSSIAN,
        // Threshold and downscale of the finished hdr color for bloom
        BM_BLOOM_PREFILTER,
        // Horizontal gaussian blur (always, the direction is ignored) with
        // weights which add up to 1.0, the same size, for the bloom streak
        BM_STREAK,
    };
    // ------------------------------------------------------------------------
    // Push constants of utils/blur.glsl
    struct BlurPushConstants
    {
        // x0, y0, x1, y1 (exclusive) in pixels, source samples are clamped
        // inside
        int32_t m_src_rect[4];
        // Pixels written
        int32_t m_dst_rect[4];
        // (1, 0) = horizontal, (0, 1) = vertical, only for BM_GAUSSIAN
        int32_t m_direction[2];
        uint32_t m_mode;
        // Multiplies the distance of the samples
        float m_scale;
        // Only for BM_UP_ADD
        float m_blend;
    };
    // ------------------------------------------------------------------------
    // The area of the FBO which one draw call (viewport / camera) renders to
    struct Viewport
    {
        GEVulkanDrawCall* m_draw_call;
        // x0, y0, x1, y1 in pixels of the FBO (x1 and y1 are exclusive)
        std::array<int32_t, 4> m_rect;
        // Height of the viewport in pixels of the FBO, without rotation
        float m_height;
    };
    // ------------------------------------------------------------------------
    // Returns false if the area is empty (so it has nothing to be applied to).
    // The same viewport as GEVulkanDrawCall::prepareViewport, in pixels of the
    // FBO
    bool getViewport(GEVulkanDrawCall* dc, GEVulkanCameraSceneNode* cam,
                     Viewport* out) const;
    // ------------------------------------------------------------------------
    static irr::core::dimension2d<irr::u32> halfSize(
                                 const irr::core::dimension2d<irr::u32>& size);
    // ------------------------------------------------------------------------
    // Covers all pixels of a rect: floor of the start, ceil of the end.
    // x0, y0, x1, y1 where x1/y1 are exclusive
    static std::array<int32_t, 4> halfRect(const std::array<int32_t, 4>& rect);
    // ------------------------------------------------------------------------
    // Only the pixels of the half size image which are entirely inside of the
    // rect: ceil of the start, floor of the end (can be empty). Rects which
    // don't overlap (viewports) never overlap after it (halfRect can share a
    // pixel at an odd border), so the dispatches of all viewports of a level
    // can run without any barrier between them, even if they read what they
    // write
    static std::array<int32_t, 4> innerHalfRect(
                                          const std::array<int32_t, 4>& rect);
    // ------------------------------------------------------------------------
    static bool isEmptyRect(const std::array<int32_t, 4>& rect)
                      { return rect[2] <= rect[0] || rect[3] <= rect[1]; }
    // ------------------------------------------------------------------------
    // Transition all images from UNDEFINED to GENERAL. These images are
    // intended to remain in GENERAL for their entire lifetime
    void initializeGeneralImages(GEVulkanAttachmentTexture** images,
                                 unsigned count) const;
    // ------------------------------------------------------------------------
    // Standard layout used by the compute blur passes:
    //
    // binding 0 = combined image sampler
    // binding 1 = storage image
    //
    // Both are compute-only
    void createImageDescriptorSetLayout(VkDescriptorSetLayout* layout) const;
    // ------------------------------------------------------------------------
    // Create a pool containing exactly set_count sets of the standard
    // two-descriptor layout
    void createImageDescriptorPool(unsigned set_count,
                                   VkDescriptorPool* pool) const;
    // ------------------------------------------------------------------------
    // Allocate set_count copies of layout into sets
    void allocateImageDescriptorSets(VkDescriptorPool pool,
                                     VkDescriptorSetLayout layout,
                                     unsigned set_count,
                                     VkDescriptorSet* sets) const;
    // ------------------------------------------------------------------------
    // Fill one standard descriptor set
    //
    // input_view/input_sampler/input_layout describe binding 0
    // output_view/output_layout describe binding 1
    void writeImageDescriptorSet(VkDescriptorSet set,
                                 VkImageView input_view,
                                 VkSampler input_sampler,
                                 VkImageLayout input_layout,
                                 VkImageView output_view,
                                 VkImageLayout output_layout) const;
    // ------------------------------------------------------------------------
    // The compute pipeline of a blur shader (blur_hdr.comp or
    // blur_unorm.comp) which uses the standard descriptor layout and
    // BlurPushConstants
    void createBlurPipeline(const char* shader, VkDescriptorSetLayout layout,
                            VkPipelineLayout* pipeline_layout,
                            VkPipeline* pipeline) const;
    // ------------------------------------------------------------------------
    // Dispatches the bound blur pipeline for every pixel of dst_rect (it also
    // binds the descriptor set)
    void dispatchBlur(VkCommandBuffer cmd, VkPipelineLayout pipeline_layout,
                      VkDescriptorSet set, BlurMode mode,
                      const std::array<int32_t, 4>& src_rect,
                      const std::array<int32_t, 4>& dst_rect, float scale,
                      float blend = 0.0f, int32_t direction_x = 0,
                      int32_t direction_y = 0) const;
    // ------------------------------------------------------------------------
    // What the compute shaders of the last dispatches wrote is read by the next
    // dispatches, and also by the stages in extra_dst_stage
    static void computeToComputeBarrier(VkCommandBuffer cmd,
                                        VkPipelineStageFlags extra_dst_stage);

public:
    // ------------------------------------------------------------------------
    GEVulkanPostProcessing(GEVulkanDriver* vk,
      const irr::core::dimension2d<irr::u32>& size) : m_vk(vk), m_size(size) {}
    // ------------------------------------------------------------------------
    virtual ~GEVulkanPostProcessing()                                        {}
    // ------------------------------------------------------------------------
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCameraSceneNode*> >& p) = 0;

};   // GEVulkanPostProcessing

}

#endif
