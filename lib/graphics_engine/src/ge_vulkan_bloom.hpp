#ifndef HEADER_GE_VULKAN_BLOOM_HPP
#define HEADER_GE_VULKAN_BLOOM_HPP

#include "ge_vulkan_post_processing.hpp"

namespace GE
{

// The areas (x0, y0, x1, y1, x1 and y1 are exclusive, all zero if there is
// nothing) which tonemap pass (deferred_convert_color.frag) of a viewport
// needs, it's pushed to it as it is
struct GEVulkanBloomRects
{
    // In the pixels of the FBO
    int32_t m_viewport[4];
    // Of getImageView() (the first level, half size)
    int32_t m_blur[4];
    // Of getWideImageView() (the widest of the blur which is always the same
    // level, eighth size)
    int32_t m_wide[4];
};

// Bloom of the finished hdr color, owned by GEVulkanDeferredFBOSplit (there is
// only one for all the viewports of it, so splitscreen shares the images and
// the pipeline). Unlike the glow outline and light scattering it needs the
// final hdr color (lit geometry, sky, fog, glow and scattering all together),
// so it runs between the lighting and the tonemap pass, all compute shaders so
// it doesn't need any graphics pipeline or render pass of its own. The
// bandwidth of memory is what it's made for (see "Bandwidth-Efficient
// Rendering", ARM, SIGGRAPH 2015):
//
// 1. Threshold and downscale: hdr is read once (at its full size) and the
//    bright part of it goes to the first level at half size (utils/blur.glsl,
//    BM_BLOOM_PREFILTER, a 13 tap filter with Karis weighting so one bright
//    pixel doesn't flicker), which is no more than a quarter of the pixels.
// 2. Dual filter downscale to half size of the previous level, up to 6 levels
//    (the smaller the viewport, the less of them, the width of the bloom is a
//    fixed fraction of the height of the viewport).
// 3. Dual filter upscale, from the smallest level up to the first one, each
//    one is mixed with what the level has already, so the result has both the
//    wide and the thin part of the bloom.
// 4. It's added to hdr right before the tonemap in the pass of tonemap itself
//    (deferred_convert_color.frag), so it's not another pass which reads and
//    writes the whole framebuffer, the result of the first level is sampled
//    by it. The same shader also adds the blend texture (lens dust) of
//    GEConfig::m_bloom_blend_texture, which is multiplied by the wide level
//    (the same as bloomblend.frag does with its 128 texture), the wide blur
//    is only one more tap there so it needs no pass of its own.
//
// The horizontal streak (optional, see STREAK_MIX in ge_vulkan_bloom.cpp, 0
// removes it and everything it costs) is the lens flare of the OpenGL bloom
// (lensblend.frag): the same bright part, only blurred in the horizontal
// direction and added to the bloom with the same weights. It's made from the
// levels 1, 2 and 3 (the sizes of the 512, 256 and 128 textures of OpenGL) when
// the down chain is done, and before the up chain changes them:
//
// a. Level 3 is one more downscale if the viewport has less than 4 levels.
// b. A horizontal blur of every one of them (BM_STREAK) to its own image.
// c. The same dual filter upscale and mix (UPSCALE_BLEND) as the bloom itself
//    to the first of them, so it has the same weights of the three.
// d. That one is upscaled and mixed to the first level of the bloom in place
//    (STREAK_MIX is how much of the final bloom is the streak), so the tonemap
//    pass doesn't know about it.
//
// The levels of every viewport are the pixels which are entirely inside of the
// rect of it (see GEVulkanPostProcessing::innerHalfRect), so no pixel is shared
// by two viewports: the dispatches of all viewports of a level run together,
// with one barrier between levels, not between viewports, and no sample ever
// reads outside of the rect of its viewport.
class GEVulkanBloom : public GEVulkanPostProcessing
{
public:
    // The first level is half the size of the FBO, every one is half of the
    // previous one
    static const unsigned MAX_LEVELS = 6;
    // The level which the blend texture (lens dust) is multiplied by, it's
    // the widest one of a viewport as high as REFERENCE_HEIGHT (eighth size)
    static const unsigned WIDE_LEVEL = 2;
    // The streak is made from the levels [1, STREAK_LEVELS]
    static const unsigned STREAK_LEVELS = 3;
private:
    std::array<GEVulkanAttachmentTexture*, MAX_LEVELS> m_levels;

    // The streak of the levels [i + 1] (the same sizes), all NULL if it's
    // disabled
    std::array<GEVulkanAttachmentTexture*, STREAK_LEVELS> m_streak;

    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    // 0: hdr -> [0], [1, MAX_LEVELS - 1]: [i - 1] -> [i] (downscale),
    // [MAX_LEVELS, 2 * MAX_LEVELS - 2]: [i - MAX_LEVELS + 1] ->
    // [i - MAX_LEVELS] (upscale, it blends with what's in it)
    std::array<VkDescriptorSet, 2 * MAX_LEVELS - 1> m_sets;

    // [0, STREAK_LEVELS - 1]: level [i + 1] -> m_streak[i] (horizontal blur),
    // [STREAK_LEVELS, 2 * STREAK_LEVELS - 2]: m_streak[i - STREAK_LEVELS + 1]
    // -> m_streak[i - STREAK_LEVELS] (upscale, blends), the last one:
    // m_streak[0] -> level 0 (upscale, blends)
    std::array<VkDescriptorSet, 2 * STREAK_LEVELS> m_streak_sets;

    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;

    // The viewports which have valid bloom of this frame, and the area of the
    // first level which it's in (x0, y0, x1, y1)
    std::vector<std::pair<const GEVulkanDrawCall*, GEVulkanBloomRects> >
        m_active;
    // ------------------------------------------------------------------------
    void destroy();
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* hdr);
public:
    // ------------------------------------------------------------------------
    // The hdr is the one of the lighting pass, which is in shader read only
    // layout after it
    GEVulkanBloom(GEVulkanDriver* vk,
                  const irr::core::dimension2d<irr::u32>& size,
                  GEVulkanAttachmentTexture* hdr);
    // ------------------------------------------------------------------------
    ~GEVulkanBloom();
    // ------------------------------------------------------------------------
    // Renders the bloom of the draw calls which are going to be tonemapped,
    // after the lighting render pass and before the tonemap one (must be
    // outside of render pass). Returns false if none of them has any, nothing
    // is done then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCameraSceneNode*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the tonemap pass in general layout
    VkImageView getImageView() const;
    // ------------------------------------------------------------------------
    // Same, the level of WIDE_LEVEL
    VkImageView getWideImageView() const;
    // ------------------------------------------------------------------------
    // The areas of the draw call which was rendered by the last render(), and
    // false if it has nothing, so the data of it is stale
    bool getRects(const GEVulkanDrawCall* dc, GEVulkanBloomRects* out) const;
};   // GEVulkanBloom

}

#endif
