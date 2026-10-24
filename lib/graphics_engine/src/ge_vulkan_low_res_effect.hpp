#ifndef HEADER_GE_VULKAN_LOW_RES_EFFECT_HPP
#define HEADER_GE_VULKAN_LOW_RES_EFFECT_HPP

#include "ge_vulkan_post_processing.hpp"

namespace GE
{

// Base of the effects which are marched by a compute shader in a smaller
// image, blurred, and then added to hdr by the lighting pass (see
// lighting_composite.frag), so they need no graphics pipeline, render pass or
// mesh of their own: GEVulkanLightScatter and GEVulkanGodRays. They are owned
// by GEVulkanDeferredFBOSplit (there is only one for all the viewports of it,
// so splitscreen shares the images and the pipelines) and they are rendered
// between the g-buffer and the lighting render pass:
//
// 1. For each viewport (draw call) which has the effect, at a half or a quarter
//    of the size of it, the compute shader of the derived class marches every
//    pixel, using the camera data of the draw call and the depth of the
//    g-buffer, with the push constants of pushConstants() (the rect of the
//    viewport must always be in them).
// 2. blur_hdr.comp (the gaussian mode of utils/blur.glsl) blurs it
//    horizontally and then vertically, never reading outside of the rect of
//    the viewport.
// 3. The lighting pass samples the result (in general layout).
//
// What the derived classes have: the shader, the size of the push constants,
// the downscale and the blur width (the effect of the viewport, see Params),
// which draw calls have the effect, and the push constants of a viewport
class GEVulkanLowResEffect : public GEVulkanPostProcessing
{
public:
    struct Params
    {
        // For the error messages
        const char* m_name;

        // Compute shader of the march, it has the depth at set 0 binding 0,
        // the image at set 0 binding 1 (writeonly, r11f_g11f_b10f if
        // SHADER_STORAGE_IMAGE_EXTENDED_FORMATS is defined, rgba16f if not)
        // and the data descriptor set of draw calls at set 1
        const char* m_shader;

        // Size of the push constants of the shader (compute stage)
        uint32_t m_push_constant_size;

        // Size of the image is the size of the FBO shifted right by this: 1 is
        // half and 2 is a quarter
        unsigned m_shift;

        // The distance of the taps of the blur (in pixels of the image) is
        // the height of the viewport (render scale and splitscreen included)
        // divided by REFERENCE_HEIGHT, multiplied by the factor and clamped
        // by min and max, so the width of the blur is the same fraction of
        // the viewport with any size of screen. The image is smaller by
        // 2 ^ shift so use a factor of 2 ^ (1 - shift) to have the width of
        // the half size effects
        float m_blur_scale_factor;
        float m_blur_scale_min;
        float m_blur_scale_max;
    };

private:
    Params m_params;

    // Smaller size, in general layout all the time (storage image and
    // sampled). 0 is the result, 1 is the horizontal blur
    std::array<GEVulkanAttachmentTexture*, 2> m_images;

    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    // 0: depth -> [0], 1: [0] -> [1] (horizontal), 2: [1] -> [0] (vertical)
    std::array<VkDescriptorSet, 3> m_sets;

    // Only BM_GAUSSIAN
    BlurPipelines m_blur_pipelines;

    // Same definition as the data descriptor set layout of draw calls (camera
    // and lights), so their sets can be bound with m_pipeline_layout. It's
    // owned here and not borrowed from a draw call, because draw calls (and
    // their layouts) are destroyed and created again (for example when
    // settings is updated) while this lives as long as the FBO, and a pipeline
    // layout can't be used after the set layout it was created with is gone
    VkDescriptorSetLayout m_data_layout;

    // Set 0 is m_layout and set 1 is m_data_layout
    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth);
    // ------------------------------------------------------------------------
    void createBlurPipeline();
    // ------------------------------------------------------------------------
    void createPipeline();
    // ------------------------------------------------------------------------
    void destroy();
protected:
    // ------------------------------------------------------------------------
    // True if the draw call (viewport) has the effect this frame
    virtual bool hasEffect(const GEVulkanDrawCall* dc) const = 0;
    // ------------------------------------------------------------------------
    // Push constants (with getPipelineLayout, compute stage and offset 0, the
    // size of Params::m_push_constant_size) before the dispatch of the
    // march of a viewport
    virtual void pushConstants(VkCommandBuffer cmd, const Viewport& vp) = 0;
    // ------------------------------------------------------------------------
    VkPipelineLayout getPipelineLayout() const    { return m_pipeline_layout; }
    // ------------------------------------------------------------------------
    // Size of the image of an effect of the size, same as halfSize with 1
    static irr::core::dimension2d<irr::u32> downscaledSize(
        const irr::core::dimension2d<irr::u32>& size, unsigned shift);
    // ------------------------------------------------------------------------
    // Covers all pixels of a rect in the image, floor of the start and ceil of
    // the end, same as halfRect with 1
    static std::array<int32_t, 4> downscaledRect(
        const std::array<int32_t, 4>& rect, unsigned shift);
public:
    // ------------------------------------------------------------------------
    // The depth is the one of the g-buffer, which is in read only depth layout
    // after it
    GEVulkanLowResEffect(GEVulkanDriver* vk,
                         const irr::core::dimension2d<irr::u32>& size,
                         GEVulkanAttachmentTexture* depth,
                         const Params& params);
    // ------------------------------------------------------------------------
    virtual ~GEVulkanLowResEffect();
    // ------------------------------------------------------------------------
    // Renders and blurs the effect of draw calls which have any, between the
    // g-buffer and the lighting render pass (must be outside of render pass).
    // Returns false if none of them has so nothing is done, the lighting pass
    // must skip it then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the lighting pass in general layout
    VkImageView getImageView() const;
};   // GEVulkanLowResEffect

}

#endif
