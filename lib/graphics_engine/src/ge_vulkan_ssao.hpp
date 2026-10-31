#ifndef HEADER_GE_VULKAN_SSAO_HPP
#define HEADER_GE_VULKAN_SSAO_HPP

#include "ge_vulkan_post_processing.hpp"

namespace GE
{
class GEVulkanArrayTexture;

// Screen space ambient occlusion with VBAO (visibility bitmask ambient
// occlusion, see ssao.comp), owned by GEVulkanDeferredFBOSplit (only one for
// all the viewports of it, so splitscreen shares the images and the
// pipelines). It's all compute shaders between the g-buffer and the lighting
// render pass, in the same place as GEVulkanLightScatter:
//
// 1. ssao_depth.comp: the linear view space depth of the g-buffer (level 0 of
//    the depth chain) and the filtered 2x2 blocks of it (level 1), then
//    ssao_depth_down.comp for the next levels (up to level 3). The chain has
//    its own filter and values, it can't be shared with the HiZ depth (see
//    GEVulkanHiZDepth) which has the raw closest depth for tracing rays
// 2. ssao.comp: the visibility at half the size, using the depth chain and
//    the normal of the g-buffer (the same as solid_ssr.frag uses). The
//    estimator is a single function of the shader (VBAO), a different one
//    can replace it without changing anything else
// 3. ssao_blur.comp: horizontal and then vertical blur of it which doesn't
//    blur over the edges of the depth, its width is relative to the height of
//    the viewport (REFERENCE_HEIGHT like the other blurs) so it's resolution
//    independent
// 4. ssao_upsample.comp: the full size image (bilinear, but weighted by the
//    depth), which is sampled by the lighting pass (deferred_pbr.frag) for
//    the ambient light only, like the OpenGL renderer does
//
// It also has the indirect light of the same estimator (SSIL of the same
// paper, bound to the same switch), which needs the finished hdr as its light
// source so it's renderIndirect(), after the lighting pass:
//
// 1. ssil.comp: the indirect light at half the size (the light of the hdr,
//    the normal and the depth chain), with the visibility bitmasks of the
//    same slices as ssao.comp
// 2. ssil_blur.comp: horizontal and then vertical blur of it, the same way as
//    the occlusion
//
// deferred_convert_color.frag multiplies the blurred light by the albedo of
// the g-buffer and adds it to the hdr, before the tonemap
//
// Everything is done for each viewport, with its own camera and rect, and
// never reads outside of it
class GEVulkanSSAO : public GEVulkanPostProcessing
{
private:
    // Levels of the depth chain which are used (ssao.comp can fetch them all)
    static const unsigned DEPTH_LEVELS = 4;

    // Linear depth, level 0 has the size of the g-buffer. In general layout
    // all the time (storage image and sampled)
    GEVulkanArrayTexture* m_depth;

    // Storage image view of every level of m_depth
    std::array<VkImageView, DEPTH_LEVELS> m_depth_views;

    // Half size, in general layout all the time. 0 is the result of ssao.comp
    // and then the blur, 1 is the horizontal blur
    std::array<GEVulkanAttachmentTexture*, 2> m_ao_half;

    // Full size, the result which the lighting pass samples (general layout
    // all the time)
    GEVulkanAttachmentTexture* m_ao;

    // Half size indirect light (rgba16f), the same: 0 is the result of
    // ssil.comp and then the blur, 1 is the horizontal blur
    std::array<GEVulkanAttachmentTexture*, 2> m_ssil_half;

    // Binding 0, 1 and 4 are combined image samplers and 2 and 3 are storage
    // images (the unused ones are written with the same image as another one
    // so a descriptor is never left unwritten)
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    enum DescriptorSet : unsigned
    {
        // depth of the g-buffer -> level 0 and 1
        DS_DEPTH = 0,
        // level 1 -> 2, level 2 -> 3
        DS_DOWN_1,
        DS_DOWN_2,
        // depth chain and normal -> m_ao_half[0]
        DS_AO,
        // m_ao_half[0] -> [1] (horizontal), [1] -> [0] (vertical)
        DS_BLUR_H,
        DS_BLUR_V,
        // m_ao_half[0] -> m_ao
        DS_UPSAMPLE,
        // depth chain, normal and hdr -> m_ssil_half[0]
        DS_SSIL,
        // m_ssil_half[0] -> [1] (horizontal), [1] -> [0] (vertical)
        DS_SSIL_BLUR_H,
        DS_SSIL_BLUR_V,
        DS_COUNT
    };
    std::array<VkDescriptorSet, DS_COUNT> m_sets;

    // Which area every pass is dispatched for (in threads)
    enum Area : unsigned
    {
        // One thread for every 2x2 block of the viewport
        AREA_HALF,
        // One thread for every pixel of the viewport
        AREA_FULL,
        // One thread for every 2x2 block of the level in level (it writes the
        // next one)
        AREA_NEXT_LEVEL
    };

    // Same definition as the data descriptor set layout of draw calls (camera),
    // so their sets can be bound with m_pipeline_layout. It's owned here for
    // the same reason as in GEVulkanLightScatter
    VkDescriptorSetLayout m_data_layout;

    // Set 0 is m_layout and set 1 is m_data_layout
    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_depth_pipeline;

    VkPipeline m_down_pipeline;

    VkPipeline m_ao_pipeline;

    VkPipeline m_blur_pipeline;

    VkPipeline m_upsample_pipeline;

    VkPipeline m_ssil_pipeline;

    VkPipeline m_ssil_blur_pipeline;
    // ------------------------------------------------------------------------
    // The viewports which have something in the images (all of them share the
    // cameras of their draw calls)
    bool getViewports(const std::vector<std::pair<GEVulkanDrawCall*,
        GEVulkanCamera*> >& p, std::vector<Viewport>* out) const;
    // ------------------------------------------------------------------------
    // Binds the pipeline and the descriptor set, then dispatches it for every
    // viewport (with its camera, rect and push constants)
    void dispatch(VkCommandBuffer cmd, const std::vector<Viewport>& viewports,
                  VkPipeline pipeline, DescriptorSet set, Area area,
                  int32_t level, int32_t direction_x,
                  int32_t direction_y) const;
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth,
                           GEVulkanAttachmentTexture* normal,
                           GEVulkanAttachmentTexture* light);
    // ------------------------------------------------------------------------
    void createPipelines();
    // ------------------------------------------------------------------------
    void createPipeline(const char* shader, VkPipeline* pipeline) const;
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    // The depth and the normal are the ones of the g-buffer, which are in read
    // only layouts after it. The light is the hdr of the frame
    GEVulkanSSAO(GEVulkanDriver* vk,
                 const irr::core::dimension2d<irr::u32>& size,
                 GEVulkanAttachmentTexture* depth,
                 GEVulkanAttachmentTexture* normal,
                 GEVulkanAttachmentTexture* light);
    // ------------------------------------------------------------------------
    ~GEVulkanSSAO();
    // ------------------------------------------------------------------------
    // Renders the ambient occlusion of all the viewports between the g-buffer
    // and the lighting render pass (must be outside of render pass). Returns
    // false if there is nothing to render, the image has no valid content then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    // Renders the indirect light of all the viewports between the lighting
    // pass (which writes the hdr it uses) and the bloom of it (must be outside
    // of render pass). Returns false if there is nothing to render, the image
    // has no valid content then
    bool renderIndirect(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the lighting pass in general layout
    VkImageView getImageView() const;
    // ------------------------------------------------------------------------
    // Sampled by the tonemap pass in general layout
    VkImageView getSSILImageView() const;
};   // GEVulkanSSAO

}

#endif
