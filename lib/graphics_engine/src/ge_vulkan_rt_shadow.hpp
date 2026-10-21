#ifndef HEADER_GE_VULKAN_RT_SHADOW_HPP
#define HEADER_GE_VULKAN_RT_SHADOW_HPP

#include "ge_vulkan_post_processing.hpp"

namespace GE
{
class GEVulkanArrayTexture;

// Ray traced shadows (see getRTShadowType()), owned by GEVulkanDeferredFBOSplit
// (only one for all the viewports of it, like GEVulkanSSAO). It's a compute
// shader between the g-buffer and the lighting render pass:
//
// rt_shadow.comp: the visibility of the sun (1.0 is lit) of every pixel, a
// ray from the surface to the sun traced in the ray traced scene
// (GEVulkanRTScene), the same query as the shadow of the hits of the
// reflection. The lighting pass samples it (layer 0) instead of the shadow
// map. Without a scene (the menus, before it's built), rt_shadow_clear.comp
// fills it with 1.0.
//
// The image has the size of the render target times getRTShadowScale() (one
// ray per texel), the lighting pass filters it to the full resolution without
// crossing the silhouettes or the viewports (utils/rt_shadow_filter.glsl).
//
// The image is an array so the layers after the sun can be the visibility of
// the point lights later, and the lighting shaders don't need to change. It
// is also created (with the size of the minimum, and nothing is rendered) if
// there is no ray tracing so the lighting pass has a valid descriptor
// of that type
class GEVulkanRTShadow : public GEVulkanPostProcessing
{
private:
    // Number of layers of the image, the sun is the first
    static const unsigned LAYERS = 1;

    // Size of the image relative to the render target, getRTShadowScale() when
    // it was created (so it's the same as the specialization constant of the
    // lighting pass, which is read from getScale()). 1.0 if it's not active
    float m_scale;

    // In general layout all the time (storage image and sampled)
    GEVulkanArrayTexture* m_mask;

    // The storage image view of the first mip level (the view of the texture
    // has all of them)
    VkImageView m_storage_view;

    // Binding 0 is the depth and 1 is the normal of the g-buffer (combined
    // image samplers), 2 is m_storage_view
    VkDescriptorSetLayout m_layout;

    VkDescriptorPool m_pool;

    VkDescriptorSet m_set;

    // Same definition as the data descriptor set layout of draw calls (camera
    // and the global light), see GEVulkanSSAO
    VkDescriptorSetLayout m_data_layout;

    // The sets are the one of GEVulkanRTScene (0), m_data_layout (1), m_layout
    // (2) and the mesh textures of the driver (3), they are all owned by
    // others except m_layout and m_data_layout
    VkPipelineLayout m_pipeline_layout;

    VkPipeline m_pipeline;

    VkPipeline m_clear_pipeline;
    // ------------------------------------------------------------------------
    void createDescriptors(GEVulkanAttachmentTexture* depth,
                           GEVulkanAttachmentTexture* normal);
    // ------------------------------------------------------------------------
    void createPipelines();
    // ------------------------------------------------------------------------
    void createPipeline(const char* shader, VkPipeline* pipeline) const;
    // ------------------------------------------------------------------------
    void destroy();
public:
    // ------------------------------------------------------------------------
    // The depth and the normal are the ones of the g-buffer, which are in read
    // only layouts after it. Nothing but the image is created if it's not
    // active (ray tracing is off or the device can't use it)
    GEVulkanRTShadow(GEVulkanDriver* vk,
                     const irr::core::dimension2d<irr::u32>& size,
                     GEVulkanAttachmentTexture* depth,
                     GEVulkanAttachmentTexture* normal, bool active);
    // ------------------------------------------------------------------------
    ~GEVulkanRTShadow();
    // ------------------------------------------------------------------------
    // Renders the visibility of all the viewports between the g-buffer and the
    // lighting render pass (must be outside of render pass). Returns false if
    // there is nothing to render (the sun is not ray traced), the image has no
    // valid content then
    virtual bool render(VkCommandBuffer cmd,
                        const std::vector<std::pair<GEVulkanDrawCall*,
                        GEVulkanCamera*> >& p);
    // ------------------------------------------------------------------------
    // Sampled by the lighting pass in general layout (an array)
    VkImageView getImageView() const;
    // ------------------------------------------------------------------------
    bool isActive() const              { return m_pipeline != VK_NULL_HANDLE; }
    // ------------------------------------------------------------------------
    // Size of the image relative to the render target. The image is the grid
    // of the whole render target (rt_shadow_grid.glsl), a viewport only writes
    // the texels which have their pixel inside of it
    float getScale() const                                 { return m_scale; }
};   // GEVulkanRTShadow

}

#endif
