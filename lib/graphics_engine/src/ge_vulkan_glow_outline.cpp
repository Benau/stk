#include "ge_vulkan_glow_outline.hpp"

#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_main.hpp"
#include "ge_vulkan_shader_manager.hpp"

#include <algorithm>
#include <stdexcept>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
struct BlurPushConstants
{
    // x0, y0, x1, y1 (exclusive) in pixels, source samples are clamped inside
    int32_t m_src_rect[4];
    // Pixels written
    int32_t m_dst_rect[4];
    uint32_t m_upscale;
    // Multiplies the distance of the samples
    float m_scale;
};
}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanGlowOutline::GEVulkanGlowOutline(GEVulkanDriver* vk,
                              const irr::core::dimension2d<irr::u32>& size,
                              GEVulkanAttachmentTexture* depth)
                   : GEVulkanPostProcessing(vk, size),
                     m_render_pass(VK_NULL_HANDLE),
                     m_framebuffer(VK_NULL_HANDLE),
                     m_blur_layout(VK_NULL_HANDLE),
                     m_blur_pool(VK_NULL_HANDLE),
                     m_blur_pipeline_layout(VK_NULL_HANDLE),
                     m_blur_pipeline(VK_NULL_HANDLE)
{
    m_blur.fill(NULL);
    m_blur_sets.fill(VK_NULL_HANDLE);
    const irr::core::dimension2d<irr::u32> half = halfSize(size);
    const irr::core::dimension2d<irr::u32> quarter = halfSize(half);

    m_color = new GEVulkanAttachmentTexture(vk, size, VK_FORMAT_B8G8R8A8_UNORM,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    // Use A2B10G10R10 for blur targets since alpha is not needed and the
    // 10-bit channels provide better precision than 8-bit RGBA.
    VkFormat blur_format = VK_FORMAT_R8G8B8A8_UNORM;
    if (GEVulkanFeatures::supportsShaderStorageImageExtendedFormats())
        blur_format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    const VkImageUsageFlags blur_usage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    m_blur[0] = new GEVulkanAttachmentTexture(vk, half, blur_format,
        blur_usage, VK_IMAGE_ASPECT_COLOR_BIT);
    m_blur[1] = new GEVulkanAttachmentTexture(vk, quarter, blur_format,
        blur_usage, VK_IMAGE_ASPECT_COLOR_BIT);
    m_blur[2] = new GEVulkanAttachmentTexture(vk, half, blur_format,
        blur_usage, VK_IMAGE_ASPECT_COLOR_BIT);
    m_blur[3] = new GEVulkanAttachmentTexture(vk, halfSize(quarter),
        blur_format, blur_usage, VK_IMAGE_ASPECT_COLOR_BIT);

    // The blurred images stay in general layout forever
    initializeGeneralImages(m_blur.data(), m_blur.size());

    createRenderPass(depth);
    createBlur();
}   // GEVulkanGlowOutline

// ----------------------------------------------------------------------------
GEVulkanGlowOutline::~GEVulkanGlowOutline()
{
    VkDevice device = m_vk->getDevice();
    if (m_blur_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_blur_pipeline, NULL);
    if (m_blur_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_blur_pipeline_layout, NULL);
    // Descriptor sets are freed with their pools
    if (m_blur_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_blur_pool, NULL);
    if (m_blur_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_blur_layout, NULL);
    if (m_framebuffer != VK_NULL_HANDLE)
        vkDestroyFramebuffer(device, m_framebuffer, NULL);
    if (m_render_pass != VK_NULL_HANDLE)
        vkDestroyRenderPass(device, m_render_pass, NULL);
    for (GEVulkanAttachmentTexture* t : m_blur)
        delete t;
    delete m_color;
}   // ~GEVulkanGlowOutline

// ----------------------------------------------------------------------------
void GEVulkanGlowOutline::createRenderPass(GEVulkanAttachmentTexture* depth)
{
    std::array<VkAttachmentDescription, 2> desc = {};
    desc[0].format = VK_FORMAT_B8G8R8A8_UNORM;
    desc[0].samples = VK_SAMPLE_COUNT_1_BIT;
    desc[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    desc[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    desc[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    desc[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    desc[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    desc[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // The depth of the g-buffer, it's only tested here (and has to be stored
    // as lighting needs it)
    desc[1].format = depth->getInternalFormat();
    desc[1].samples = VK_SAMPLE_COUNT_1_BIT;
    desc[1].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    desc[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    desc[1].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    desc[1].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    desc[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    desc[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference color_reference =
        { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depth_reference =
        { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_reference;
    subpass.pDepthStencilAttachment = &depth_reference;

    std::array<VkSubpassDependency, 2> deps = {};
    // The depth written by the g-buffer, and the last frame which sampled the
    // glow images (the lighting pass and the blur)
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    // The glow color is sampled by the blur and the lighting pass
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = desc.size();
    render_pass_info.pAttachments = desc.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = deps.size();
    render_pass_info.pDependencies = deps.data();
    if (vkCreateRenderPass(m_vk->getDevice(), &render_pass_info, NULL,
        &m_render_pass) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateRenderPass failed for "
            "GEVulkanGlowOutline");
    }

    std::array<VkImageView, 2> views =
    {{
        (VkImageView)m_color->getTextureHandler(),
        (VkImageView)depth->getTextureHandler()
    }};
    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = m_render_pass;
    framebuffer_info.attachmentCount = views.size();
    framebuffer_info.pAttachments = views.data();
    framebuffer_info.width = m_size.Width;
    framebuffer_info.height = m_size.Height;
    framebuffer_info.layers = 1;
    if (vkCreateFramebuffer(m_vk->getDevice(), &framebuffer_info, NULL,
        &m_framebuffer) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateFramebuffer failed for "
            "GEVulkanGlowOutline");
    }
}   // createRenderPass

// ----------------------------------------------------------------------------
VkImageView GEVulkanGlowOutline::getColorImageView() const
{
    return (VkImageView)m_color->getTextureHandler();
}   // getColorImageView

// ----------------------------------------------------------------------------
VkImageView GEVulkanGlowOutline::getBlurImageView() const
{
    return (VkImageView)m_blur[2]->getTextureHandler();
}   // getBlurImageView

// ----------------------------------------------------------------------------
void GEVulkanGlowOutline::createBlur()
{
    VkDevice device = m_vk->getDevice();
    createImageDescriptorSetLayout(&m_blur_layout);
    createImageDescriptorPool(m_blur_sets.size(), &m_blur_pool);
    allocateImageDescriptorSets(
        m_blur_pool, m_blur_layout, m_blur_sets.size(),
        m_blur_sets.data());

    VkPushConstantRange push_constant = {};
    push_constant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant.offset = 0;
    push_constant.size = sizeof(BlurPushConstants);
    VkPipelineLayoutCreateInfo pipeline_layout_info = {};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &m_blur_layout;
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_constant;
    if (vkCreatePipelineLayout(device, &pipeline_layout_info, NULL,
        &m_blur_pipeline_layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreatePipelineLayout failed for blur in "
            "GEVulkanGlowOutline");
    }

    VkComputePipelineCreateInfo pipeline_info = {};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeline_info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeline_info.stage.module =
        GEVulkanShaderManager::getShader("glow_outline_blur.comp");
    pipeline_info.stage.pName = "main";
    pipeline_info.layout = m_blur_pipeline_layout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
        NULL, &m_blur_pipeline) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateComputePipelines failed for blur in "
            "GEVulkanGlowOutline");
    }

    // Index of m_blur, -1 is m_color
    const int inputs[5] = { -1, 0, 1, 1, 3 };
    const int outputs[5] = { 0, 1, 2, 3, 1 };
    for (unsigned i = 0; i < m_blur_sets.size(); i++)
    {
        VkImageView input_view;
        VkImageLayout input_layout;

        if (inputs[i] < 0)
        {
            input_view =
                (VkImageView)m_color->getTextureHandler();
            input_layout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        }
        else
        {
            input_view =
                (VkImageView)m_blur[inputs[i]]->getTextureHandler();
            input_layout = VK_IMAGE_LAYOUT_GENERAL;
        }

        writeImageDescriptorSet(m_blur_sets[i], input_view,
            m_vk->getSampler(GVS_SKYBOX), input_layout,
            (VkImageView)m_blur[outputs[i]]->getTextureHandler(),
            VK_IMAGE_LAYOUT_GENERAL);
    }
}   // createBlur

// ----------------------------------------------------------------------------
void GEVulkanGlowOutline::blur(VkCommandBuffer cmd,
                               const std::vector<BlurViewport>& viewports)
{
    // Size level of the input and output of every set (0 = full size, 1 =
    // half, 2 = quarter, 3 = eighth)
    static const unsigned src_level[5] = { 0, 1, 2, 2, 3 };
    static const unsigned dst_level[5] = { 1, 2, 1, 3, 2 };
    static const unsigned upscale[5] = { 0, 0, 1, 0, 1 };
    // Down to quarter size and up to half size, the reference
    static const unsigned shallow[3] = { 0, 1, 2 };
    // One more level down and up: with the double distance of samples it's
    // twice as wide as shallow, without skipping pixels
    static const unsigned deep[5] = { 0, 1, 3, 4, 2 };

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_blur_pipeline);
    // Every viewport has its own dispatches (and bounds, the pixels of a
    // viewport never read what's outside of it: other viewports, or the
    // pixels left by the last frame), the images are shared. The barriers are
    // between all dispatches so it also avoids writing the shared border
    // pixels of two viewports at the same time
    for (const BlurViewport& vp : viewports)
    {
        std::array<std::array<int32_t, 4>, 4> rects;
        rects[0] = vp.m_rect;
        for (unsigned i = 1; i < rects.size(); i++)
            rects[i] = halfRect(rects[i - 1]);

        const float k = vp.m_height / REFERENCE_HEIGHT;
        const bool use_deep = k > 1.5f;
        // Each level down doubles the width of the glow in pixels already
        const float scale = std::max(use_deep ? k * 0.5f : k, 0.25f);
        const unsigned* sequence = use_deep ? deep : shallow;
        const unsigned count = use_deep ? 5 : 3;
        for (unsigned j = 0; j < count; j++)
        {
            const unsigned set = sequence[j];
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                m_blur_pipeline_layout, 0, 1, &m_blur_sets[set], 0, NULL);
            const std::array<int32_t, 4>& src = rects[src_level[set]];
            const std::array<int32_t, 4>& dst = rects[dst_level[set]];
            BlurPushConstants pc = {};
            for (unsigned i = 0; i < 4; i++)
            {
                pc.m_src_rect[i] = src[i];
                pc.m_dst_rect[i] = dst[i];
            }
            pc.m_upscale = upscale[set];
            pc.m_scale = scale;
            vkCmdPushConstants(cmd, m_blur_pipeline_layout,
                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            const uint32_t w = (uint32_t)(dst[2] - dst[0]);
            const uint32_t h = (uint32_t)(dst[3] - dst[1]);
            vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);

            // The next dispatch reads what was written, the last one of a
            // viewport is sampled by the lighting pass. Writing it again next
            // frame is ordered by the dependency of the render pass
            VkMemoryBarrier barrier = {};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                (j == count - 1 ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : 0),
                0, 1, &barrier, 0, NULL, 0, NULL);
        }
    }
}   // blur

// ----------------------------------------------------------------------------
bool GEVulkanGlowOutline::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCameraSceneNode*> >& p)
{
    std::vector<std::pair<GEVulkanDrawCall*, GEVulkanCameraSceneNode*> > glow;
    for (auto& q : p)
    {
        if (q.first->hasGlowOutline())
            glow.push_back(q);
    }
    if (glow.empty())
        return false;

    VkClearValue clear = {};
    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = m_render_pass;
    render_pass_info.framebuffer = m_framebuffer;
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = { m_size.Width, m_size.Height };
    // The depth is loaded
    VkClearValue clears[2] = { clear, clear };
    render_pass_info.clearValueCount = 2;
    render_pass_info.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

    // Another pass could have bound other buffers
    bool rebind_base_vertex = true;
    std::vector<BlurViewport> viewports;
    // Same as GEVulkanDrawCall::getRenderViewport, before rotation
    const float scale = m_vk->getSeparateRTTTexture() ? 1.0f :
        getGEConfig()->m_render_scale;
    for (auto& q : glow)
    {
        q.first->prepareViewport(m_vk, q.second->getViewPort(), cmd);
        q.first->renderGlowOutline(m_vk, cmd, rebind_base_vertex);

        // The same viewport as prepareViewport, in pixels of the FBO
        const VkViewport vp = q.first->getRenderViewport(m_vk,
            q.second->getViewPort());
        const int32_t x0 = std::max(0, (int32_t)vp.x);
        const int32_t y0 = std::max(0, (int32_t)vp.y);
        const int32_t x1 = std::min((int32_t)m_size.Width,
            (int32_t)vp.x + (int32_t)vp.width);
        const int32_t y1 = std::min((int32_t)m_size.Height,
            (int32_t)vp.y + (int32_t)vp.height);
        if (x1 > x0 && y1 > y0)
        {
            viewports.push_back({ {{ x0, y0, x1, y1 }},
                q.second->getViewPort().getHeight() * scale });
        }
    }
    vkCmdEndRenderPass(cmd);

    blur(cmd, viewports);
    return true;
}   // render

}
