#include "ge_vulkan_deferred_fbo_split.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_hiz_depth.hpp"

#include <array>
#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
VkAttachmentDescription makeAttachment(VkFormat format,
                                       VkAttachmentLoadOp load_op,
                                       VkAttachmentStoreOp store_op,
                                       VkImageLayout initial_layout,
                                       VkImageLayout final_layout)
{
    VkAttachmentDescription desc = {};
    desc.format = format;
    desc.samples = VK_SAMPLE_COUNT_1_BIT;
    desc.loadOp = load_op;
    desc.storeOp = store_op;
    desc.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    desc.initialLayout = initial_layout;
    desc.finalLayout = final_layout;
    return desc;
}   // makeAttachment

// ----------------------------------------------------------------------------
// Dependencies between render passes are not framebuffer-local (no
// VK_DEPENDENCY_BY_REGION_BIT), the next pass is a different render pass
VkSubpassDependency makeExternalDependency(VkPipelineStageFlags src_stage,
                                           VkPipelineStageFlags dst_stage,
                                           VkAccessFlags src_access,
                                           VkAccessFlags dst_access,
                                           bool from_external = true)
{
    VkSubpassDependency dependency = {};
    dependency.srcSubpass = from_external ? VK_SUBPASS_EXTERNAL : 0;
    dependency.dstSubpass = from_external ? 0 : VK_SUBPASS_EXTERNAL;
    dependency.srcStageMask = src_stage;
    dependency.dstStageMask = dst_stage;
    dependency.srcAccessMask = src_access;
    dependency.dstAccessMask = dst_access;
    dependency.dependencyFlags = 0;
    return dependency;
}   // makeExternalDependency

// ----------------------------------------------------------------------------
VkRenderPass createRenderPass(VkDevice device,
                              const std::vector<VkAttachmentDescription>& desc,
                              const VkSubpassDescription& subpass,
                              const std::vector<VkSubpassDependency>& deps,
                              const char* name)
{
    VkRenderPassCreateInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_pass_info.attachmentCount = desc.size();
    render_pass_info.pAttachments = desc.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = deps.size();
    render_pass_info.pDependencies = deps.data();

    VkRenderPass render_pass = VK_NULL_HANDLE;
    if (vkCreateRenderPass(device, &render_pass_info, NULL,
        &render_pass) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkCreateRenderPass failed for ") +
            name + " in GEVulkanDeferredFBOSplit");
    }
    return render_pass;
}   // createRenderPass

// ----------------------------------------------------------------------------
VkFramebuffer createFramebuffer(VkDevice device, VkRenderPass render_pass,
                                const std::vector<VkImageView>& views,
                                const core::dimension2d<u32>& size,
                                const char* name)
{
    VkFramebufferCreateInfo framebuffer_info = {};
    framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebuffer_info.renderPass = render_pass;
    framebuffer_info.attachmentCount = views.size();
    framebuffer_info.pAttachments = views.data();
    framebuffer_info.width = size.Width;
    framebuffer_info.height = size.Height;
    framebuffer_info.layers = 1;

    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    if (vkCreateFramebuffer(device, &framebuffer_info, NULL,
        &framebuffer) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkCreateFramebuffer failed for ")
            + name + " in GEVulkanDeferredFBOSplit");
    }
    return framebuffer;
}   // createFramebuffer

// ----------------------------------------------------------------------------
// Allocates one descriptor set with count combined image samplers (binding 0
// to count - 1) in the given slot
void createSamplerDescriptor(GEVulkanDriver* vk, unsigned count,
                             VkDescriptorSetLayout* layout,
                             VkDescriptorPool* pool, VkDescriptorSet* set,
                             const VkDescriptorImageInfo* image_infos,
                             const char* name)
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(count);
    for (unsigned i = 0; i < count; i++)
    {
        bindings[i] = {};
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].pImmutableSamplers = NULL;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo setinfo = {};
    setinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setinfo.pBindings = bindings.data();
    setinfo.bindingCount = bindings.size();
    if (vkCreateDescriptorSetLayout(vk->getDevice(), &setinfo, NULL,
        layout) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkCreateDescriptorSetLayout "
            "failed for ") + name + " in GEVulkanDeferredFBOSplit");
    }

    VkDescriptorPoolSize pool_size;
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = count;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(vk->getDevice(), &pool_info, NULL,
        pool) != VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkCreateDescriptorPool failed "
            "for ") + name + " in GEVulkanDeferredFBOSplit");
    }

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = *pool;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = layout;
    if (vkAllocateDescriptorSets(vk->getDevice(), &alloc_info, set) !=
        VK_SUCCESS)
    {
        throw std::runtime_error(std::string("vkAllocateDescriptorSets failed "
            "for ") + name + " in GEVulkanDeferredFBOSplit");
    }

    VkWriteDescriptorSet write_descriptor_set = {};
    write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set.dstBinding = 0;
    write_descriptor_set.dstArrayElement = 0;
    write_descriptor_set.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write_descriptor_set.descriptorCount = count;
    write_descriptor_set.dstSet = *set;
    write_descriptor_set.pImageInfo = image_infos;
    vkUpdateDescriptorSets(vk->getDevice(), 1, &write_descriptor_set, 0, NULL);
}   // createSamplerDescriptor
}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanDeferredFBOSplit::GEVulkanDeferredFBOSplit(GEVulkanDriver* vk,
                                          const core::dimension2d<u32>& size,
                                          bool swapchain_output)
                        : GEVulkanDeferredFBO(vk, size, swapchain_output)
{
    // GEVulkanDeferredFBO creates attachments with sampled usage and no input
    // attachment descriptors if m_deferred_split is true, which is expected
    // to be true if this class is used (see GEVulkanDriver)
    assert(getGEConfig()->m_deferred_split);
    initSplitGBufferDescriptor(vk);
    initSplitConvertColorDescriptor(vk);
}   // GEVulkanDeferredFBOSplit

// ----------------------------------------------------------------------------
void GEVulkanDeferredFBOSplit::initSplitGBufferDescriptor(GEVulkanDriver* vk)
{
    // Same bindings as the input attachments in deferred_pbr.frag and
    // deferred_pointlight.frag: 0 = color, 1 = normal (with pbr data), 2 = depth
    std::array<VkDescriptorImageInfo, 3> image_infos = {};
    image_infos[0].sampler = vk->getSampler(GVS_NEAREST);
    image_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_infos[0].imageView =
        (VkImageView)m_attachments[GVDFT_COLOR]->getTextureHandler();
    image_infos[1].sampler = vk->getSampler(GVS_NEAREST);
    image_infos[1].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_infos[1].imageView =
        (VkImageView)m_attachments[GVDFT_NORMAL]->getTextureHandler();
    image_infos[2].sampler = vk->getSampler(GVS_NEAREST);
    // The lighting pass also uses the depth as its read only depth attachment
    image_infos[2].imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    image_infos[2].imageView =
        (VkImageView)m_depth_texture->getTextureHandler();
    createSamplerDescriptor(vk, image_infos.size(),
        &m_descriptor_layout[GVDFP_HDR], &m_descriptor_pool[GVDFP_HDR],
        &m_descriptor_set[GVDFP_HDR], image_infos.data(), "GVDFP_HDR");
}   // initSplitGBufferDescriptor

// ----------------------------------------------------------------------------
void GEVulkanDeferredFBOSplit::initSplitConvertColorDescriptor(
                                                            GEVulkanDriver* vk)
{
    std::array<VkDescriptorImageInfo, 1> image_infos = {};
    image_infos[0].sampler = vk->getSampler(GVS_NEAREST);
    image_infos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    image_infos[0].imageView =
        (VkImageView)m_attachments[GVDFT_HDR]->getTextureHandler();
    createSamplerDescriptor(vk, image_infos.size(),
        &m_descriptor_layout[GVDFP_CONVERT_COLOR],
        &m_descriptor_pool[GVDFP_CONVERT_COLOR],
        &m_descriptor_set[GVDFP_CONVERT_COLOR], image_infos.data(),
        "GVDFP_CONVERT_COLOR");
}   // initSplitConvertColorDescriptor

// ----------------------------------------------------------------------------
void GEVulkanDeferredFBOSplit::createRTT()
{
    if (!useSwapChainOutput())
        createOutputImage();

    VkDevice device = m_vk->getDevice();
    const bool has_displace = getAttachment<GVDFT_DISPLACE_COLOR>() != NULL;
    // Without displace the tonemap pass is the last pass, and it renders to
    // the swapchain image directly (so 2D can keep drawing in it)
    const bool swapchain_last = useSwapChainOutput() && !has_displace;
    const VkFormat color_format =
        m_attachments[GVDFT_COLOR]->getInternalFormat();
    const VkFormat normal_format =
        m_attachments[GVDFT_NORMAL]->getInternalFormat();
    const VkFormat hdr_format = m_attachments[GVDFT_HDR]->getInternalFormat();
    const VkFormat depth_format = m_depth_texture->getInternalFormat();

    m_rtt_render_pass.resize(GVDSP_TONEMAP + 1, VK_NULL_HANDLE);
    m_rtt_frame_buffer.resize(GVDSP_TONEMAP, VK_NULL_HANDLE);

    // GVDSP_GBUFFER: color + normal (mixed with pbr data) + depth
    {
        std::vector<VkAttachmentDescription> desc =
        {
            makeAttachment(color_format, VK_ATTACHMENT_LOAD_OP_CLEAR,
                VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            makeAttachment(normal_format, VK_ATTACHMENT_LOAD_OP_CLEAR,
                VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            // Sampled and used as read only depth attachment by lighting
            makeAttachment(depth_format, VK_ATTACHMENT_LOAD_OP_CLEAR,
                VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        };
        std::array<VkAttachmentReference, 2> color_references =
            {{
                { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL },
                { 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL }
            }};
        VkAttachmentReference depth_reference =
            { 2, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = color_references.size();
        subpass.pColorAttachments = color_references.data();
        subpass.pDepthStencilAttachment = &depth_reference;

        // Wait for whatever read or wrote these images last frame (sampled
        // by lighting / tonemap / displace, depth read by hiz generation)
        std::vector<VkSubpassDependency> deps =
        {
            makeExternalDependency(
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
        };
        m_rtt_render_pass[GVDSP_GBUFFER] = createRenderPass(device, desc,
            subpass, deps, "GVDSP_GBUFFER");

        m_rtt_frame_buffer[GVDSP_GBUFFER] = createFramebuffer(device,
            m_rtt_render_pass[GVDSP_GBUFFER],
            {
                (VkImageView)m_attachments[GVDFT_COLOR]->getTextureHandler(),
                (VkImageView)m_attachments[GVDFT_NORMAL]->getTextureHandler(),
                (VkImageView)m_depth_texture->getTextureHandler()
            }, getSize(), "GVDSP_GBUFFER");
    }

    // GVDSP_LIGHTING: hdr, depth is only read (depth test of point lights and
    // skybox, and the depth sampled in shader)
    {
        std::vector<VkAttachmentDescription> desc =
        {
            makeAttachment(hdr_format, VK_ATTACHMENT_LOAD_OP_CLEAR,
                VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            makeAttachment(depth_format, VK_ATTACHMENT_LOAD_OP_LOAD,
                VK_ATTACHMENT_STORE_OP_STORE,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        };
        VkAttachmentReference color_reference =
            { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkAttachmentReference depth_reference =
            { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL };
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_reference;
        subpass.pDepthStencilAttachment = &depth_reference;

        // g-buffer written -> sampled, and hdr was sampled by tonemap last
        // frame
        std::vector<VkSubpassDependency> deps =
        {
            makeExternalDependency(
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT)
        };
        m_rtt_render_pass[GVDSP_LIGHTING] = createRenderPass(device, desc,
            subpass, deps, "GVDSP_LIGHTING");

        m_rtt_frame_buffer[GVDSP_LIGHTING] = createFramebuffer(device,
            m_rtt_render_pass[GVDSP_LIGHTING],
            {
                (VkImageView)m_attachments[GVDFT_HDR]->getTextureHandler(),
                (VkImageView)m_depth_texture->getTextureHandler()
            }, getSize(), "GVDSP_LIGHTING");
    }

    // GVDSP_TONEMAP: output (swapchain format if it's the swapchain image,
    // the displace color texture if there is displace, or the texture of this
    // FBO), depth is writable again for ghost / transparent
    {
        const VkFormat output_format = swapchain_last ?
            m_vk->getSwapChainImageFormat() : has_displace ?
            m_attachments[GVDFT_DISPLACE_COLOR]->getInternalFormat() :
            getInternalFormat();
        std::vector<VkAttachmentDescription> desc =
        {
            makeAttachment(output_format, VK_ATTACHMENT_LOAD_OP_CLEAR,
                VK_ATTACHMENT_STORE_OP_STORE, VK_IMAGE_LAYOUT_UNDEFINED,
                swapchain_last ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR :
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            // Displace passes after this one load the depth again
            makeAttachment(depth_format, VK_ATTACHMENT_LOAD_OP_LOAD,
                has_displace ? VK_ATTACHMENT_STORE_OP_STORE :
                VK_ATTACHMENT_STORE_OP_DONT_CARE,
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                has_displace ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL :
                VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        };
        VkAttachmentReference color_reference =
            { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
        VkAttachmentReference depth_reference =
            { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_reference;
        subpass.pDepthStencilAttachment = &depth_reference;

        // hdr written -> sampled, the depth was only read by lighting
        std::vector<VkSubpassDependency> deps =
        {
            makeExternalDependency(
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
        };
        if (swapchain_last)
        {
            // Wait for the swapchain image to be acquired
            deps.push_back(makeExternalDependency(
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT));
        }
        if (has_displace)
        {
            // The displace color texture, hdr and depth are sampled by the
            // displace passes and hiz generation (compute) afterwards
            deps.push_back(makeExternalDependency(
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                VK_ACCESS_SHADER_READ_BIT, false/*from_external*/));
        }
        m_rtt_render_pass[GVDSP_TONEMAP] = createRenderPass(device, desc,
            subpass, deps, "GVDSP_TONEMAP");

        // One framebuffer per swapchain image if it's the last pass writing
        // to it, see GEVulkanDeferredSplitPass
        auto& sciv = m_vk->getSwapChainImageViews();
        const unsigned count = swapchain_last ? sciv.size() : 1;
        for (unsigned i = 0; i < count; i++)
        {
            VkImageView output_view;
            if (swapchain_last)
                output_view = sciv[i];
            else if (has_displace)
            {
                output_view = (VkImageView)
                    getAttachment<GVDFT_DISPLACE_COLOR>()->getTextureHandler();
            }
            else
                output_view = (VkImageView)getTextureHandler();
            m_rtt_frame_buffer.push_back(createFramebuffer(device,
                m_rtt_render_pass[GVDSP_TONEMAP],
                {
                    output_view,
                    (VkImageView)m_depth_texture->getTextureHandler()
                }, getSize(), "GVDSP_TONEMAP"));
        }
    }

    // Always rendered if there is displace support in this FBO, which is
    // decided when it's created, same as GEVulkanDeferredFBO
    if (has_displace)
        createDisplacePasses(GVDSP_DISPLACE_MASK, GVDSP_DISPLACE_COLOR);
}   // createRTT

// ----------------------------------------------------------------------------
void GEVulkanDeferredFBOSplit::beginPass(VkCommandBuffer cmd, unsigned pass,
                                         unsigned framebuffer,
                                         uint32_t clear_count,
                                         const VkClearValue* clears) const
{
    VkRenderPassBeginInfo render_pass_info = {};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_info.renderPass = getRTTRenderPass(pass);
    render_pass_info.framebuffer = getRTTFramebuffer(framebuffer);
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = { getSize().Width, getSize().Height };
    render_pass_info.clearValueCount = clear_count;
    render_pass_info.pClearValues = clears;
    vkCmdBeginRenderPass(cmd, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);
}   // beginPass

// ----------------------------------------------------------------------------
void GEVulkanDeferredFBOSplit::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCameraSceneNode*> >& p)
{
    bool rebind_base_vertex = true;
    const bool bind_mesh_textures =
        GEVulkanFeatures::supportsBindMeshTexturesAtOnce();
    bool multiple_viewports = p.size() > 1;
    const bool has_displace_fbo = getAttachment<GVDFT_DISPLACE_COLOR>() != NULL;
    const bool swapchain_last = useSwapChainOutput() && !has_displace_fbo;
    const unsigned image_index = m_vk->getCurrentImageIndex();

    // 1. g-buffer
    {
        std::array<VkClearValue, 3> clears = {};
        clears[2].depthStencil = {0.0f, 0};
        beginPass(cmd, GVDSP_GBUFFER, GVDSP_GBUFFER, clears.size(),
            clears.data());
        for (auto& q : p)
        {
            if (bind_mesh_textures)
                q.first->bindAllMaterials(cmd);
            else
                rebind_base_vertex = true;
            q.first->updateDataDescriptorSets(m_vk, q.second);
            q.first->prepareViewport(m_vk, q.second->getViewPort(), cmd);
            if (q.first->doDepthOnlyRenderingFirst())
            {
                q.first->renderPipeline(m_vk, cmd, GVPT_DEPTH,
                    rebind_base_vertex);
            }
            q.first->renderPipeline(m_vk, cmd, GVPT_SOLID, rebind_base_vertex);
            m_vk->getPrimitivesDrawn() += q.first->getPolyCount();
        }
        vkCmdEndRenderPass(cmd);
    }

    // 2. lighting, the background is the clear color of hdr
    {
        video::SColorf cf(m_vk->getSeparateRTTTexture() == this ?
            m_vk->getRTTClearColor() : m_vk->getClearColor());
        VkClearValue clear = {};
        clear.color = { cf.getRed(), cf.getGreen(), cf.getBlue(), cf.getAlpha() };
        beginPass(cmd, GVDSP_LIGHTING, GVDSP_LIGHTING, 1, &clear);
        for (auto& q : p)
        {
            if (multiple_viewports)
                q.first->prepareViewport(m_vk, q.second->getViewPort(), cmd);
            q.first->renderDeferredLighting(m_vk, cmd);
            q.first->renderSkyBox(m_vk, cmd);
        }
        vkCmdEndRenderPass(cmd);
    }

    // 3. tonemap, then ghost and transparent materials
    {
        VkClearValue clear = {};
        beginPass(cmd, GVDSP_TONEMAP,
            swapchain_last ? GVDSP_TONEMAP + image_index : GVDSP_TONEMAP, 1,
            &clear);
        for (auto& q : p)
        {
            if (multiple_viewports)
                q.first->prepareViewport(m_vk, q.second->getViewPort(), cmd);
            q.first->renderDeferredConvertColor(m_vk, cmd);
            if (bind_mesh_textures)
                q.first->bindAllMaterials(cmd);
            else
                rebind_base_vertex = true;
            q.first->renderPipeline(m_vk, cmd, GVPT_GHOST_DEPTH,
                rebind_base_vertex);
            q.first->renderPipeline(m_vk, cmd, GVPT_TRANSPARENT,
                rebind_base_vertex);
        }
    }
    // The last pass is left open
    if (!has_displace_fbo)
        return;
    vkCmdEndRenderPass(cmd);

    // 4. displace, the mask is skipped if no material uses it, but the color
    // pass is always rendered (a copy of the tonemap output then)
    bool has_displace = false;
    for (auto& q : p)
    {
        if (q.first->hasShaderForRendering("displace") ||
            q.first->hasShaderForRendering("displace_skinning"))
        {
            has_displace = true;
            break;
        }
    }
    std::array<VkClearValue, GVDFT_COUNT> zeros = {};
    if (has_displace)
    {
        for (auto& q : p)
        {
            GEVulkanHiZDepth* hiz = q.first->getHiZDepth();
            if (hiz)
                hiz->generate(cmd);
        }
        beginPass(cmd, GVDSP_DISPLACE_MASK, GVDSP_DISPLACE_MASK,
            getZeroClearCountForPass(GVDFP_DISPLACE_MASK), zeros.data());
        for (auto& q : p)
        {
            if (multiple_viewports)
                q.first->prepareViewport(m_vk, q.second->getViewPort(), cmd);
            if (bind_mesh_textures)
                q.first->bindAllMaterials(cmd);
            else
                rebind_base_vertex = true;
            q.first->renderPipeline(m_vk, cmd, GVPT_DISPLACE_MASK,
                rebind_base_vertex);
        }
        vkCmdEndRenderPass(cmd);
    }
    beginPass(cmd, GVDSP_DISPLACE_COLOR, useSwapChainOutput() ?
        GVDSP_DISPLACE_COLOR + image_index : GVDSP_DISPLACE_COLOR,
        getZeroClearCountForPass(GVDFP_DISPLACE_COLOR), zeros.data());
    for (auto& q : p)
    {
        if (multiple_viewports)
            q.first->prepareViewport(m_vk, q.second->getViewPort(), cmd);
        q.first->renderDisplaceColor(m_vk, cmd, has_displace);
        if (has_displace)
        {
            if (bind_mesh_textures)
                q.first->bindAllMaterials(cmd);
            else
                rebind_base_vertex = true;
            q.first->renderPipeline(m_vk, cmd, GVPT_DISPLACE_COLOR,
                rebind_base_vertex);
        }
    }
}   // render

// ----------------------------------------------------------------------------
VkRenderPass GEVulkanDeferredFBOSplit::getRenderPassForPipeline(
                                               unsigned pipeline_type) const
{
    switch (pipeline_type)
    {
    case GVPT_DEFERRED_LIGHTING:
    case GVPT_SKYBOX:
        return getRTTRenderPass(GVDSP_LIGHTING);
    case GVPT_DEFERRED_CONVERT_COLOR:
    case GVPT_GHOST_DEPTH:
    case GVPT_TRANSPARENT:
        return getRTTRenderPass(GVDSP_TONEMAP);
    case GVPT_DISPLACE_MASK:
        return getRTTRenderPass(GVDSP_DISPLACE_MASK);
    case GVPT_DISPLACE_COLOR:
        return getRTTRenderPass(GVDSP_DISPLACE_COLOR);
    case GVPT_DEPTH:
    case GVPT_SOLID:
    default:
        return getRTTRenderPass(GVDSP_GBUFFER);
    }
}   // getRenderPassForPipeline

}
