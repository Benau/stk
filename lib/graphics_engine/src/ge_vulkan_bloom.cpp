#include "ge_vulkan_bloom.hpp"

#include "ge_vulkan_attachment_texture.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_draw_call.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"

#include <algorithm>
#include <cmath>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
// How much of the upscaled smaller level is in the next bigger one: higher is
// a wider bloom, lower is a thinner one. The final weight of level i of n is
// (1 - b) * b^i (the last one has b^(n - 1)), so 0.42 with 3 levels is
// 0.58 / 0.24 / 0.18, which is the 0.5 / 0.25 / 0.125 (normalized) of the
// 512 / 256 / 128 textures of the OpenGL bloom (bloomblend.frag)
const float UPSCALE_BLEND = 0.42f;
// Levels of a viewport as high as REFERENCE_HEIGHT, 3 are as wide as the three
// textures of the OpenGL bloom
const int REFERENCE_LEVELS = 3;
// Width of the whole bloom (and of the area which the threshold averages)
// relative to the height of the viewport, 1.0 is the one which a viewport as
// high as REFERENCE_HEIGHT has with the distance 1.0 between the taps of the
// first 3 levels. It's the same at any resolution and render scale (the
// distance of the taps and the number of levels are made from it), lower is
// thinner and with more peak
const float BLOOM_WIDTH = 0.6f;
// How much of the final bloom is the horizontal streak, 0 disables it (and all
// of its passes and images, the bloom is then the same as without the streak).
// OpenGL adds the same amount of both (bloomblend.frag and lensblend.frag) so
// 0.5 with the same total as OpenGL (BLOOM_INTENSITY of
// deferred_convert_color.frag, which is for both of them together)
const float STREAK_MIX = 0.5f;
}   // anonymous namespace

// ----------------------------------------------------------------------------
GEVulkanBloom::GEVulkanBloom(GEVulkanDriver* vk,
                             const irr::core::dimension2d<irr::u32>& size,
                             GEVulkanAttachmentTexture* hdr)
             : GEVulkanPostProcessing(vk, size), m_layout(VK_NULL_HANDLE),
               m_pool(VK_NULL_HANDLE), m_pipeline_layout(VK_NULL_HANDLE),
               m_pipeline(VK_NULL_HANDLE)
{
    m_sets.fill(VK_NULL_HANDLE);
    m_levels.fill(NULL);
    m_streak.fill(NULL);
    m_streak_sets.fill(VK_NULL_HANDLE);
    try
    {
        // Same as the light scattering: no alpha is needed and the values are
        // never negative
        VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT;
        if (GEVulkanFeatures::supportsShaderStorageImageExtendedFormats())
            format = VK_FORMAT_B10G11R11_UFLOAT_PACK32;
        const VkImageUsageFlags usage =
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        irr::core::dimension2d<irr::u32> level_size = size;
        for (GEVulkanAttachmentTexture*& t : m_levels)
        {
            level_size = halfSize(level_size);
            t = new GEVulkanAttachmentTexture(vk, level_size, format, usage,
                VK_IMAGE_ASPECT_COLOR_BIT);
        }

        // The images stay in general layout forever
        initializeGeneralImages(m_levels.data(), m_levels.size());
        if (STREAK_MIX > 0.0f)
        {
            for (unsigned i = 0; i < STREAK_LEVELS; i++)
            {
                m_streak[i] = new GEVulkanAttachmentTexture(vk,
                    m_levels[i + 1]->getSize(), format, usage,
                    VK_IMAGE_ASPECT_COLOR_BIT);
            }
            initializeGeneralImages(m_streak.data(), m_streak.size());
        }

        createDescriptors(hdr);
        createBlurPipeline("blur_hdr.comp", m_layout, &m_pipeline_layout,
            &m_pipeline);
    }
    catch (...)
    {
        destroy();
        throw;
    }
}   // GEVulkanBloom

// ----------------------------------------------------------------------------
GEVulkanBloom::~GEVulkanBloom()
{
    destroy();
}   // ~GEVulkanBloom

// ----------------------------------------------------------------------------
void GEVulkanBloom::destroy()
{
    VkDevice device = m_vk->getDevice();
    if (m_pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(device, m_pipeline, NULL);
    if (m_pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(device, m_pipeline_layout, NULL);
    // Descriptor sets are freed with their pool
    if (m_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(device, m_pool, NULL);
    if (m_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(device, m_layout, NULL);
    for (GEVulkanAttachmentTexture* t : m_levels)
        delete t;
    for (GEVulkanAttachmentTexture* t : m_streak)
        delete t;
}   // destroy

// ----------------------------------------------------------------------------
VkImageView GEVulkanBloom::getImageView() const
{
    return (VkImageView)m_levels[0]->getTextureHandler();
}   // getImageView

// ----------------------------------------------------------------------------
VkImageView GEVulkanBloom::getWideImageView() const
{
    return (VkImageView)m_levels[WIDE_LEVEL]->getTextureHandler();
}   // getWideImageView

// ----------------------------------------------------------------------------
bool GEVulkanBloom::getRects(const GEVulkanDrawCall* dc,
                             GEVulkanBloomRects* out) const
{
    for (auto& p : m_active)
    {
        if (p.first == dc)
        {
            *out = p.second;
            return true;
        }
    }
    return false;
}   // getRects

// ----------------------------------------------------------------------------
void GEVulkanBloom::createDescriptors(GEVulkanAttachmentTexture* hdr)
{
    const bool streak = m_streak[0] != NULL;
    createImageDescriptorSetLayout(&m_layout);
    createImageDescriptorPool(m_sets.size() +
        (streak ? m_streak_sets.size() : 0), &m_pool);
    allocateImageDescriptorSets(m_pool, m_layout, m_sets.size(),
        m_sets.data());
    if (streak)
    {
        allocateImageDescriptorSets(m_pool, m_layout, m_streak_sets.size(),
            m_streak_sets.data());
    }

    // Linear for the filters of the blur
    VkSampler sampler = m_vk->getSampler(GVS_SKYBOX);
    // The finished hdr color, and then the level to the first one
    writeImageDescriptorSet(m_sets[0], (VkImageView)hdr->getTextureHandler(),
        sampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        (VkImageView)m_levels[0]->getTextureHandler(),
        VK_IMAGE_LAYOUT_GENERAL);
    for (unsigned i = 1; i < MAX_LEVELS; i++)
    {
        // Downscale
        writeImageDescriptorSet(m_sets[i],
            (VkImageView)m_levels[i - 1]->getTextureHandler(), sampler,
            VK_IMAGE_LAYOUT_GENERAL,
            (VkImageView)m_levels[i]->getTextureHandler(),
            VK_IMAGE_LAYOUT_GENERAL);
        // Upscale
        writeImageDescriptorSet(m_sets[MAX_LEVELS + i - 1],
            (VkImageView)m_levels[i]->getTextureHandler(), sampler,
            VK_IMAGE_LAYOUT_GENERAL,
            (VkImageView)m_levels[i - 1]->getTextureHandler(),
            VK_IMAGE_LAYOUT_GENERAL);
    }
    if (!streak)
        return;
    for (unsigned i = 0; i < STREAK_LEVELS; i++)
    {
        // Horizontal blur
        writeImageDescriptorSet(m_streak_sets[i],
            (VkImageView)m_levels[i + 1]->getTextureHandler(), sampler,
            VK_IMAGE_LAYOUT_GENERAL,
            (VkImageView)m_streak[i]->getTextureHandler(),
            VK_IMAGE_LAYOUT_GENERAL);
        if (i + 1 < STREAK_LEVELS)
        {
            // Upscale
            writeImageDescriptorSet(m_streak_sets[STREAK_LEVELS + i],
                (VkImageView)m_streak[i + 1]->getTextureHandler(), sampler,
                VK_IMAGE_LAYOUT_GENERAL,
                (VkImageView)m_streak[i]->getTextureHandler(),
                VK_IMAGE_LAYOUT_GENERAL);
        }
    }
    // To the first level
    writeImageDescriptorSet(m_streak_sets[2 * STREAK_LEVELS - 1],
        (VkImageView)m_streak[0]->getTextureHandler(), sampler,
        VK_IMAGE_LAYOUT_GENERAL,
        (VkImageView)m_levels[0]->getTextureHandler(),
        VK_IMAGE_LAYOUT_GENERAL);
}   // createDescriptors

// ----------------------------------------------------------------------------
bool GEVulkanBloom::render(VkCommandBuffer cmd,
    const std::vector<std::pair<GEVulkanDrawCall*,
    GEVulkanCameraSceneNode*> >& p)
{
    m_active.clear();

    struct Item
    {
        Viewport m_viewport;
        // The number of levels this viewport uses
        unsigned m_levels;
        // [0] is the rect of the viewport, [i + 1] is the rect of level i
        std::array<std::array<int32_t, 4>, MAX_LEVELS + 1> m_rects;
        // How many of the levels [1, STREAK_LEVELS] have a streak (0 if
        // disabled), and if the last one needs a downscale of its own
        unsigned m_streak_levels = 0;
        bool m_streak_extra_down = false;
        // Distance of the taps of the horizontal blur
        float m_streak_scale = 1.0f;
        // Distance of the taps of the threshold and downscale, and of the
        // dual filters of the levels (1.0 is the one of the original filters)
        float m_prefilter_scale = 1.0f;
        float m_blur_scale = 1.0f;
    };
    std::vector<Item> items;
    unsigned max_levels = 0;
    for (auto& q : p)
    {
        // Same as the tonemap pass, which is the one using the bloom
        if (!q.first->hasDeferredOutput())
            continue;
        Item item;
        if (!getViewport(q.first, q.second, &item.m_viewport))
            continue;
        item.m_rects[0] = item.m_viewport.m_rect;
        for (unsigned i = 1; i < item.m_rects.size(); i++)
            item.m_rects[i] = innerHalfRect(item.m_rects[i - 1]);
        // A viewport which is too small to have a single pixel at half size
        if (isEmptyRect(item.m_rects[1]))
            continue;

        // The bloom is as wide as a fraction of the height of the viewport,
        // each level more is twice as wide
        // Every level is twice as wide as the previous one, so the number of
        // them only gets as close as that (a power of 2) and the distance of
        // the taps is the rest, like the glow outline and the light
        // scattering do, so the width is proportional to the height of the
        // viewport and doesn't jump or stay the same between the limits of
        // the levels
        const float k = item.m_viewport.m_height / REFERENCE_HEIGHT;
        const float width = std::max(k, 0.0625f) * BLOOM_WIDTH;
        int levels = (int)std::lround(REFERENCE_LEVELS + std::log2(width));
        levels = std::min(std::max(levels, 3), (int)MAX_LEVELS);
        while (levels > 1 && isEmptyRect(item.m_rects[levels]))
            levels--;
        item.m_levels = levels;
        item.m_prefilter_scale = std::min(std::max(width, 0.25f), 2.0f);
        item.m_blur_scale = std::min(std::max(
            width / std::exp2((float)(levels - REFERENCE_LEVELS)), 0.25f),
            2.0f);
        if (m_streak[0] != NULL)
        {
            if (levels >= 2)
                item.m_streak_levels = 1;
            if (levels >= 3)
                item.m_streak_levels = 2;
            if (levels >= 3 && !isEmptyRect(item.m_rects[STREAK_LEVELS + 1]))
                item.m_streak_levels = 3;
            item.m_streak_extra_down = item.m_streak_levels == STREAK_LEVELS &&
                levels < (int)STREAK_LEVELS + 1;
            // Same width relative to the viewport as the levels
            item.m_streak_scale = std::min(std::max(k, 0.5f), 2.0f);
        }
        max_levels = std::max(max_levels, item.m_levels);
        items.push_back(item);
    }
    if (items.empty())
        return false;

    // hdr was written by the lighting pass, and what the last frame read or
    // wrote of the images (the tonemap pass, and this) has to be finished
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
            VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd,
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    // Every level is dispatched for all the viewports, and then there is a
    // barrier (their areas never overlap)
    // 1. Threshold and downscale of hdr to the first level
    for (const Item& item : items)
    {
        dispatchBlur(cmd, m_pipeline_layout, m_sets[0], BM_BLOOM_PREFILTER,
            item.m_rects[0], item.m_rects[1], item.m_prefilter_scale);
    }
    // 2. The rest of the levels, each one is half of the previous one
    for (unsigned level = 1; level < max_levels; level++)
    {
        computeToComputeBarrier(cmd, 0);
        for (const Item& item : items)
        {
            if (item.m_levels <= level)
                continue;
            dispatchBlur(cmd, m_pipeline_layout, m_sets[level], BM_DOWN,
                item.m_rects[level], item.m_rects[level + 1],
                item.m_blur_scale);
        }
    }
    // 2b. The streak, from the levels as they are now (the next step changes
    // them)
    bool has_streak = false;
    for (const Item& item : items)
        has_streak = has_streak || item.m_streak_levels > 0;
    if (has_streak)
    {
        bool extra_down = false;
        for (const Item& item : items)
            extra_down = extra_down || item.m_streak_extra_down;
        if (extra_down)
        {
            computeToComputeBarrier(cmd, 0);
            for (const Item& item : items)
            {
                if (!item.m_streak_extra_down)
                    continue;
                dispatchBlur(cmd, m_pipeline_layout, m_sets[STREAK_LEVELS],
                    BM_DOWN, item.m_rects[STREAK_LEVELS],
                    item.m_rects[STREAK_LEVELS + 1], item.m_blur_scale);
            }
        }
        computeToComputeBarrier(cmd, 0);
        for (const Item& item : items)
        {
            for (unsigned i = 0; i < item.m_streak_levels; i++)
            {
                dispatchBlur(cmd, m_pipeline_layout, m_streak_sets[i],
                    BM_STREAK, item.m_rects[i + 2], item.m_rects[i + 2],
                    item.m_streak_scale);
            }
        }
        // Same as the up chain below, from the smallest to the first one
        for (int i = (int)STREAK_LEVELS - 2; i >= 0; i--)
        {
            computeToComputeBarrier(cmd, 0);
            for (const Item& item : items)
            {
                if (item.m_streak_levels <= (unsigned)i + 1)
                    continue;
                dispatchBlur(cmd, m_pipeline_layout,
                    m_streak_sets[STREAK_LEVELS + i], BM_UP_ADD,
                    item.m_rects[i + 3], item.m_rects[i + 2],
                    item.m_blur_scale, UPSCALE_BLEND);
            }
        }
    }

    // 3. Back up to the first level, from the smallest one of every viewport
    for (int level = (int)max_levels - 2; level >= 0; level--)
    {
        computeToComputeBarrier(cmd, 0);
        for (const Item& item : items)
        {
            if (item.m_levels <= (unsigned)level + 1)
                continue;
            dispatchBlur(cmd, m_pipeline_layout,
                m_sets[MAX_LEVELS + level], BM_UP_ADD,
                item.m_rects[level + 2], item.m_rects[level + 1],
                item.m_blur_scale, UPSCALE_BLEND);
        }
    }

    // 3b. The streak goes in the first level
    if (has_streak)
    {
        computeToComputeBarrier(cmd, 0);
        for (const Item& item : items)
        {
            if (item.m_streak_levels == 0)
                continue;
            dispatchBlur(cmd, m_pipeline_layout,
                m_streak_sets[2 * STREAK_LEVELS - 1], BM_UP_ADD,
                item.m_rects[2], item.m_rects[1], item.m_blur_scale,
                STREAK_MIX);
        }
    }

    // 4. The first level is sampled by the tonemap pass
    {
        VkMemoryBarrier barrier = {};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0,
            NULL);
    }
    // hdr is read by the compute shader (execution dependency only), the next
    // frame writes it again in the lighting pass, which comes after its
    // g-buffer
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, NULL, 0, NULL, 0,
        NULL);

    for (const Item& item : items)
    {
        GEVulkanBloomRects r = {};
        for (unsigned i = 0; i < 4; i++)
        {
            r.m_viewport[i] = item.m_rects[0][i];
            r.m_blur[i] = item.m_rects[1][i];
            // Empty (all zero) if the viewport is too small for it
            if (item.m_levels > WIDE_LEVEL)
                r.m_wide[i] = item.m_rects[WIDE_LEVEL + 1][i];
        }
        m_active.push_back({ item.m_viewport.m_draw_call, r });
    }
    return true;
}   // render

}
