#include "ge_vulkan_texture_descriptor.hpp"

#include "ge_main.hpp"
#include "ge_material_manager.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_texture.hpp"

#include <algorithm>
#include <exception>
#include <stdexcept>

namespace GE
{
// ============================================================================
static void destroyLayout(VkDescriptorSetLayout* layout)
{
    vkDestroyDescriptorSetLayout(
        static_cast<GEVulkanDriver*>(getDriver())->getDevice(), *layout, NULL);
    delete layout;
}   // destroyLayout

// ----------------------------------------------------------------------------
GEVulkanTextureDescriptor::GEVulkanTextureDescriptor(unsigned max_texture_list,
                                                     unsigned max_layer,
                                                     bool single_descriptor,
                                                     unsigned binding)
                         : m_descriptor_pool(VK_NULL_HANDLE),
                           m_original_capacity(max_texture_list),
                           m_max_texture_list(max_texture_list),
                           m_max_layer(max_layer), m_binding(binding),
                           m_next_id(0),
                           m_single_descriptor(single_descriptor),
                           m_needs_update_descriptor(false)
{
    if (m_max_layer > _IRR_MATERIAL_MAX_TEXTURES_)
    {
        throw std::runtime_error(
            "Too large max_layer for GEVulkanTextureDescriptor");
    }

    m_vk = getVKDriver();

    // m_descriptor_set_layout
    m_descriptor_set_layout = createLayout(m_max_texture_list);
    buildPool(m_max_texture_list);

    m_sampler_use = GVS_NEAREST;
    m_white_image = static_cast<GEVulkanTexture*>(
        m_vk->getWhiteTexture());
    m_transparent_image = static_cast<GEVulkanTexture*>(
        m_vk->getTransparentTexture());
}   // GEVulkanTextureDescriptor

// ----------------------------------------------------------------------------
GEVulkanTextureDescriptor::~GEVulkanTextureDescriptor()
{
    m_descriptor_set_layout.reset();
    vkDestroyDescriptorPool(m_vk->getDevice(), m_descriptor_pool, NULL);
}   // ~GEVulkanTextureDescriptor

// ----------------------------------------------------------------------------
std::shared_ptr<VkDescriptorSetLayout> GEVulkanTextureDescriptor::createLayout(
                                                       unsigned capacity) const
{
    std::vector<VkDescriptorSetLayoutBinding> texture_layout_binding;
    texture_layout_binding.resize(1);
    texture_layout_binding[0].binding = m_binding;
    texture_layout_binding[0].descriptorCount =
        m_single_descriptor ? capacity * m_max_layer : 1;
    texture_layout_binding[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    texture_layout_binding[0].pImmutableSamplers = NULL;
    texture_layout_binding[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    if (!m_single_descriptor)
    {
        texture_layout_binding.resize(m_max_layer, texture_layout_binding[0]);
        for (unsigned i = 1; i < m_max_layer; i++)
            texture_layout_binding[i].binding = m_binding + i;
    }

    VkDescriptorSetLayoutCreateInfo setinfo = {};
    setinfo.flags = 0;
    setinfo.pNext = NULL;
    setinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setinfo.pBindings = texture_layout_binding.data();
    setinfo.bindingCount = texture_layout_binding.size();

    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    if (vkCreateDescriptorSetLayout(m_vk->getDevice(), &setinfo,
        NULL, &layout) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorSetLayout failed for "
            "GEVulkanTextureDescriptor");
    }
    return std::shared_ptr<VkDescriptorSetLayout>(new VkDescriptorSetLayout(
        layout), destroyLayout);
}   // createLayout

// ----------------------------------------------------------------------------
void GEVulkanTextureDescriptor::buildPool(unsigned new_capacity)
{
    if (m_descriptor_pool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_vk->getDevice(), m_descriptor_pool, NULL);
        m_descriptor_pool = VK_NULL_HANDLE;
    }

    // For single_descriptor the layout's descriptorCount encodes the array
    // size seen by the shader — it must change together with the pool.
    if (m_single_descriptor && new_capacity != m_max_texture_list)
        m_descriptor_set_layout = createLayout(new_capacity);
    m_max_texture_list = new_capacity;

    // m_descriptor_pool
    VkDescriptorPoolSize pool_size;
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = m_max_texture_list * m_max_layer;

    VkDescriptorPoolCreateInfo pool_info = {};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags = 0;
    pool_info.maxSets = m_single_descriptor ? 1 : m_max_texture_list;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;
    if (vkCreateDescriptorPool(m_vk->getDevice(), &pool_info, NULL,
        &m_descriptor_pool) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed in "
            "GEVulkanTextureDescriptor::buildPool");
    }

    // m_descriptor_sets
    if (m_single_descriptor)
        m_descriptor_sets.resize(1);
    else
        m_descriptor_sets.resize(m_max_texture_list);
    std::vector<VkDescriptorSetLayout> layouts(m_descriptor_sets.size(),
        *m_descriptor_set_layout);

    VkDescriptorSetAllocateInfo alloc_info = {};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = m_descriptor_pool;
    alloc_info.descriptorSetCount = layouts.size();
    alloc_info.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(m_vk->getDevice(), &alloc_info,
        m_descriptor_sets.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed in "
            "GEVulkanTextureDescriptor::buildPool");
    }
}   // buildPool

// ----------------------------------------------------------------------------
/** Destroys and recreates the descriptor pool (and, for single_descriptor
 *  mode, the layout too) at the requested capacity, then re-uploads all live
 *  descriptors. Must be called with no frames in flight.
 */
void GEVulkanTextureDescriptor::rebuildPool(unsigned new_capacity)
{
    m_vk->waitIdle();
    buildPool(new_capacity);
    // All existing descriptor sets are fresh — force a full re-upload.
    m_needs_update_descriptor = true;
}   // rebuildPool

// ----------------------------------------------------------------------------
void GEVulkanTextureDescriptor::updateDescriptor()
{
    if (m_texture_list.empty())
        return;

    std::vector<VkDescriptorImageInfo> image_infos;
    VkDescriptorImageInfo dummy_info;
    dummy_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    dummy_info.sampler = m_vk->getSampler(m_sampler_use);
    dummy_info.imageView = m_transparent_image->getImageView();
    // Size by current pool capacity so sparse IDs never go out of bounds.
    image_infos.resize(m_max_texture_list * m_max_layer, dummy_info);

    bool image_views_changed = false;
    auto it = m_texture_list.begin();
    while (it != m_texture_list.end())
    {
        bool all_layers_expired = true;
        TextureDescriptorData& tdd = it->second;
        int slot = *tdd.m_slot;
        for (unsigned i = 0; i < m_max_layer; i++)
        {
            VkDescriptorImageInfo& info = image_infos[slot * m_max_layer + i];
            GEVulkanTexture* cur_layer = it->first[i].first;
            bool cur_srgb = it->first[i].second;
            if (!cur_layer)
                continue;
            if (tdd.m_texture_observers[i].expired() &&
                tdd.m_image_views[i] != VK_NULL_HANDLE)
            {
                image_views_changed = true;
                tdd.m_image_views[i] = VK_NULL_HANDLE;
                continue;
            }
            all_layers_expired = false;
            VkImageView v = cur_layer->getImageView(cur_srgb);
            if (v != tdd.m_image_views[i])
            {
                if (cur_layer->useOnDemandLoad() && v == dummy_info.imageView)
                {
                    // Notify the observer to load the texture on demand
                    tdd.m_slot = std::make_shared<int>(slot);
                }
                tdd.m_image_views[i] = v;
                image_views_changed = true;
            }
            info.imageView = v;
        }
        if (all_layers_expired)
        {
            // Return the slot so it can be reused for a new texture entry.
            m_free_slots.push_back(slot);
            image_views_changed = true;
            it = m_texture_list.erase(it);
        }
        else
            it++;
    }

    if (!image_views_changed && !m_needs_update_descriptor)
        return;

    m_needs_update_descriptor = false;
    bool single_descriptor = (m_descriptor_sets.size() == 1);

    m_vk->waitIdle();
    if (single_descriptor)
    {
        VkWriteDescriptorSet write_descriptor_set = {};
        write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_descriptor_set.dstBinding = m_binding;
        write_descriptor_set.dstArrayElement = 0;
        write_descriptor_set.descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        write_descriptor_set.descriptorCount = m_max_texture_list * m_max_layer;
        write_descriptor_set.pBufferInfo = 0;
        write_descriptor_set.dstSet = m_descriptor_sets[0];
        write_descriptor_set.pImageInfo = image_infos.data();

        vkUpdateDescriptorSets(m_vk->getDevice(), 1, &write_descriptor_set, 0,
            NULL);
    }
    else
    {
        std::vector<VkWriteDescriptorSet> all_sets;
        for (unsigned i = 0; i < image_infos.size(); i += m_max_layer)
        {
            const unsigned set_idx = i / m_max_layer;
            VkWriteDescriptorSet write_descriptor_set = {};
            write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write_descriptor_set.dstBinding = m_binding;
            write_descriptor_set.dstArrayElement = 0;
            write_descriptor_set.descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write_descriptor_set.descriptorCount = m_max_layer;
            write_descriptor_set.pBufferInfo = 0;
            write_descriptor_set.dstSet = m_descriptor_sets[set_idx];
            write_descriptor_set.pImageInfo = &image_infos[i];
            all_sets.push_back(write_descriptor_set);
        }
        vkUpdateDescriptorSets(m_vk->getDevice(), all_sets.size(),
            all_sets.data(), 0, NULL);
    }
}   // updateDescriptor

// ----------------------------------------------------------------------------
std::shared_ptr<int>& GEVulkanTextureDescriptor::getTextureID(
                                         const irr::video::SMaterial& material,
                                         const GEMaterial* ge_material)
{
    TextureList key =
    {{
        std::make_pair(m_white_image, false),
        std::make_pair((GEVulkanTexture*)NULL, false),
        std::make_pair((GEVulkanTexture*)NULL, false),
        std::make_pair((GEVulkanTexture*)NULL, false),
        std::make_pair((GEVulkanTexture*)NULL, false),
        std::make_pair((GEVulkanTexture*)NULL, false),
        std::make_pair((GEVulkanTexture*)NULL, false),
        std::make_pair((GEVulkanTexture*)NULL, false)
    }};
    for (unsigned i = 0; i < m_max_layer; i++)
    {
        if (material.getTexture(i))
        {
            key[i].first = static_cast<
                GEVulkanTexture*>(material.getTexture(i));
            if (key[i].first->useOnDemandLoad())
                key[i].first->getTextureHandler();
            key[i].second = getGEConfig()->m_pbr && ge_material ?
                ge_material->m_srgb_settings[i] : false;
        }
    }
    auto it = m_texture_list.find(key);
    if (it != m_texture_list.end())
        return it->second.m_slot;

    // Stable slot allocation: prefer recycled IDs from expired entries,
    // fall back to a fresh counter value.
    int slot;
    if (!m_free_slots.empty())
    {
        slot = m_free_slots.back();
        m_free_slots.pop_back();
    }
    else
    {
        slot = m_next_id;
        m_next_id++;
    }

    // Grow the pool if this slot exceeds current capacity.
    if ((unsigned)slot >= m_max_texture_list)
    {
        unsigned new_cap = m_max_texture_list;
        while (new_cap <= (unsigned)slot)
            new_cap *= 2;
        rebuildPool(new_cap);
    }

    auto& ret = m_texture_list[key];
    ret.m_slot = std::make_shared<int>(slot);
    ret.m_image_views = {};
    ret.m_texture_observers = {};
    for (unsigned i = 0; i < m_max_layer; i++)
    {
        GEVulkanTexture* t = key[i].first;
        if (!t)
            continue;
        ret.m_image_views[i] = VK_NULL_HANDLE;
        ret.m_texture_observers[i] = t->getTextureObserver();
    }
    return ret.m_slot;
}   // getTextureID

}
