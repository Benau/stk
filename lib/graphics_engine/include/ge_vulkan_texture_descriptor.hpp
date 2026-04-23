#ifndef HEADER_GE_VULKAN_TEXTURE_DESCRIPTOR_HPP
#define HEADER_GE_VULKAN_TEXTURE_DESCRIPTOR_HPP

#include "vulkan_wrapper.h"

#include "IrrCompileConfig.h"
#include "SMaterial.h"
namespace irr
{
    namespace video { class ITexture; }
}

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace GE
{
class GEMaterial;
class GEVulkanDriver;
class GEVulkanTexture;

enum GEVulkanSampler : unsigned;

class GEVulkanTextureDescriptor
{
    struct TextureDescriptorData
    {
        std::shared_ptr<int> m_slot;
        std::array<VkImageView, _IRR_MATERIAL_MAX_TEXTURES_> m_image_views;
        std::array<std::weak_ptr<bool>, _IRR_MATERIAL_MAX_TEXTURES_>
            m_texture_observers;
    };

    typedef std::array<std::pair<GEVulkanTexture*, bool>,
        _IRR_MATERIAL_MAX_TEXTURES_> TextureList;

    std::map<TextureList, TextureDescriptorData> m_texture_list;

    GEVulkanTexture* m_white_image;

    GEVulkanTexture* m_transparent_image;

    std::shared_ptr<VkDescriptorSetLayout> m_descriptor_set_layout;

    VkDescriptorPool m_descriptor_pool;

    std::vector<VkDescriptorSet> m_descriptor_sets;

    // Initial capacity; rebuildPool() can grow m_max_texture_list beyond this,
    // and clear() shrinks it back.
    const unsigned m_original_capacity;

    // Current pool capacity — may grow past m_original_capacity at runtime.
    unsigned m_max_texture_list;

    const unsigned m_max_layer;

    const unsigned m_binding;

    // Recycled slot IDs returned by expired entries.
    std::vector<int> m_free_slots;

    // Monotonically-increasing counter; only advances when m_free_slots
    // is empty, so every live entry keeps its ID indefinitely.
    int m_next_id;

    // Stored so rebuildPool() can reconstruct the pool/layout correctly.
    bool m_single_descriptor;

    GEVulkanSampler m_sampler_use;

    GEVulkanDriver* m_vk;

    bool m_needs_update_descriptor;

    // ------------------------------------------------------------------------
    std::shared_ptr<VkDescriptorSetLayout> createLayout(
                                                      unsigned capacity) const;
    // ------------------------------------------------------------------------
    void buildPool(unsigned new_capacity);
    // ------------------------------------------------------------------------
    void rebuildPool(unsigned new_capacity);
public:
    // ------------------------------------------------------------------------
    GEVulkanTextureDescriptor(unsigned max_texture_list, unsigned max_layer,
        bool single_descriptor, unsigned binding = 0);
    // ------------------------------------------------------------------------
    ~GEVulkanTextureDescriptor();
    // ------------------------------------------------------------------------
    void clear()
    {
        m_texture_list.clear();
        m_free_slots.clear();
        m_next_id = 0;
        // Shrink the descriptor pool back to its original size if it grew.
        if (m_max_texture_list != m_original_capacity)
            rebuildPool(m_original_capacity);
    }
    // ------------------------------------------------------------------------
    std::shared_ptr<int>& getTextureID(const irr::video::SMaterial& material,
                                       const GEMaterial* ge_material);
    // ------------------------------------------------------------------------
    std::shared_ptr<int>& getTextureID(const irr::video::ITexture* t)
    {
        static irr::video::SMaterial single_material;
        single_material.setTexture(0, const_cast<irr::video::ITexture*>(t));
        return getTextureID(single_material, NULL);
    }
    // ------------------------------------------------------------------------
    void setSamplerUse(GEVulkanSampler sampler)
    {
        if (m_sampler_use == sampler)
            return;
        m_sampler_use = sampler;
        m_needs_update_descriptor = true;
    }
    // ------------------------------------------------------------------------
    void updateDescriptor();
    // ------------------------------------------------------------------------
    unsigned getMaxTextureList() const           { return m_max_texture_list; }
    // ------------------------------------------------------------------------
    unsigned getMaxLayer() const                        { return m_max_layer; }
    // ------------------------------------------------------------------------
    std::shared_ptr<VkDescriptorSetLayout> getDescriptorSetLayout() const
                                            { return m_descriptor_set_layout; }
    // ------------------------------------------------------------------------
    VkDescriptorSet* getDescriptorSet()
                                           { return m_descriptor_sets.data(); }
    // ------------------------------------------------------------------------
    GEVulkanSampler getSamplerUse() const             { return m_sampler_use; }
};   // GEVulkanTextureDescriptor

}

#endif
