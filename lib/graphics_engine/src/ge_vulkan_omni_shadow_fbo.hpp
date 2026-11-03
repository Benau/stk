#ifndef HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP
#define HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP

#include "ge_vulkan_shadow_fbo.hpp"
#include "aabbox3d.h"
#include "vector3d.h"

#include <unordered_map>

namespace GE
{
struct GEOmniLight
{
    irr::core::vector3df m_position;
    float m_radius;
};

// ----------------------------------------------------------------------------
// Bevy style point light shadows: a point light has the 6 faces of a cube map,
// every face is a 90 degree perspective map. A spot light has a single
// perspective map of its cone and the other slots of the light are unused.
const unsigned OMNI_FACES_PER_LIGHT = 6;
// ----------------------------------------------------------------------------
class GEVulkanOmniShadowFBO : public GEVulkanShadowFBO
{
private:
    GEVulkanLightHandler* m_light_handler;

    std::vector<irr::scene::ISceneNode*> m_nodes;

    std::vector<GEOmniLight> m_lights;

    // The atlas is 2 * shadow_size wide, the tiles of the slots are a square
    // grid of m_tiles_per_row x m_tiles_per_row tiles of m_tile_size pixels
    unsigned m_tile_size;
    unsigned m_tiles_per_row;

    // What a tile holds in the atlas, to keep the tiles which don't change
    struct SlotState
    {
        bool m_active;
        // The tile has been rendered or cleared to 1.0 at least once
        bool m_rendered;
        bool m_spot;
        irr::core::vector3df m_position;
        float m_radius;
        irr::core::vector3df m_direction;
        float m_outer_cone;
        // --------------------------------------------------------------------
        SlotState()
            : m_active(false), m_rendered(false), m_spot(false),
              m_position(0.0f, 0.0f, 0.0f), m_radius(0.0f),
              m_direction(0.0f, 0.0f, 0.0f), m_outer_cone(0.0f)
        {
        }
    };
    std::vector<SlotState> m_slot_states;

    struct NodeBox
    {
        irr::core::aabbox3df m_box;
        bool m_animated;
    };
    // The nodes and their boxes of the last generate, to find the ones which
    // moved (or were added/removed) and with them the buckets which changed
    std::unordered_map<irr::scene::ISceneNode*, NodeBox> m_cached_nodes;
    std::unordered_map<irr::scene::ISceneNode*, NodeBox> m_next_cached_nodes;
    bool m_has_cached_nodes;

    // Work for render(), queued by generate()
    std::vector<unsigned> m_dirty_buckets;
    std::vector<unsigned> m_clear_tiles;

    // ------------------------------------------------------------------------
    virtual unsigned getSharingDrawCallCount() const;
    // ------------------------------------------------------------------------
    virtual int getLayerOwner(int pointlight_layer) const;
    // ------------------------------------------------------------------------
    // The atlas keeps the content of the tiles which didn't change
    virtual VkImageLayout getShadowInitialLayout() const
    {
        return VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    }
    // ------------------------------------------------------------------------
    // Builds the matrices of the map of a light: the 6 faces of the cube map
    // for a point light, a perspective projection of the cone for a spot
    void buildLightMatrices(unsigned light_id);
    // ------------------------------------------------------------------------
    bool bucketIntersectsBox(unsigned bucket,
                             const irr::core::aabbox3df& box) const;
    // ------------------------------------------------------------------------
    void addTileClear(unsigned slot);
    // ------------------------------------------------------------------------
    void renderTile(VkCommandBuffer cmd, GEVulkanCamera* cam, unsigned slot,
                    bool draw, bool& rebind_base_vertex);
public:
    // ------------------------------------------------------------------------
    GEVulkanOmniShadowFBO(GEVulkanDriver* vk, unsigned shadow_size,
                          GEVulkanDrawCall* master_dc);
    // ------------------------------------------------------------------------
    virtual void createDrawCalls();
    // ------------------------------------------------------------------------
    virtual void prepare(GEVulkanCamera* cam, GEVulkanLightHandler* lh);
    // ------------------------------------------------------------------------
    virtual void addNode(irr::scene::ISceneNode* n);
    // ------------------------------------------------------------------------
    virtual void generate();
    // ------------------------------------------------------------------------
    virtual void render(VkCommandBuffer cmd, GEVulkanCamera* cam);
    // ------------------------------------------------------------------------
    const std::vector<GEOmniLight>& getLights() const      { return m_lights; }

};   // GEVulkanOmniShadowFBO

}   // namespace GE

#endif   // HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP
