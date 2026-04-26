#ifndef HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP
#define HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP

#include "ge_vulkan_shadow_fbo.hpp"
#include "vector3d.h"

namespace GE
{
struct GEOmniLight
{
    irr::core::vector3df m_position;
    float m_radius;
};

// ----------------------------------------------------------------------------
const unsigned OMNI_FACES_PER_LIGHT = 6;
// ----------------------------------------------------------------------------
class GEVulkanOmniShadowFBO : public GEVulkanShadowFBO
{
private:
    GEVulkanLightHandler* m_light_handler;

    std::vector<irr::scene::ISceneNode*> m_nodes;

    std::vector<GEOmniLight> m_lights;
    // ------------------------------------------------------------------------
    virtual unsigned getLayerOffset() const                       { return 0; }
    // ------------------------------------------------------------------------
    // Groups every 2 lights into one bucket so only shadow_limit/2 draw calls
    // need to have geometry built; the other layers borrow via
    // swapDrawCallData.
    virtual unsigned getSharingDrawCallCount() const
    {
        const unsigned count = 2 * OMNI_FACES_PER_LIGHT;
        static_assert((count % OMNI_FACES_PER_LIGHT) == 0,
            "it should be a multiple of OMNI_FACES_PER_LIGHT");
        return count;
    }
    // ------------------------------------------------------------------------
    virtual int getLayerOwner(int pointlight_layer) const
    {
        if (pointlight_layer >= 0)
        {
            return (pointlight_layer / getSharingDrawCallCount()) *
                getSharingDrawCallCount();
        }
        return -1;
    }
    // ------------------------------------------------------------------------
    // Builds the single perspective matrix for a spotlight into face slot 0.
    void buildSingleFaceMatrices(unsigned light_id);
    // ------------------------------------------------------------------------
    // Builds all six cubemap face matrices for a point light (or wide-cone
    // spotlight fallback).
    void buildSixFaceMatrices(unsigned light_id);
public:
    // ------------------------------------------------------------------------
    GEVulkanOmniShadowFBO(GEVulkanDriver* vk, unsigned shadow_size,
                          GEVulkanDrawCall* master_dc,
                          irr::scene::ILightSceneNode* sun);
    // ------------------------------------------------------------------------
    virtual void createDrawCalls();
    // ------------------------------------------------------------------------
    virtual void prepare(irr::scene::ICameraSceneNode* cam,
                         GEVulkanLightHandler* lh)
    {
        m_light_handler = lh;
        m_nodes.clear();
    }
    // ------------------------------------------------------------------------
    virtual void addNode(irr::scene::ISceneNode* n)   { m_nodes.push_back(n); }
    // ------------------------------------------------------------------------
    virtual void generate();
    // ------------------------------------------------------------------------
    const std::vector<GEOmniLight>& getLights() const      { return m_lights; }

};   // GEVulkanOmniShadowFBO

}   // namespace GE

#endif   // HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP
