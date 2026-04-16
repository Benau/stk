#ifndef HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP
#define HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP

#include "ge_vulkan_shadow_fbo.hpp"

namespace GE
{

// ----------------------------------------------------------------------------
const unsigned OMNI_FACES_PER_LIGHT = 6;
// ----------------------------------------------------------------------------
class GEVulkanOmniShadowFBO : public GEVulkanShadowFBO
{
private:
    GEVulkanLightHandler* m_light_handler;

    std::vector<irr::scene::ISceneNode*> m_nodes;
    // ------------------------------------------------------------------------
    virtual unsigned getLayerOffset() const                       { return 0; }
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

};   // GEVulkanOmniShadowFBO

}   // namespace GE

#endif   // HEADER_GE_VULKAN_OMNI_SHADOW_FBO_HPP
