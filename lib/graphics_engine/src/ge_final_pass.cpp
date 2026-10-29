#include "ge_final_pass.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_anti_aliasing.hpp"
#include "ge_vulkan_depth_of_field.hpp"
#include "ge_vulkan_features.hpp"
#include "ge_vulkan_final_chain.hpp"
#include "ge_vulkan_fxaa.hpp"

#include <algorithm>

namespace GE
{
namespace
{
// ----------------------------------------------------------------------------
struct RegisteredPass
{
    unsigned m_handle;

    GEFinalPassInfo m_info;
};
// ----------------------------------------------------------------------------
unsigned g_next_handle = 1;
// ----------------------------------------------------------------------------
// The passes of GE are registered the same way as the ones of the game, they
// are in the registry from the start (the first use of it)
std::vector<RegisteredPass>& getRegistry()
{
    static std::vector<RegisteredPass> registry = []()
    {
        std::vector<RegisteredPass> passes;

        GEFinalPassInfo dof;
        dof.m_name = "depth_of_field";
        dof.m_order = GFPO_DEPTH_OF_FIELD;
        dof.m_needs_depth = true;
        dof.m_main_target_only = true;
        dof.m_enabled = []() { return isDepthOfFieldEnabled(); };
        dof.m_create = [](GEVulkanDriver* vk, VkDescriptorSetLayout io_layout,
            const irr::core::dimension2d<irr::u32>& capacity)
            -> GEVulkanFinalStage*
        {
            return new GEVulkanDepthOfField(vk, io_layout, capacity);
        };
        passes.push_back({ g_next_handle++, dof });

        GEFinalPassInfo aa;
        aa.m_name = "anti_aliasing";
        aa.m_order = GFPO_ANTI_ALIASING;
        aa.m_needs_depth = false;
        aa.m_main_target_only = false;
        aa.m_enabled = []() { return isAntiAliasingEnabled(); };
        aa.m_create = [](GEVulkanDriver* vk, VkDescriptorSetLayout io_layout,
            const irr::core::dimension2d<irr::u32>& capacity)
            -> GEVulkanFinalStage*
        {
            switch (getGEConfig()->m_anti_aliasing)
            {
            case GAA_FXAA:
                return new GEVulkanFXAA(vk, io_layout, capacity);
            case GAA_CMAA2:
                return new GEVulkanAntiAliasing(vk, io_layout, capacity);
            default:
                return NULL;
            }
        };
        passes.push_back({ g_next_handle++, aa });
        return passes;
    }();
    return registry;
}
// ----------------------------------------------------------------------------
bool isPassEnabled(const GEFinalPassInfo& info)
{
    // The passes are compute shaders in the main queue
    if (!GEVulkanFeatures::supportsComputeInMainQueue())
        return false;
    // The depth of the FBO is only kept by the split deferred FBO, which is
    // only used with the pbr shaders (refreshDeferredSplit)
    if (info.m_needs_depth && !getGEConfig()->m_pbr)
        return false;
    return !info.m_enabled || info.m_enabled();
}
}   // anonymous namespace

// ----------------------------------------------------------------------------
unsigned addFinalPass(const GEFinalPassInfo& info)
{
    std::vector<RegisteredPass>& registry = getRegistry();
    registry.push_back({ g_next_handle++, info });
    return registry.back().m_handle;
}   // addFinalPass

// ----------------------------------------------------------------------------
void removeFinalPass(unsigned handle)
{
    std::vector<RegisteredPass>& registry = getRegistry();
    registry.erase(std::remove_if(registry.begin(), registry.end(),
        [handle](const RegisteredPass& p) { return p.m_handle == handle; }),
        registry.end());
}   // removeFinalPass

// ----------------------------------------------------------------------------
std::vector<GEFinalPassInfo> getEnabledFinalPasses()
{
    std::vector<GEFinalPassInfo> passes;
    // The registry is in the order the passes were added, and the sort is
    // stable
    for (const RegisteredPass& p : getRegistry())
    {
        if (isPassEnabled(p.m_info))
            passes.push_back(p.m_info);
    }
    std::stable_sort(passes.begin(), passes.end(),
        [](const GEFinalPassInfo& a, const GEFinalPassInfo& b)
        { return a.m_order < b.m_order; });
    return passes;
}   // getEnabledFinalPasses

// ----------------------------------------------------------------------------
bool isFinalChainEnabled()
{
    for (const RegisteredPass& p : getRegistry())
    {
        if (isPassEnabled(p.m_info))
            return true;
    }
    return false;
}   // isFinalChainEnabled

// ----------------------------------------------------------------------------
bool needsFinalPassDepth()
{
    for (const RegisteredPass& p : getRegistry())
    {
        if (p.m_info.m_needs_depth && isPassEnabled(p.m_info))
            return true;
    }
    return false;
}   // needsFinalPassDepth

}
