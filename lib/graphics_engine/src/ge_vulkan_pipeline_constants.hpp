#ifndef HEADER_GE_VULKAN_PIPELINE_CONSTANTS_HPP
#define HEADER_GE_VULKAN_PIPELINE_CONSTANTS_HPP

#include "vulkan_wrapper.h"

#include <array>
#include <cstdint>

namespace GE
{
class GEVulkanDriver;

// The specialization constants of utils/constants_utils.glsl, shared by the
// pipelines which include it: the graphics pipelines of GEVulkanDrawCall and
// the compute pipelines which shade like them (the trace of
// GEVulkanSolidReflection). Most of them come from the config and the FBO,
// the rest from the draw call which uses the pipeline (Options).
//
// getInfo() points into this object, so it isn't copied.
class GEVulkanPipelineConstants
{
public:
    // What depends on the draw call of the pipeline
    struct Options
    {
        // The draw call renders a shadow map
        bool m_shadow;
        // It has a skybox, image based lighting needs the environment maps
        // of it
        bool m_has_skybox;
        // It renders to a deferred FBO
        bool m_deferred;
        // It has a shadow map (GEVulkanShadowFBO)
        bool m_shadow_map;
        // The ray traced scene can be used for the reflection of its displace
        // materials
        bool m_displace_rt;
        // IBL is used
        bool m_ibl;
        // --------------------------------------------------------------------
        bool operator==(const Options& o) const
        {
            return m_shadow == o.m_shadow && m_has_skybox == o.m_has_skybox &&
                m_deferred == o.m_deferred && m_shadow_map == o.m_shadow_map &&
                m_displace_rt == o.m_displace_rt && m_ibl == o.m_ibl;
        }
    };
private:
    // The order of the members is the order of the constant_id
#define GE_SPECIALIZATION_CONSTANTS(X)    \
    X(VkBool32, m_ibl)                    \
    X(float, m_specular_levels_minus_one) \
    X(VkBool32, m_deferred)               \
    X(uint32_t, m_solid_reflection)       \
    X(uint32_t, m_displace_reflection)    \
    X(uint32_t, m_shadow_size)            \
    X(uint32_t, m_shadow_type)            \
    X(uint32_t, m_point_shadow_limit)     \
    X(VkBool32, m_glow_outline)           \
    X(VkBool32, m_light_scatter)          \
    X(VkBool32, m_bloom)                  \
    X(VkBool32, m_bloom_blend)            \
    X(VkBool32, m_ssao)                   \
    X(uint32_t, m_rt_shadow_type)         \
    X(float, m_rt_shadow_scale)           \
    X(VkBool32, m_god_rays)               \
    X(VkBool32, m_surfel_gi)              \
    X(VkBool32, m_solid_reflection_image)
    struct Data
    {
#define GE_DECLARE_MEMBER(type, name) type name;
        GE_SPECIALIZATION_CONSTANTS(GE_DECLARE_MEMBER)
#undef GE_DECLARE_MEMBER
    };

    enum
    {
        COUNT = 0
#define GE_COUNT_MEMBER(type, name) + 1
        GE_SPECIALIZATION_CONSTANTS(GE_COUNT_MEMBER)
#undef GE_COUNT_MEMBER
    };

    Data m_data;

    std::array<VkSpecializationMapEntry, COUNT> m_entries;

    VkSpecializationInfo m_info;
public:
    // ------------------------------------------------------------------------
    GEVulkanPipelineConstants(GEVulkanDriver* vk, const Options& options);
    // ------------------------------------------------------------------------
    GEVulkanPipelineConstants(const GEVulkanPipelineConstants&) = delete;
    // ------------------------------------------------------------------------
    GEVulkanPipelineConstants& operator=(const GEVulkanPipelineConstants&)
                                                                      = delete;
    // ------------------------------------------------------------------------
    const VkSpecializationInfo* getInfo() const             { return &m_info; }
};   // GEVulkanPipelineConstants

}

#endif
