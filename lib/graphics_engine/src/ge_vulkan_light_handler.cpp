#include "ge_vulkan_light_handler.hpp"

#include "ge_main.hpp"
#include "ge_occlusion_culling.hpp"
#include "ge_vulkan_camera_scene_node.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_fbo_texture.hpp"
#include "ge_vulkan_skybox_renderer.hpp"
#include "ge_vulkan_omni_shadow_fbo.hpp"

#include "ILightSceneNode.h"
#include "ISceneManager.h"
#include "IrrlichtDevice.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>

namespace GE
{
using namespace irr;
// ----------------------------------------------------------------------------
// Mirrors the GLSL "LightData" struct
struct GELight
{
    irr::core::vector3df m_position;
    irr::f32             m_radius;
    irr::core::vector3df m_color;
    irr::f32             m_inverse_range_squared;
    irr::core::vector2df m_direction;
    irr::f32             m_scale;
    irr::f32             m_offset;
    irr::core::matrix4   m_shadow_projection_view_matrix[OMNI_FACES_PER_LIGHT];
};

// ----------------------------------------------------------------------------
// Mirrors the GLSL "GlobalLightBuffer" uniform
struct GEGlobalLightBuffer
{
    irr::core::matrix4   m_shadow_projection_view_matrix[GVSCC_COUNT];
    irr::core::matrix4   m_shadow_view_matrix;
    irr::core::vector3df m_ambient_color;
    irr::f32             m_sun_scatter;
    irr::core::vector3df m_sun_color;
    irr::f32             m_sun_angle_tan_half;
    irr::core::vector3df m_sun_direction;
    irr::f32             m_fog_density;
    irr::video::SColorf  m_fog_color;
    irr::core::vector3df m_skytop_color;
    irr::u32             m_light_count;
    //std::array<GELight, MAX_RENDERING_LIGHT> m_rendering_lights;
};

// ----------------------------------------------------------------------------
struct GEVulkanLightHandler::GELightStorage
{
    std::vector<GELight> m_data;
};

// ----------------------------------------------------------------------------
GEGlobalLightBuffer* GEVulkanLightHandler::getGlobalLightPtr() const
{
    return (GEGlobalLightBuffer*)m_buffer.data();
}   // getGlobalLightPtr

// ----------------------------------------------------------------------------
GELight* GEVulkanLightHandler::getRenderingLightsPtr() const
{
    return (GELight*)(m_buffer.data() + sizeof(GEGlobalLightBuffer));
}   // getRenderingLightsPtr

// ----------------------------------------------------------------------------
size_t GEVulkanLightHandler::getSize() const
{
    return sizeof(GEGlobalLightBuffer) + getLightCount() * sizeof(GELight);
}   // getSize

// ----------------------------------------------------------------------------
size_t GEVulkanLightHandler::getMaxSize()
{
    return sizeof(GEGlobalLightBuffer) +
        sizeof(GELight) * getGEConfig()->m_max_pointlights;
}   // getMaxSize

// ----------------------------------------------------------------------------
unsigned GEVulkanLightHandler::getLightCount() const
{
    GEGlobalLightBuffer* buffer = getGlobalLightPtr();
    return buffer->m_light_count;
}   // getLightCount

// ----------------------------------------------------------------------------
const irr::core::vector3df& GEVulkanLightHandler::getLightPosition(
                                                       unsigned light_id) const
{
    GELight* lights = getRenderingLightsPtr();
    return lights[light_id].m_position;
}   // getLightPosition

// ----------------------------------------------------------------------------
float GEVulkanLightHandler::getLightRadius(unsigned light_id) const
{
    GELight* lights = getRenderingLightsPtr();
    return lights[light_id].m_radius;
}   // getLightRadius

// ----------------------------------------------------------------------------
GEVulkanLightHandler::GEVulkanLightHandler(GEVulkanDriver* vk)
{
    m_vk = vk;
    m_lights.reset(new GELightStorage);
    prepare();
}   // GEVulkanLightHandler

// ----------------------------------------------------------------------------
GEVulkanLightHandler::~GEVulkanLightHandler()
{
}   // ~GEVulkanLightHandler

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::prepare()
{
    m_buffer.resize(getMaxSize());
    GEGlobalLightBuffer* buffer = getGlobalLightPtr();
    *buffer = {};
    m_lights->m_data.clear();
    m_fullscreen_light_count = 0;
    m_non_occluded_lights = 0;
    video::SColorf c = m_vk->getIrrlichtDevice()->getSceneManager()
        ->getAmbientLight();
    buffer->m_ambient_color.X = c.r * c.a;
    buffer->m_ambient_color.Y = c.g * c.a;
    buffer->m_ambient_color.Z = c.b * c.a;
    buffer->m_sun_scatter = 0.2f;
    buffer->m_sun_color = core::vector3df(0.75f, 0.75f, 0.75f);
    buffer->m_sun_angle_tan_half = 0.0022f;
    buffer->m_sun_direction = core::vector3df(0.15f, 0.2f, 1.0f).normalize();
    buffer->m_skytop_color.X = 0.325f;
    buffer->m_skytop_color.Y = 0.35f;
    buffer->m_skytop_color.Z = 0.375f;
}   // prepare

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::generate(const irr::core::vector3df& cam_pos,
                                    GEVulkanSkyBoxRenderer* skybox)
{
    GEGlobalLightBuffer* buffer = getGlobalLightPtr();
    if (skybox)
    {
        irr::video::SColorf c(
            srgb255ToLinearFromSColor(skybox->getSkytopColor()));
        buffer->m_skytop_color.X = c.r;
        buffer->m_skytop_color.Y = c.g;
        buffer->m_skytop_color.Z = c.b;
    }
    if (m_lights->m_data.size() > getGEConfig()->m_max_pointlights)
    {
        std::sort(m_lights->m_data.begin(), m_lights->m_data.end(),
            [cam_pos](GELight& a, GELight& b)
            {
                float al = a.m_position.getDistanceFromSQ(cam_pos);
                float bl = b.m_position.getDistanceFromSQ(cam_pos);
                return al < bl;
            });
        m_lights->m_data.resize(getGEConfig()->m_max_pointlights);
    }
    if (m_lights->m_data.empty())
        return;

    bool set_occluded_light = false;
    GEVulkanFBOTexture* t =
        static_cast<GEVulkanDriver*>(getDriver())->getRTTTexture();
    if (t && t->isDeferredFBO())
    {
        auto i = std::stable_partition(m_lights->m_data.begin(),
            m_lights->m_data.end(), [cam_pos](const GELight& l)
        {
            float radius_2 = l.m_radius * l.m_radius;
            float distance_2 = (cam_pos - l.m_position).getLengthSQ();
            return distance_2 <= radius_2;
        });
        m_fullscreen_light_count = std::distance(m_lights->m_data.begin(), i);

        // Non-occluded lights before occluded ones. Fullscreen lights
        // (camera inside radius) are inherently non-occluded, so start the
        // partition from the iterator just past them.
        if (hasOcclusionCulling() &&
            (getGEConfig()->m_shadow_type & GST_POINTLIGHT) != 0)
        {
            auto j = std::stable_partition(i, m_lights->m_data.end(),
                [cam_pos](const GELight& l)
            {
                return !getOcclusionCulling()->isOccluded(
                    cam_pos, l.m_position, l.m_radius);
            });
            m_non_occluded_lights =
                (unsigned)std::distance(m_lights->m_data.begin(), j);
            set_occluded_light = true;
        }
    }
    // Deferred fbo supports light culling using depth test
    if (hasOcclusionCulling() && (!t || !t->isDeferredFBO()))
    {
        auto l = m_lights->m_data.begin();
        GELight* rl = getRenderingLightsPtr();
        while (l != m_lights->m_data.end())
        {
            if (getOcclusionCulling()->isOccluded(cam_pos, l->m_position,
                l->m_radius))
            {
                l++;
                continue;
            }
            *rl = *l;
            l++;
            rl++;
        }
        getGlobalLightPtr()->m_light_count = rl - getRenderingLightsPtr();
        // All lights that made it into the render buffer are non-occluded.
        m_non_occluded_lights = getLightCount();
    }
    else
    {
        memcpy((void*)getRenderingLightsPtr(), m_lights->m_data.data(),
            m_lights->m_data.size() * sizeof(GELight));
        getGlobalLightPtr()->m_light_count = m_lights->m_data.size();
        if (!set_occluded_light)
            m_non_occluded_lights = getLightCount();
    }
}   // generate

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::addLightNode(irr::scene::ILightSceneNode* node)
{
    const video::SLight& l = node->getLightData();
    GEGlobalLightBuffer* buffer = getGlobalLightPtr();
    if (node->getLightType() == irr::video::ELT_DIRECTIONAL)
    {
        buffer->m_sun_color.X = l.DiffuseColor.r;
        buffer->m_sun_color.Y = l.DiffuseColor.g;
        buffer->m_sun_color.Z = l.DiffuseColor.b;
        buffer->m_sun_scatter = l.DiffuseColor.a;
        core::vector3df dir = l.Direction;
        buffer->m_sun_direction = -dir.normalize();
        buffer->m_sun_angle_tan_half = tanf(l.Radius * 0.5f);
    }
    else
    {
        GELight gl = {};
        gl.m_position = l.Position;
        gl.m_radius = l.Radius;
        gl.m_color.X = l.DiffuseColor.r * l.Attenuation.X;
        gl.m_color.Y = l.DiffuseColor.g * l.Attenuation.X;
        gl.m_color.Z = l.DiffuseColor.b * l.Attenuation.X;
        gl.m_inverse_range_squared = l.Attenuation.Y * l.Attenuation.Y;
        if (l.Type == irr::video::ELT_SPOT)
        {
            gl.m_direction.X = l.Direction.X;
            gl.m_direction.Y = l.Direction.Y;
            float cos_outer = cosf(l.OuterCone);
            gl.m_scale = 1.0f / std::max(cosf(l.InnerCone) - cos_outer, 1e-4f);
            gl.m_offset = -cos_outer * gl.m_scale;
            gl.m_scale *= l.Direction.Z > 0.f ? 1.f : -1.f;
        }
        m_lights->m_data.push_back(gl);
    }
}   // addLightNode

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::setShadowMatrices(GEVulkanCameraUBO* ubo,
                                             GEVulkanShadowCameraCascade cc)
{
    GEGlobalLightBuffer* buffer = getGlobalLightPtr();
    if (cc == GVSCC_NEAR)
        buffer->m_shadow_view_matrix = ubo->m_view_matrix;
    buffer->m_shadow_projection_view_matrix[cc] =
        ubo->m_projection_view_matrix;
}   // setShadowMatrices

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::setLightShadowMatrices(GEVulkanCameraUBO* ubo,
                                                  unsigned light,
                                                  unsigned face)
{
    GELight* lights = getRenderingLightsPtr();
    lights[light].m_shadow_projection_view_matrix[face] =
        ubo->m_projection_view_matrix;
}   // setLightShadowMatrices

// ----------------------------------------------------------------------------
bool GEVulkanLightHandler::getLightIsSpot(unsigned light_id) const
{
    // m_scale is set to a non-zero value only for spotlights in addLightNode.
    GELight* lights = getRenderingLightsPtr();
    return lights[light_id].m_scale != 0.0f;
}   // getLightIsSpot

// ----------------------------------------------------------------------------
irr::core::vector3df GEVulkanLightHandler::getLightDirection(
                                                       unsigned light_id) const
{
    // addLightNode stores only (X, Y); Z is reconstructed as
    //   Z = sqrt(1 - X² - Y²) * sign(m_scale)
    // because m_scale was multiplied by sign(Direction.Z) on the way in.
    GELight* lights  = getRenderingLightsPtr();
    const float x    = lights[light_id].m_direction.X;
    const float y    = lights[light_id].m_direction.Y;
    const float z_sq = 1.0f - x * x - y * y;
    float z = sqrtf(z_sq > 0.0f ? z_sq : 0.0f);
    if (lights[light_id].m_scale < 0.0f)
        z = -z;
    return irr::core::vector3df(x, y, z);
}   // getLightDirection

// ----------------------------------------------------------------------------
float GEVulkanLightHandler::getLightOuterCone(unsigned light_id) const
{
    // From addLightNode:
    //   m_scale  = ±1 / (cos_inner - cos_outer)      [sign from Direction.Z]
    //   m_offset = -cos_outer * |m_scale|
    //   => cos_outer = -m_offset / |m_scale|
    GELight* lights       = getRenderingLightsPtr();
    const float abs_scale = fabsf(lights[light_id].m_scale);
    if (abs_scale < 1e-6f)
        return 0.0f;
    const float cos_outer = std::max(-1.0f,
        std::min(1.0f, -lights[light_id].m_offset / abs_scale));
    return acosf(cos_outer);   // half-angle in radians
}   // getLightOuterCone

}
