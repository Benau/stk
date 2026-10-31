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
#include <limits>
#include <unordered_map>

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
    // Density of the exponential fog of solid materials and light scattering
    // (1 / (40 * (start + 0.001)), same as the OpenGL renderer), 0 = no fog
    irr::f32             m_fog_density;
    // rgb: raw color of the fog (not converted to linear, same as OpenGL),
    // a: maximum amount of fog of transparent materials
    irr::video::SColorf  m_fog_color;
    irr::core::vector3df m_skytop_color;
    irr::u32             m_light_count;
    // Start and end of the linear fog of transparent materials
    irr::f32             m_fog_range[4];
    // xyz: view depth where each sun cascade starts, w: cosine of the largest
    // angle between the view direction and a frustum ray (see setShadowSplit)
    irr::f32             m_shadow_split[4];
    //std::array<GELight, MAX_RENDERING_LIGHT> m_rendering_lights;
};

// ----------------------------------------------------------------------------
// Time to fade a light in or out, same rate as the OpenGL renderer fade-in
static const float LIGHT_FADE_TIME = 1.0f;
// A light that was already selected has its squared distance multiplied by
// this when competing for the m_max_pointlights slots, so lights near the
// cutoff don't swap in and out every frame (0.8 is ~10% in distance)
static const float LIGHT_STICKY_DIST_SQ = 0.8f;
// Same for being in the point shadow slots, larger because a shadow
// appearing or disappearing is more visible (0.6 is ~22% in distance)
static const float LIGHT_SHADOW_STICKY_DIST_SQ = 0.6f;
// A light fading in needs at least this weight to take a shadow slot
static const float LIGHT_SHADOW_MIN_WEIGHT = 0.5f;
// After a hitch (loading, pause) longer than this the fade is skipped
static const float LIGHT_SNAP_TIME = 0.5f;
// Lights not submitted nor touched for this long are forgotten (deleted,
// hidden), seconds of accumulated frame time
static const float LIGHT_FORGET_TIME = 2.0f;
static const float LIGHT_PRUNE_INTERVAL = 1.0f;

// ----------------------------------------------------------------------------
// State of a light that survives from frame to frame, keyed by
// ILightSceneNode::getUniqueLightID so the node doesn't need to be referenced
struct GELightState
{
    // 0 to 1, multiplied to the light color
    float    m_weight;
    // Whether the light is in the m_max_pointlights set (or was when it was
    // last submitted), the weight moves toward 1 if so, otherwise toward 0
    bool     m_selected;
    // Whether the light had a point shadow slot in the last frame
    bool     m_shadow;
    // Light has ILightSceneNode::isNoFadeInOutActive set: no fading
    bool     m_no_fade;
    unsigned m_frame;
    float    m_last_seen;
    // ------------------------------------------------------------------------
    GELightState()
        : m_weight(0.0f), m_selected(true), m_shadow(false), m_no_fade(false),
          m_frame(0),
          m_last_seen(0.0f)
    {
    }   // GELightState
};

// ----------------------------------------------------------------------------
struct GELightEntry
{
    GELight       m_light;
    // Stable while the entry is in the map, unordered_map is node based
    GELightState* m_state;
    float         m_distance_sq;
};

// ----------------------------------------------------------------------------
struct GEVulkanLightHandler::GELightStorage
{
    // Lights submitted this frame, rebuilt every frame
    std::vector<GELightEntry> m_data;

    // Persistent
    std::unordered_map<irr::u64, GELightState> m_states;
    unsigned m_frame;
    float m_dt;
    float m_time;
    float m_next_prune;
    uint64_t m_last_ms;
    bool m_has_last_ms;
    // First frame or after a hitch: fade is skipped
    bool m_snap;
    // ------------------------------------------------------------------------
    GELightStorage()
        : m_frame(0), m_dt(0.0f), m_time(0.0f), m_next_prune(0.0f),
          m_last_ms(0), m_has_last_ms(false), m_snap(true)
    {
    }   // GELightStorage
    // ------------------------------------------------------------------------
    void beginFrame()
    {
        uint64_t now = GE::getMonoTimeMs();
        float raw = m_has_last_ms ? (now - m_last_ms) / 1000.0f : 0.0f;
        m_snap = !m_has_last_ms || raw > LIGHT_SNAP_TIME;
        m_dt = std::min(raw, 0.1f);
        m_time += m_dt;
        m_last_ms = now;
        m_has_last_ms = true;
        m_frame++;
    }   // beginFrame
    // ------------------------------------------------------------------------
    GELightState* getState(irr::u64 id, bool no_fade)
    {
        auto ret = m_states.emplace(id, GELightState());
        // A new light fades in, unless everything is new (first frame) or
        // it doesn't fade at all
        if (ret.second)
            ret.first->second.m_weight = (m_snap || no_fade) ? 1.0f : 0.0f;
        // Can be changed on the node at any time
        ret.first->second.m_no_fade = no_fade;
        return &ret.first->second;
    }   // getState
    // ------------------------------------------------------------------------
    // Once per frame for each known light
    void advance(GELightState* s)
    {
        if (s->m_frame == m_frame)
            return;
        s->m_frame = m_frame;
        s->m_last_seen = m_time;
        const float target = s->m_selected ? 1.0f : 0.0f;
        if (m_snap || s->m_no_fade)
        {
            s->m_weight = target;
            return;
        }
        const float step = m_dt / LIGHT_FADE_TIME;
        if (s->m_weight < target)
            s->m_weight = std::min(target, s->m_weight + step);
        else
            s->m_weight = std::max(target, s->m_weight - step);
    }   // advance
    // ------------------------------------------------------------------------
    void prune()
    {
        if (m_time < m_next_prune)
            return;
        m_next_prune = m_time + LIGHT_PRUNE_INTERVAL;
        for (auto i = m_states.begin(); i != m_states.end();)
        {
            if (m_time - i->second.m_last_seen > LIGHT_FORGET_TIME)
                i = m_states.erase(i);
            else
                i++;
        }
    }   // prune
};

// ----------------------------------------------------------------------------
// Extra slots for lights that are fading out, see getMaxRenderingLights
unsigned GEVulkanLightHandler::getFadeExtraSlot()
{
    return getGEConfig()->m_max_pointlights / 8;
}   // getFadeExtraSlot

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
        sizeof(GELight) * getMaxRenderingLights();
}   // getMaxSize

// ----------------------------------------------------------------------------
unsigned GEVulkanLightHandler::getMaxRenderingLights()
{
    return getGEConfig()->m_max_pointlights + getFadeExtraSlot();
}   // getMaxRenderingLights

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
    // The first real frame starts without fading, the lights that are
    // already in the scene should not all fade in
    m_lights->m_has_last_ms = false;
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
    m_lights->beginFrame();
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

    // The fog can change every frame (animated by scripts), the buffer is
    // generated every frame from the driver so it's always up to date. Same
    // values (not in linear color space) as the OpenGL renderer
    if (m_vk->isFogEnabled())
    {
        video::SColor fog_color;
        video::E_FOG_TYPE fog_type;
        f32 start, end, max_fog;
        bool pixel_fog, range_fog;
        m_vk->getFog(fog_color, fog_type, start, end, max_fog, pixel_fog,
            range_fog);
        buffer->m_fog_density = 1.0f / (40.0f * (start + 0.001f));
        buffer->m_fog_color.r = fog_color.getRed() / 255.0f;
        buffer->m_fog_color.g = fog_color.getGreen() / 255.0f;
        buffer->m_fog_color.b = fog_color.getBlue() / 255.0f;
        buffer->m_fog_color.a = max_fog;
        buffer->m_fog_range[0] = start;
        buffer->m_fog_range[1] = end;
    }
}   // prepare

// ----------------------------------------------------------------------------
float GEVulkanLightHandler::getFogDensity() const
{
    return getGlobalLightPtr()->m_fog_density;
}   // getFogDensity

// ----------------------------------------------------------------------------
// Light order is the shadow slot order (a light has a point shadow if its
// index in the rendered array is below m_point_shadow_limit), so choose the
// lights in [first, last) that fill "slots" shadow slots and put them in
// front. Lights that already had a shadow are favored so the last slot
// doesn't alternate between two lights of similar distance every frame.
static void arrangeShadowLights(GELightEntry* first, GELightEntry* last,
                                unsigned slots)
{
    const size_t count = last - first;
    if (slots > count)
        slots = (unsigned)count;
    if (slots == 0 || slots == count)
        return;
    auto key = [](const GELightEntry& e)
    {
        const GELightState* s = e.m_state;
        // Lights fading out or barely faded in only get a slot if nothing
        // else is left
        if (!s->m_selected || s->m_weight < LIGHT_SHADOW_MIN_WEIGHT)
            return std::numeric_limits<float>::max();
        return s->m_shadow ?
            e.m_distance_sq * LIGHT_SHADOW_STICKY_DIST_SQ : e.m_distance_sq;
    };
    std::nth_element(first, first + slots, last,
        [&key](const GELightEntry& a, const GELightEntry& b)
        {
            return key(a) < key(b);
        });
}   // arrangeShadowLights

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::selectLights(const irr::core::vector3df& cam_pos)
{
    auto& data = m_lights->m_data;
    const unsigned max_lights = getGEConfig()->m_max_pointlights;

    for (GELightEntry& e : data)
        e.m_distance_sq = e.m_light.m_position.getDistanceFromSQ(cam_pos);

    // 1. Which lights are in the set. Previously selected lights are
    // favored (sticky), so a light at the cutoff (including spotlights)
    // doesn't flip in and out of the set while the camera moves.
    size_t selected_count = data.size();
    if (data.size() > max_lights)
    {
        auto key = [](const GELightEntry& e)
        {
            return e.m_state->m_selected ?
                e.m_distance_sq * LIGHT_STICKY_DIST_SQ : e.m_distance_sq;
        };
        std::nth_element(data.begin(), data.begin() + max_lights, data.end(),
            [&key](const GELightEntry& a, const GELightEntry& b)
            {
                return key(a) < key(b);
            });
        selected_count = max_lights;
    }

    // 2. Fade in the selected ones, fade out the rest
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i].m_state->m_selected = i < selected_count;
        m_lights->advance(data[i].m_state);
    }

    // 3. Lights that were just dropped from the set stay (after the selected
    // ones) while they fade out, up to the extra slots of the buffer
    size_t out = selected_count;
    for (size_t i = selected_count; i < data.size(); i++)
    {
        GELightState* s = data[i].m_state;
        if (s->m_weight > 0.0f && out - selected_count < getFadeExtraSlot())
        {
            if (out != i)
                std::swap(data[out], data[i]);
            out++;
        }
        else
            s->m_weight = 0.0f;
    }
    data.erase(data.begin() + out, data.end());

    // 4. Apply the weight, lights not visible yet are not rendered
    auto invisible = std::remove_if(data.begin(), data.end(),
        [](const GELightEntry& e) { return e.m_state->m_weight <= 0.0f; });
    data.erase(invisible, data.end());
    for (GELightEntry& e : data)
    {
        const float w = e.m_state->m_weight;
        if (w < 1.0f)
            e.m_light.m_color *= w;
    }

    m_lights->prune();
}   // selectLights

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
    selectLights(cam_pos);
    if (m_lights->m_data.empty())
        return;

    auto& data = m_lights->m_data;
    const bool point_shadow =
        (getGEConfig()->m_shadow_type & GST_POINTLIGHT) != 0;
    const unsigned shadow_limit =
        point_shadow ? getGEConfig()->m_point_shadow_limit : 0;

    // Lights in [shadow_first, shadow_last) compete for the point shadow
    // slots, the ones before shadow_first (fullscreen) always have a slot
    // while there is any, the ones from shadow_last (occluded) never do
    size_t shadow_first = 0;
    size_t shadow_last = data.size();
    bool set_occluded_light = false;
    GEVulkanFBOTexture* t =
        static_cast<GEVulkanDriver*>(getDriver())->getRTTTexture();
    const bool deferred = t && t->isDeferredFBO();
    if (deferred)
    {
        auto i = std::stable_partition(data.begin(), data.end(),
            [cam_pos](const GELightEntry& e)
        {
            const GELight& l = e.m_light;
            float radius_2 = l.m_radius * l.m_radius;
            float distance_2 = (cam_pos - l.m_position).getLengthSQ();
            return distance_2 <= radius_2;
        });
        m_fullscreen_light_count = std::distance(data.begin(), i);
        shadow_first = m_fullscreen_light_count;

        // Non-occluded lights before occluded ones. Fullscreen lights
        // (camera inside radius) are inherently non-occluded, so start the
        // partition from the iterator just past them.
        if (hasOcclusionCulling() && point_shadow)
        {
            auto j = std::stable_partition(i, data.end(),
                [cam_pos](const GELightEntry& e)
            {
                return !getOcclusionCulling()->isOccluded(
                    cam_pos, e.m_light.m_position, e.m_light.m_radius);
            });
            shadow_last = std::distance(data.begin(), j);
            m_non_occluded_lights = (unsigned)shadow_last;
            set_occluded_light = true;
        }
    }
    // Deferred fbo supports light culling using depth test
    const bool cull_in_cpu = hasOcclusionCulling() && !deferred;
    if (cull_in_cpu)
    {
        auto j = std::stable_partition(data.begin(), data.end(),
            [cam_pos](const GELightEntry& e)
        {
            return !getOcclusionCulling()->isOccluded(
                cam_pos, e.m_light.m_position, e.m_light.m_radius);
        });
        // Occluded lights are not rendered at all
        shadow_last = std::distance(data.begin(), j);
        data.erase(data.begin() + shadow_last, data.end());
    }

    // Choose the lights that have a point shadow and remember them for the
    // next frame
    arrangeShadowLights(data.data() + shadow_first, data.data() + shadow_last,
        shadow_limit > shadow_first ? shadow_limit - (unsigned)shadow_first : 0);
    for (size_t i = 0; i < data.size(); i++)
        data[i].m_state->m_shadow = i < shadow_last && i < shadow_limit;

    GELight* rl = getRenderingLightsPtr();
    for (const GELightEntry& e : data)
        *rl++ = e.m_light;
    getGlobalLightPtr()->m_light_count = (unsigned)data.size();
    if (cull_in_cpu || !set_occluded_light)
        m_non_occluded_lights = getLightCount();
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
        GELightEntry e = {};
        e.m_light = gl;
        e.m_state = m_lights->getState(node->getUniqueLightID(),
            node->isNoFadeInOutActive());
        m_lights->m_data.push_back(e);
    }
}   // addLightNode

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::touchLightNode(irr::scene::ILightSceneNode* node)
{
    if (node->getLightType() == irr::video::ELT_DIRECTIONAL)
        return;
    // Same fade as if it was submitted, so a light doesn't fade in again
    // every time it comes back to the view
    m_lights->advance(m_lights->getState(node->getUniqueLightID(),
            node->isNoFadeInOutActive()));
}   // touchLightNode

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
void GEVulkanLightHandler::setShadowSplit(
    const irr::core::vector3df& near_depths, float min_ray_cos)
{
    GEGlobalLightBuffer* buffer = getGlobalLightPtr();
    buffer->m_shadow_split[0] = near_depths.X;
    buffer->m_shadow_split[1] = near_depths.Y;
    buffer->m_shadow_split[2] = near_depths.Z;
    buffer->m_shadow_split[3] = min_ray_cos;
}   // setShadowSplit

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
