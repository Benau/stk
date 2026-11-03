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
#include <cstring>
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
    // View depth where each sun cascade starts (see setShadowSplit)
    irr::f32             m_shadow_split[4];
    // x: cosine of the largest angle between the view direction and a frustum
    // ray (see setShadowSplit)
    irr::f32             m_shadow_split_min_ray_cos[4];
    //std::array<GELight, MAX_RENDERING_LIGHT> m_rendering_lights;
};

// ----------------------------------------------------------------------------
// Light clusters (same algorithm as the GEClusterDataGenerator of the
// upstream commit, with the lights of a depth slice stored only once):
// the screen is split in TILE_SIZE tiles on x and y, the view distance in
// DEPTH_DIVISIONS slices uniform in sqrt(distance). A light is put in every
// depth slice its sphere reaches, each slice stores the list of its lights and
// a bit per light (not per global light) in the bitmaps of the x strips and y
// strips. A tile tests the x strip and y strip bitmaps of its depth slice, so
// their intersection is a conservative set of the lights of the tile.
//
// Mirrors the GLSL "LightClusterBuffer" of utils/light_cluster.glsl, the data
// after the header is an array of u32: the light indices of every slice, then
// for every slice the x strip words followed by the y strip words.
static const irr::u32 CLUSTER_TILE_SIZE = 16;
static const irr::u32 CLUSTER_DEPTH_DIVISIONS = 64;

struct GELightClusterHeader
{
    irr::u32 m_depth_divisions;
    irr::u32 m_x_tiles;
    irr::u32 m_y_tiles;
    irr::u32 m_light_index_count;
    // slice = (sqrt(distance) - m_kernel_near) * m_kernel_scale
    irr::f32 m_kernel_near;
    irr::f32 m_kernel_scale;
    irr::f32 m_kernel_far;
    irr::u32 m_reserved;
    // Offset of the light indices of a slice in the u32 data after the header
    irr::u32 m_slice_light_offset[CLUSTER_DEPTH_DIVISIONS];
    // Offset of the bitmap of a slice in the u32 data after the header
    irr::u32 m_slice_bitmap_offset[CLUSTER_DEPTH_DIVISIONS];
    // Words of each strip bitmap (x strips, then y strips) of a slice
    irr::u32 m_slice_word_count[CLUSTER_DEPTH_DIVISIONS];
};
static_assert(sizeof(GELightClusterHeader) == 800,
    "the header must match utils/light_cluster.glsl");

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
size_t GEVulkanLightHandler::getMaxLightBlockSize()
{
    return sizeof(GEGlobalLightBuffer) +
        sizeof(GELight) * getMaxRenderingLights();
}   // getMaxLightBlockSize

// ----------------------------------------------------------------------------
size_t GEVulkanLightHandler::getClusterOffset()
{
    // 256 is at least the storage and uniform buffer offset alignments of any
    // device, so the light block and the cluster block after it are aligned
    // for their dynamic offsets
    return getMaxLightBlockSize() +
        getPadding(getMaxLightBlockSize(), 256);
}   // getClusterOffset

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
// Upper bound of the cluster block: every slice can have all the lights, and
// every strip bitmap of a slice needs a word per 32 lights
static size_t getClusterMaxSize(unsigned x_tiles, unsigned y_tiles)
{
    const irr::u32 lights = std::max(
        GEVulkanLightHandler::getMaxRenderingLights(), (unsigned)1);
    const irr::u32 words = (lights + 31) / 32;
    return sizeof(GELightClusterHeader) + sizeof(irr::u32) *
        (CLUSTER_DEPTH_DIVISIONS * lights +
         (x_tiles + y_tiles) * CLUSTER_DEPTH_DIVISIONS * words);
}   // getClusterMaxSize

// ----------------------------------------------------------------------------
GEVulkanLightHandler::GEVulkanLightHandler(GEVulkanDriver* vk)
{
    m_vk = vk;
    m_lights.reset(new GELightStorage);
    prepare(NULL);
    // The first real frame starts without fading, the lights that are
    // already in the scene should not all fade in
    m_lights->m_has_last_ms = false;
}   // GEVulkanLightHandler

// ----------------------------------------------------------------------------
GEVulkanLightHandler::~GEVulkanLightHandler()
{
}   // ~GEVulkanLightHandler

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::initClusterCamera(GEVulkanCamera* cam)
{
    const irr::scene::SViewFrustum* frustum = cam->getViewFrustum();
    m_cluster_view_matrix = cam->getViewMatrix();
    m_cluster_far_left_down = frustum->getFarLeftDown();
    m_cluster_far_right_up = frustum->getFarRightUp();
    m_cluster_view_matrix.transformVect(m_cluster_far_left_down);
    m_cluster_view_matrix.transformVect(m_cluster_far_right_up);
    m_cluster_near = cam->getNearValue();
    const irr::core::rectf& viewport = cam->getUBOData()->m_viewport;
    m_cluster_screen_size.X = viewport.getWidth();
    m_cluster_screen_size.Y = viewport.getHeight();
    // Orthographic cameras (the track graph) and the camera-snapshot warm-up
    // have no perspective frustum to split into strips, they don't need
    // clusters because nothing they draw is lit by point lights
    if (m_cluster_screen_size.X < 1.0f || m_cluster_screen_size.Y < 1.0f ||
        m_cluster_near <= 0.0f || m_cluster_far_left_down.Z <= 0.0f ||
        m_cluster_far_right_up.Z <= 0.0f)
        return;

    const float width = m_cluster_screen_size.X;
    const float height = m_cluster_screen_size.Y;
    // The x strips go from the left of the screen to the right, the y strips
    // from the bottom to the top (view space)
    const unsigned x_tiles = (unsigned)ceilf(width / CLUSTER_TILE_SIZE);
    const unsigned y_tiles = (unsigned)ceilf(height / CLUSTER_TILE_SIZE);
    for (unsigned i = 0; i < x_tiles; i++)
    {
        // The frustum corners are in the order of vector3d::getInterpolated:
        // d = 0 is the left, d = 1 the right
        irr::core::vector3df ray =
            m_cluster_far_right_up.getInterpolated(m_cluster_far_left_down,
                (float)(i * CLUSTER_TILE_SIZE) / width);
        irr::core::vector3df plane(-ray.Z, 0.0f, ray.X);
        if (plane.getLengthSQ() <= 1e-12f)
            return;
        m_cluster_x_planes.push_back(plane.normalize());
    }
    irr::core::vector3df plane(-m_cluster_far_right_up.Z, 0.0f,
        m_cluster_far_right_up.X);
    if (plane.getLengthSQ() <= 1e-12f)
        return;
    m_cluster_x_planes.push_back(plane.normalize());

    for (unsigned i = 0; i < y_tiles; i++)
    {
        // Only y and z are used, the z of the two far corners is the same so
        // the ray moves from the bottom of the screen to the top
        irr::core::vector3df ray =
            m_cluster_far_right_up.getInterpolated(m_cluster_far_left_down,
                (float)(i * CLUSTER_TILE_SIZE) / height);
        irr::core::vector3df y_plane(0.0f, -ray.Z, ray.Y);
        if (y_plane.getLengthSQ() <= 1e-12f)
            return;
        m_cluster_y_planes.push_back(y_plane.normalize());
    }
    plane.set(0.0f, -m_cluster_far_right_up.Z, m_cluster_far_right_up.Y);
    if (plane.getLengthSQ() <= 1e-12f)
        return;
    m_cluster_y_planes.push_back(plane.normalize());

    m_cluster_camera_valid = true;
}   // initClusterCamera

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::generateClusters()
{
    GELightClusterHeader* header =
        (GELightClusterHeader*)(m_buffer.data() + m_cluster_offset);
    *header = {};
    header->m_depth_divisions = CLUSTER_DEPTH_DIVISIONS;
    m_cluster_size = sizeof(GELightClusterHeader);

    const unsigned light_count = getLightCount();
    if (!m_cluster_camera_valid || light_count == 0)
        return;

    const unsigned x_tiles = (unsigned)m_cluster_x_planes.size() - 1;
    const unsigned y_tiles = (unsigned)m_cluster_y_planes.size() - 1;
    if (x_tiles == 0 || y_tiles == 0)
        return;

    GELight* lights = getRenderingLightsPtr();
    std::vector<irr::core::vector3df> positions(light_count);
    std::vector<float> distances(light_count);
    float far = 0.0f;
    for (unsigned i = 0; i < light_count; i++)
    {
        positions[i] = lights[i].m_position;
        m_cluster_view_matrix.transformVect(positions[i]);
        distances[i] = positions[i].getLength();
        far = std::max(far, distances[i] + lights[i].m_radius);
    }

    const float near = std::max(m_cluster_near, 0.001f);
    if (!(far > near) || !std::isfinite(far))
        far = near + 1.0f;
    // Slices are uniform in sqrt(distance), like the upstream generator
    const float kernel_near = sqrtf(near);
    const float kernel_scale =
        (float)CLUSTER_DEPTH_DIVISIONS / (sqrtf(far) - kernel_near);

    header->m_x_tiles = x_tiles;
    header->m_y_tiles = y_tiles;
    header->m_kernel_near = kernel_near;
    header->m_kernel_scale = kernel_scale;
    header->m_kernel_far = far;

    auto get_slice_range = [&](float n, float f, int& first, int& last)
    {
        if (f < near || n > far)
            return false;
        n = std::max(n, near);
        f = std::min(f, far);
        first = (int)((sqrtf(std::max(n, 0.0f)) - kernel_near) *
            kernel_scale);
        last = (int)((sqrtf(std::max(f, 0.0f)) - kernel_near) *
            kernel_scale);
        first = std::min(std::max(first, 0),
            (int)CLUSTER_DEPTH_DIVISIONS - 1);
        last = std::min(std::max(last, 0),
            (int)CLUSTER_DEPTH_DIVISIONS - 1);
        return first <= last;
    };

    // 1. The lights of every slice, a light is in a slice if its sphere
    // reaches the slice in the radial distance
    std::vector<irr::u32> slice_counts(CLUSTER_DEPTH_DIVISIONS, 0);
    std::vector<irr::u32> slice_lights(
        CLUSTER_DEPTH_DIVISIONS * light_count, 0);
    for (unsigned i = 0; i < light_count; i++)
    {
        int first, last;
        if (!get_slice_range(distances[i] - lights[i].m_radius,
            distances[i] + lights[i].m_radius, first, last))
            continue;
        for (int s = first; s <= last; s++)
            slice_lights[s * light_count + slice_counts[s]++] = i;
    }

    // Position of a light in the compact list of a slice, 0xffffffff if the
    // light isn't in the slice, the bitmaps of the slice only have a bit per
    // light of the slice
    std::vector<irr::u32> local_index(
        CLUSTER_DEPTH_DIVISIONS * light_count, 0xffffffff);
    for (unsigned s = 0; s < CLUSTER_DEPTH_DIVISIONS; s++)
    {
        for (unsigned k = 0; k < slice_counts[s]; k++)
            local_index[s * light_count + slice_lights[s * light_count + k]] =
                k;
    }

    std::vector<irr::u32> light_offsets(CLUSTER_DEPTH_DIVISIONS);
    std::vector<irr::u32> bitmap_offsets(CLUSTER_DEPTH_DIVISIONS);
    std::vector<irr::u32> word_counts(CLUSTER_DEPTH_DIVISIONS);
    irr::u32 data_words = 0;
    for (unsigned s = 0; s < CLUSTER_DEPTH_DIVISIONS; s++)
    {
        light_offsets[s] = data_words;
        data_words += slice_counts[s];
    }
    const irr::u32 light_index_count = data_words;
    for (unsigned s = 0; s < CLUSTER_DEPTH_DIVISIONS; s++)
    {
        word_counts[s] = (slice_counts[s] + 31) / 32;
        bitmap_offsets[s] = data_words;
        data_words += (x_tiles + y_tiles) * word_counts[s];
    }

    m_cluster_size = sizeof(GELightClusterHeader) +
        sizeof(irr::u32) * data_words;
    if (m_buffer.size() < m_cluster_offset + m_cluster_size)
        m_buffer.resize(m_cluster_offset + m_cluster_size);
    header = (GELightClusterHeader*)(m_buffer.data() + m_cluster_offset);
    irr::u32* data = (irr::u32*)(m_buffer.data() + m_cluster_offset +
        sizeof(GELightClusterHeader));

    for (unsigned s = 0; s < CLUSTER_DEPTH_DIVISIONS; s++)
    {
        header->m_slice_light_offset[s] = light_offsets[s];
        header->m_slice_bitmap_offset[s] = bitmap_offsets[s];
        header->m_slice_word_count[s] = word_counts[s];
        for (unsigned k = 0; k < slice_counts[s]; k++)
            data[light_offsets[s] + k] =
                slice_lights[s * light_count + k];
    }
    header->m_light_index_count = light_index_count;
    memset(data + light_index_count, 0,
        sizeof(irr::u32) * (data_words - light_index_count));

    // Set the bit of a light in the bitmap of a strip for the slices
    auto mark = [&](bool x_axis, unsigned light, unsigned strip, int first,
                    int last)
    {
        for (int s = first; s <= last; s++)
        {
            const irr::u32 index =
                local_index[s * light_count + light];
            // Possible with the rounding of a chord touching the border of a
            // slice the light isn't in
            if (index == 0xffffffff)
                continue;
            const irr::u32 offset = bitmap_offsets[s] +
                (x_axis ? 0 : x_tiles * word_counts[s]) +
                strip * word_counts[s];
            data[offset + (index >> 5)] |= 1u << (index & 31);
        }
    };
    // Mark the slices of a chord of the sphere through a plane of the strip
    auto mark_chord = [&](bool x_axis, unsigned light,
                          const irr::core::vector3df& point, float distance,
                          float radius, unsigned strip,
                          const irr::core::vector3df& plane)
    {
        const float plane_distance = fabsf(plane.dotProduct(point));
        if (plane_distance > radius)
            return;
        const float chord_distance = sqrtf(std::max(
            distance * distance - plane_distance * plane_distance, 0.0f));
        const float chord_radius = sqrtf(std::max(
            radius * radius - plane_distance * plane_distance, 0.0f));
        int first, last;
        if (get_slice_range(chord_distance - chord_radius,
            chord_distance + chord_radius, first, last))
            mark(x_axis, light, strip, first, last);
    };

    // 2. The strips of every light, along x or y. The strips of the
    // projection of the center use the whole radial range of the sphere, the
    // strips next to it the chord of the sphere through their closest plane
    auto fill_axis = [&](bool x_axis,
                         const std::vector<irr::core::vector3df>& planes)
    {
        const unsigned strips = x_axis ? x_tiles : y_tiles;
        const float far_size = x_axis ?
            m_cluster_far_right_up.X - m_cluster_far_left_down.X :
            m_cluster_far_right_up.Y - m_cluster_far_left_down.Y;
        const float far_min = x_axis ?
            m_cluster_far_left_down.X : m_cluster_far_left_down.Y;
        const float far_z = m_cluster_far_left_down.Z;
        for (unsigned i = 0; i < light_count; i++)
        {
            const irr::core::vector3df& point = positions[i];
            const float radius = lights[i].m_radius;
            const float distance = distances[i];
            int first, last;
            // The camera is inside the sphere, it reaches every strip
            if (distance <= radius)
            {
                if (get_slice_range(distance - radius, distance + radius,
                    first, last))
                {
                    for (unsigned s = 0; s < strips; s++)
                        mark(x_axis, i, s, first, last);
                }
                continue;
            }
            if (point.Z > 0.0f &&
                planes.front().dotProduct(point) < 0.0f &&
                planes.back().dotProduct(point) > 0.0f)
            {
                // The center is in the frustum, project it to the far plane
                const float center_coord = (x_axis ? point.X : point.Y) *
                    (far_z / point.Z);
                const float t = (center_coord - far_min) / far_size;
                const int center = std::min(std::max(
                    (int)(t * (float)strips), 0), (int)strips - 1);
                if (get_slice_range(distance - radius, distance + radius,
                    first, last))
                    mark(x_axis, i, center, first, last);
                for (int s = center - 1; s >= 0; s--)
                    mark_chord(x_axis, i, point, distance, radius, s,
                        planes[s + 1]);
                for (int s = center + 1; s < (int)strips; s++)
                    mark_chord(x_axis, i, point, distance, radius, s,
                        planes[s]);
            }
            else
            {
                // The center is out of the frustum, it can only reach the
                // strips of the boundary plane it is close enough to
                const bool left =
                    fabsf(planes.front().dotProduct(point)) < radius;
                const bool right =
                    fabsf(planes.back().dotProduct(point)) < radius;
                if (left)
                {
                    for (int s = 0; s < (int)strips; s++)
                        mark_chord(x_axis, i, point, distance, radius, s,
                            planes[s]);
                }
                else if (right)
                {
                    for (int s = (int)strips - 1; s >= 0; s--)
                        mark_chord(x_axis, i, point, distance, radius, s,
                            planes[s + 1]);
                }
            }
        }
    };
    fill_axis(true, m_cluster_x_planes);
    fill_axis(false, m_cluster_y_planes);
}   // generateClusters

// ----------------------------------------------------------------------------
void GEVulkanLightHandler::prepare(GEVulkanCamera* cam)
{
    m_lights->m_data.clear();
    m_lights->beginFrame();
    m_fullscreen_light_count = 0;
    m_non_occluded_lights = 0;

    m_cluster_camera_valid = false;
    m_cluster_x_planes.clear();
    m_cluster_y_planes.clear();
    m_cluster_offset = getClusterOffset();
    m_cluster_size = sizeof(GELightClusterHeader);
    size_t buffer_size = m_cluster_offset + getClusterMaxSize(0, 0);
    if (cam)
    {
        initClusterCamera(cam);
        const unsigned x_tiles = m_cluster_x_planes.empty() ? 0 :
            (unsigned)m_cluster_x_planes.size() - 1;
        const unsigned y_tiles = m_cluster_y_planes.empty() ? 0 :
            (unsigned)m_cluster_y_planes.size() - 1;
        buffer_size = m_cluster_offset + getClusterMaxSize(x_tiles, y_tiles);
    }
    m_buffer.resize(buffer_size);
    // No lights yet, the shader has nothing to loop
    GELightClusterHeader* cluster =
        (GELightClusterHeader*)(m_buffer.data() + m_cluster_offset);
    *cluster = {};
    cluster->m_depth_divisions = CLUSTER_DEPTH_DIVISIONS;

    GEGlobalLightBuffer* buffer = getGlobalLightPtr();
    *buffer = {};
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
        // The height fog of the volumetric light scattering
        // (light_scatter.comp): the density is faded from 1 at this start
        // height to 0 at the end height, it's constant if the end is not above
        // the start (the default of the tracks without height fog)
        buffer->m_fog_range[2] = m_vk->getFogHeightStart();
        buffer->m_fog_range[3] = m_vk->getFogHeightEnd();
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
    generateClusters();
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
    const float* near_depths, float min_ray_cos)
{
    GEGlobalLightBuffer* buffer = getGlobalLightPtr();
    for (unsigned i = 0; i < 4; i++)
        buffer->m_shadow_split[i] = near_depths[i];
    buffer->m_shadow_split_min_ray_cos[0] = min_ray_cos;
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
