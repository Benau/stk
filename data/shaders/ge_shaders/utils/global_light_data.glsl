struct LightData
{
    vec4  m_position_radius;
    vec4  m_color_inverse_square_range;
    vec4  m_direction_scale_offset; // Spotlight only
    mat4  m_shadow_projection_view_matrix[OMNI_FACES_PER_LIGHT];
};

layout(std140, set = 1, binding = 3) uniform GlobalLightBuffer
{
    mat4  m_shadow_projection_view_matrix[SUN_SHADOW_CASCADE_COUNT];
    mat4  m_shadow_view_matrix;
    vec3  m_ambient_color;
    float m_sun_scatter;
    vec3  m_sun_color;
    float m_sun_angle_tan_half;
    vec3  m_sun_direction;
    // Exponential fog of solid materials (deferred lighting) and the density
    // of the volumetric light scattering (light_scatter.comp), 0 = no fog
    float m_fog_density;
    // rgb: fog color, a: maximum amount of fog of transparent materials
    vec4  m_fog_color;
    vec3  m_skytop_color;
    int   m_light_count;
    // x: start, y: end (linear fog of transparent materials)
    // z: start height, w: end height of the volumetric fog (the density is 1
    // below the start and 0 above the end, see light_scatter.comp)
    vec4  m_fog_range;
    // View depth where each sun cascade starts (see GEVulkanShadowFBO::prepare)
    vec4  m_shadow_split;
    // x: cosine of the largest angle between the view direction and a frustum
    // ray (see GEVulkanShadowFBO::prepare)
    vec4  m_shadow_split_min_ray_cos;
    LightData m_lights[MAX_RENDERING_LIGHT];
} u_global_light;
