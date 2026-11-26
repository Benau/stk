struct SurfaceLighting
{
    vec4 local_irradiance;
    uint short_irradiance;
    float variance;
    float ray_tail;
    float bias;
    uint frame;
    uint buried;
    uint guide;
    uvec4 anchor;
};

layout(std430, set = SURFEL_SET, binding = 12) buffer SurfaceCache
{
    SurfaceLighting lighting[];
};
