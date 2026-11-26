vec4 getSurfelSpecularGI(vec4 irradiance)
{
    float coverage = clamp(irradiance.a, 0.0, 1.0);
    return vec4(irradiance.rgb * coverage, coverage);
}

vec3 resolveSpecularGISky(vec3 sky, vec3 position, vec3 normal, vec3 direction)
{
#ifdef SURFEL_GI_FORWARD
    if (u_surfel_gi)
    {
        float visibility;
        vec4 local = lookupVolume(position, normal, true, direction, visibility);
        return mix(local.rgb, sky, clamp(visibility, 0.0, 1.0));
    }
#endif
    return sky;
}
