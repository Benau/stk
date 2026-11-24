// The test of an omni shadow view: the sphere of at least one of the lights
// of the bucket of the draw call has to touch the box, the frustum is not
// used. Same as GECullingTool::isCulled, the culling data is the position and
// the squared radius of each light, which is negative if there is no light
#define CULL_LIGHT(i) u_camera.m_culling[i]

// ----------------------------------------------------------------------------
bool isInView(const Source source)
{
    for (int i = 0; i < OMNI_LIGHTS_PER_BUCKET; i++)
    {
        vec4 light = CULL_LIGHT(i);
        // The closest point of the box to the center of the light
        vec3 closest = clamp(light.xyz, source.min_edge, source.max_edge);
        vec3 delta = light.xyz - closest;
        if (dot(delta, delta) <= light.w)
            return true;
    }
    return false;
}   // isInView
