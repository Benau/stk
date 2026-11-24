// The test of a frustum, the culling data of the camera uniform
// buffer is the six planes of the frustum, then the min and max edge of the
// box of the camera (see GECullingTool::updateCullingData)
#define CULL_PLANE(i) u_camera.m_culling[i]
#define CULL_MIN_EDGE u_camera.m_culling[6].xyz
#define CULL_MAX_EDGE u_camera.m_culling[7].xyz

// ----------------------------------------------------------------------------
// Same tests as GECullingTool::isCulled, the box of the camera first
bool isInFrustum(const Source source)
{
    if (any(lessThan(source.max_edge, CULL_MIN_EDGE)) ||
        any(greaterThan(source.min_edge, CULL_MAX_EDGE)))
        return false;
    for (int i = 0; i < 6; i++)
    {
        vec4 plane = CULL_PLANE(i);
        // The corner of the box which is the furthest along the normal
        vec3 corner = mix(source.min_edge, source.max_edge,
            greaterThanEqual(plane.xyz, vec3(0.0)));
        if (dot(plane, vec4(corner, 1.0)) < 0.0)
            return false;
    }
    return true;
}   // isInFrustum
