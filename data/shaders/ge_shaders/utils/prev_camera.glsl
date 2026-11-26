// The frame cache (GEConfig::m_frame_cache): a pass keeps what it rendered in
// a history, and the next frame reuses it where its surface is still on the
// screen. A position of this frame is reprojected with the camera of the
// viewport in the last frame (PrevCamera, GEVulkanPrevCamera) to the pixel of
// the history which saw it, and the history keeps the surface of every pixel
// (packHistorySurface: its distance to that camera and its normal) so the
// reuse can check that it's the same surface, not the one of something which
// was in front of it or behind it, or the other side of a corner. Shared by
// the irradiance of the global illumination (utils/surfel_gi_apply.glsl) and
// the specular GI (specular_gi_temporal.comp).
//
// The includer declares the buffer of the cameras, and includes
// ../utils/decodeNormal.frag before it

// Same as GEVulkanPrevCamera (std430)
struct PrevCamera
{
    mat4 m_projection_view_matrix;
    mat4 m_inverse_projection_view_matrix;
    // The offset and the size of the viewport in pixels of the FBO (same as
    // the m_viewport of the camera)
    vec4 m_viewport;
    vec4 m_position;
};

// ----------------------------------------------------------------------------
// The pixel (in pixels of the FBO, not rounded) where the camera saw a world
// position, false if it's behind the camera. The inverse of getPosFromUVDepth
bool getPrevCameraPixel(PrevCamera camera, vec3 position, out vec2 pixel)
{
    pixel = vec2(0.0);
    vec4 clip = camera.m_projection_view_matrix * vec4(position, 1.0);
    if (clip.w <= 0.0)
        return false;
    pixel = camera.m_viewport.xy + (clip.xy / clip.w * 0.5 + 0.5) *
        camera.m_viewport.zw;
    return true;
}   // getPrevCameraPixel

// ----------------------------------------------------------------------------
// The world position at a distance from the camera along the ray of a pixel
// (in pixels of the FBO, the center of a pixel is at .5), the surface of a
// pixel of the history from its distance
vec3 getPrevCameraPosition(PrevCamera camera, vec2 pixel, float distance)
{
    vec2 ndc = (pixel - camera.m_viewport.xy) / camera.m_viewport.zw * 2.0 -
        1.0;
    // On the near plane (reverse z, the far plane can be at infinity)
    vec4 near = camera.m_inverse_projection_view_matrix * vec4(ndc, 1.0, 1.0);
    vec3 direction = normalize(near.xyz / near.w - camera.m_position.xyz);
    return camera.m_position.xyz + direction * distance;
}   // getPrevCameraPosition

// ----------------------------------------------------------------------------
// The surface of a pixel of a history: the distance to the camera (half float)
// and the normal of the g-buffer (8 bits for both of its components), never 0
// (the camera is not on the surface), so 0 is a pixel which can't be reused
uint packHistorySurface(float camera_distance, vec2 encoded_normal)
{
    uvec2 normal = uvec2(round(clamp(encoded_normal, 0.0, 1.0) * 255.0));
    return (packHalf2x16(vec2(camera_distance, 0.0)) & 0xFFFFu) |
        (normal.x << 16u) | (normal.y << 24u);
}   // packHistorySurface

// ----------------------------------------------------------------------------
float getHistorySurfaceDistance(uint packed)
{
    return unpackHalf2x16(packed & 0xFFFFu).x;
}   // getHistorySurfaceDistance

// ----------------------------------------------------------------------------
vec3 getHistorySurfaceNormal(uint packed)
{
    return DecodeNormal(vec2(uvec2(packed >> 16u, packed >> 24u) & 255u) /
        255.0);
}   // getHistorySurfaceNormal

// ----------------------------------------------------------------------------
// The pixel of the history saw the surface of a position (its distance to the
// camera and its normal): it's not hidden by something closer, it's not the
// pixel of another surface behind it, and it's not on the other side of a
// corner
bool isHistorySurface(uint packed, float camera_distance, vec3 normal)
{
    if (packed == 0u)
        return false;
    if (abs(getHistorySurfaceDistance(packed) - camera_distance) >
        camera_distance * 0.01 + 0.01)
        return false;
    return dot(getHistorySurfaceNormal(packed), normal) >= 0.9;
}   // isHistorySurface
