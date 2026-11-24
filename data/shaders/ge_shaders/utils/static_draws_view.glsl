// The test of the view of a camera, which is its frustum
#include "static_draws_frustum.glsl"

// ----------------------------------------------------------------------------
bool isInView(const Source source)
{
    return isInFrustum(source);
}   // isInView
