#ifndef HEADER_MOTION_BLUR_HPP
#define HEADER_MOTION_BLUR_HPP

namespace MotionBlur
{
    // Preserve the existing tuning at 60 FPS and normalize other frame times
    // to the same 1/60 second camera displacement.
    inline float getFrameRateFactor(float dt)
    {
        return dt > 0.0f ? 1.0f / (60.0f * dt) : 1.0f;
    }
}

#endif // HEADER_MOTION_BLUR_HPP
