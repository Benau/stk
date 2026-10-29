#ifndef HEADER_GE_FINAL_EFFECTS_HPP
#define HEADER_GE_FINAL_EFFECTS_HPP

#ifndef SERVER_ONLY

#include <vector>

class Camera;

/** The post processing effects of the game which are passes of the final
 *  chain of GE Vulkan (see ge_final_pass.hpp): the speed streaks and the
 *  motion blur, which are the ones of the OpenGL renderer (PostProcessing)
 *  too. GE only knows what
 *  the passes ask for, what they show and when is decided here: the settings
 *  of the user (UserConfigParams, which GEVulkanDriver::updateDriver() has to
 *  be called after a change of), and the data of every camera which is given
 *  to GE before its drawAll().
 */
class GEFinalEffects
{
public:
    /** The meaning of the values of GE::GECameraUserData of a camera, which
     *  the shaders get as push constants (see the m_fill of the passes). */
    enum UserData
    {
        UD_STREAKS_INTENSITY = 0,
        UD_STREAKS_RADIAL_SCROLL_OFFSET,
        UD_TIME,
        UD_MOTION_BLUR_AMOUNT,
    };

private:
    unsigned m_speed_streaks_handle;

    unsigned m_motion_blur_handle;

    /** How long the motion blur of every camera (index of the camera) lasts
     *  still, like PostProcessing::m_boost_time. */
    std::vector<float> m_boost_time;

public:
    // ------------------------------------------------------------------------
    /** Adds the passes to GE, and makes GE use the ones which are enabled. */
    GEFinalEffects();
    // ------------------------------------------------------------------------
    ~GEFinalEffects();
    // ------------------------------------------------------------------------
    /** Called before a race is started, after all cameras are set up. */
    void reset();
    // ------------------------------------------------------------------------
    /** The kart of the camera is boosted: it has the motion blur for a while
     *  (if it's enabled). */
    void giveBoost(unsigned int camera_index);
    // ------------------------------------------------------------------------
    /** Updates the boost times of all cameras, once per frame. */
    void update(float dt);
    // ------------------------------------------------------------------------
    /** Gives GE the data of the camera, call it before the drawAll() of
     *  the camera. */
    void prepareCamera(Camera* camera);
};   // GEFinalEffects

#endif   // !SERVER_ONLY
#endif
