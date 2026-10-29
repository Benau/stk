#ifndef HEADER_GE_FINAL_EFFECTS_HPP
#define HEADER_GE_FINAL_EFFECTS_HPP

#ifndef SERVER_ONLY

class Camera;

/** The post processing effects of the game which are passes of the final
 *  chain of GE Vulkan (see ge_final_pass.hpp): the speed streaks, which are
 *  the ones of the OpenGL renderer (PostProcessing) too. GE only knows what
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
    };

private:
    unsigned m_speed_streaks_handle;

public:
    // ------------------------------------------------------------------------
    /** Adds the passes to GE, and makes GE use the ones which are enabled. */
    GEFinalEffects();
    // ------------------------------------------------------------------------
    ~GEFinalEffects();
    // ------------------------------------------------------------------------
    /** Gives GE the data of the camera, call it before the drawAll() of
     *  the camera. */
    void prepareCamera(Camera* camera);
};   // GEFinalEffects

#endif   // !SERVER_ONLY
#endif
