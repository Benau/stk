#ifndef HEADER_GE_RTT_TEST_HPP
#define HEADER_GE_RTT_TEST_HPP

#ifndef SERVER_ONLY

#include <irrlicht.h>

class Camera;

/** Manual test for the target groups of GEVulkanSceneManager (several drawAll
 *  calls for the same separate render target in the same frame) with the fixed
 *  pipeline renderer and a single player.
 *
 *  It is only active when the environment variable STK_GE_RTT_TEST is set, so
 *  it has no effect on normal runs. For example:
 *      STK_GE_RTT_TEST=1 ./supertuxkart
 *
 *  Every frame, the camera scene node of the player is reused for 3 drawAll
 *  calls which are rendered to an offscreen RTT with 3 viewports side by side:
 *      [ left side of kart | view of the player | right side of kart ]
 *  and then the RTT is drawn at the top right corner of the screen. The
 *  default framebuffer is still rendered normally with its own drawAll.
 *
 *  What should be seen:
 *  - 3 different views in the RTT (if the camera snapshot is wrong, all 3
 *    would show the same view, the last one of the camera)
 *  - no flickering, no validation layer errors when resizing the window, when
 *    switching between the deferred types or when leaving the race
 *  - the main view is not changed by the RTT
 *
 *  Some checks (state of the camera node and the viewport are restored) are
 *  also done every frame, the result is logged once (and again if it changes).
 */
class GERTTTest
{
private:
    irr::video::ITexture* m_texture;

    irr::core::dimension2du m_texture_size;

    // Size of one of the 3 views in the RTT
    irr::core::dimension2du m_view_size;

    // The RTT is only drawn if it was rendered in this frame
    bool m_rendered;

    // 0: no result yet, 1: all checks passed, 2: some checks failed
    unsigned m_reported_result;

    unsigned m_frame_count;
    // ------------------------------------------------------------------------
    GERTTTest();
    // ------------------------------------------------------------------------
    ~GERTTTest();
    // ------------------------------------------------------------------------
    bool prepareTexture(const irr::core::dimension2du& screen);
    // ------------------------------------------------------------------------
    void report(bool passed, const char* what);

public:
    // ------------------------------------------------------------------------
    static GERTTTest* getInstance();
    // ------------------------------------------------------------------------
    static bool isEnabled();
    // ------------------------------------------------------------------------
    /** Call after the normal drawAll() of the (only) player camera, with this
     *  camera still the active one. Makes 3 more drawAll() calls with the
     *  same camera scene node but for the offscreen RTT. */
    void renderViews(Camera* camera);
    // ------------------------------------------------------------------------
    /** Call when the viewport is the whole screen (like for the race gui), it
     *  draws the RTT at the top right corner. */
    void drawResult();
    // ------------------------------------------------------------------------
    /** Destroys the RTT, call when the world is unloaded. */
    void reset();

};   // GERTTTest

#endif
#endif
