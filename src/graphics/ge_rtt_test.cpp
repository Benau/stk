#ifndef SERVER_ONLY

#include "graphics/ge_rtt_test.hpp"

#include "config/user_config.hpp"
#include "graphics/2dutils.hpp"
#include "graphics/camera/camera.hpp"
#include "graphics/irr_driver.hpp"
#include "karts/abstract_kart.hpp"
#include "utils/log.hpp"
#include "utils/vec3.hpp"

#include "ge_main.hpp"
#include "ge_vulkan_driver.hpp"

#include <algorithm>
#include <cstdlib>

using namespace irr;

// ----------------------------------------------------------------------------
GERTTTest::GERTTTest()
{
    m_texture = NULL;
    m_texture_size = core::dimension2du(0, 0);
    m_view_size = core::dimension2du(0, 0);
    m_rendered = false;
    m_reported_result = 0;
    m_frame_count = 0;
}   // GERTTTest

// ----------------------------------------------------------------------------
GERTTTest::~GERTTTest()
{
    reset();
}   // ~GERTTTest

// ----------------------------------------------------------------------------
GERTTTest* GERTTTest::getInstance()
{
    static GERTTTest test;
    return &test;
}   // getInstance

// ----------------------------------------------------------------------------
bool GERTTTest::isEnabled()
{
    static const bool enabled = getenv("STK_GE_RTT_TEST") != NULL &&
        strcmp(getenv("STK_GE_RTT_TEST"), "1") == 0;
    return enabled;
}   // isEnabled

// ----------------------------------------------------------------------------
void GERTTTest::reset()
{
    m_rendered = false;
    if (!m_texture)
        return;
    // The target may still be used by frames in flight
    if (GE::getDriver() &&
        GE::getDriver()->getDriverType() == video::EDT_VULKAN)
        GE::getVKDriver()->waitIdle();
    // GE doesn't add the rtt texture to the cache, so the user (us) owns it
    m_texture->drop();
    m_texture = NULL;
    m_texture_size = core::dimension2du(0, 0);
    m_view_size = core::dimension2du(0, 0);
}   // reset

// ----------------------------------------------------------------------------
/** Creates the RTT (3 views with the same aspect ratio as the screen next to
 *  each other, 1/8 of the screen width each, so the whole RTT is 3/8 of the
 *  screen width and 1/8 of the screen height), or recreate it if the screen
 *  size is changed. */
bool GERTTTest::prepareTexture(const core::dimension2du& screen)
{
    core::dimension2du view;
    view.Width = std::max(64u, screen.Width / 8);
    view.Height = std::max(36u, (unsigned)((uint64_t)view.Width *
        screen.Height / std::max(1u, screen.Width)));
    if (m_texture && view == m_view_size)
        return true;

    reset();
    m_view_size = view;
    m_texture_size = core::dimension2du(view.Width * 3, view.Height);
    m_texture = irr_driver->getVideoDriver()->addRenderTargetTexture(
        m_texture_size, "ge_rtt_test", video::ECF_A8R8G8B8);
    if (!m_texture)
    {
        Log::error("GERTTTest", "Cannot create the render target texture.");
        m_view_size = core::dimension2du(0, 0);
        return false;
    }
    return true;
}   // prepareTexture

// ----------------------------------------------------------------------------
void GERTTTest::report(bool passed, const char* what)
{
    const unsigned result = passed ? 1 : 2;
    if (m_reported_result == result)
        return;
    m_reported_result = result;
    if (passed)
        Log::info("GERTTTest", "PASS: all checks passed.");
    else
        Log::error("GERTTTest", "FAIL: %s", what);
}   // report

// ----------------------------------------------------------------------------
void GERTTTest::renderViews(Camera* camera)
{
    m_rendered = false;
    if (!camera)
        return;
    video::IVideoDriver* vd = irr_driver->getVideoDriver();
    AbstractKart* kart = camera->getKart();
    scene::ICameraSceneNode* node = camera->getCameraSceneNode();
    if (!kart || !node || Camera::getNumCameras() != 1)
        return;
    const core::dimension2du screen(UserConfigParams::m_width,
        UserConfigParams::m_height);
    if (!prepareTexture(screen))
        return;
    m_frame_count++;

    // State of the camera node (shared by all drawAll calls) which will be
    // restored, GEVulkanSceneManager must have copied what it needs in drawAll
    const core::vector3df saved_pos = node->getPosition();
    const core::vector3df saved_target = node->getTarget();
    const core::vector3df saved_up = node->getUpVector();
    const f32 saved_aspect = node->getAspectRatio();
    const core::recti saved_viewport = vd->getViewPort();

    // The kart's directions, up is always the world one for the test
    const core::vector3df kart_pos = kart->getXYZ().toIrrVector();
    core::vector3df forward = kart->getFrontXYZ().toIrrVector() - kart_pos;
    forward.Y = 0.0f;
    if (forward.getLengthSQ() < 0.0001f)
        forward = core::vector3df(0.0f, 0.0f, 1.0f);
    forward.normalize();
    const core::vector3df up(0.0f, 1.0f, 0.0f);
    // Irrlicht is left handed (x right, y up, z forward)
    const core::vector3df right = up.crossProduct(forward).normalize();
    const float height = std::max(kart->getKartHeight(), 0.5f);
    const float dist = std::max(kart->getKartLength() * 1.5f, 2.5f);
    const core::vector3df center = kart_pos + up * (height * 0.5f);
    const core::vector3df left_pos = center - right * dist + up * height * 0.5f;
    const core::vector3df right_pos = center + right * dist + up * height * 0.5f;

    // Same clear color as the default framebuffer, the viewport of the RTT is
    // the whole target after this
    vd->setRenderTarget(m_texture, true, true, irr_driver->getClearColor());

    enum View { VIEW_LEFT, VIEW_PLAYER, VIEW_RIGHT };
    // The player's one first, then the two more
    const View order[3] = { VIEW_PLAYER, VIEW_LEFT, VIEW_RIGHT };
    for (unsigned i = 0; i < 3; i++)
    {
        const View view = order[i];
        // 3 views side by side: left, player, right
        const int x = (int)(view * m_view_size.Width);
        vd->setViewPort(core::recti(x, 0, x + m_view_size.Width,
            m_view_size.Height));
        node->setAspectRatio((f32)m_view_size.Width / m_view_size.Height);
        if (view == VIEW_PLAYER)
        {
            // Unchanged camera position and target
        }
        else
        {
            node->setPosition(view == VIEW_LEFT ? left_pos : right_pos);
            node->setTarget(center);
            node->setUpVector(up);
            node->updateAbsolutePosition();
        }
        irr_driver->getSceneManager()->drawAll();
        if (view != VIEW_PLAYER)
        {
            // Back to the player's camera for the next drawAll
            node->setPosition(saved_pos);
            node->setTarget(saved_target);
            node->setUpVector(saved_up);
            node->updateAbsolutePosition();
        }
    }

    // The draw calls will be rendered in endScene
    vd->setRenderTarget(NULL, false, false);
    // setRenderTarget doesn't know about the aspect ratio
    node->setAspectRatio(saved_aspect);
    m_rendered = true;

    // Checks, the state of the camera must be the same as before
    bool ok = true;
    const char* error = "";
    if (!node->getPosition().equals(saved_pos) ||
        !node->getTarget().equals(saved_target) ||
        !node->getUpVector().equals(saved_up))
    {
        ok = false;
        error = "camera position, target or up vector is not restored";
    }
    else if (vd->getViewPort() != saved_viewport)
    {
        ok = false;
        error = "viewport is not restored by setRenderTarget(NULL)";
    }
    else if (left_pos.equals(right_pos))
    {
        // The 3 views must be different
        ok = false;
        error = "left and right views are at the same position";
    }
    report(ok, error);
}   // renderViews

// ----------------------------------------------------------------------------
void GERTTTest::drawResult()
{
    if (!m_texture || !m_rendered)
        return;
    m_rendered = false;
    const int screen_w = UserConfigParams::m_width;
    const int margin = std::max(4, screen_w / 100);
    const int border = std::max(1, screen_w / 640);
    const core::recti dest(screen_w - margin - (int)m_texture_size.Width,
        margin, screen_w - margin, margin + (int)m_texture_size.Height);

    video::IVideoDriver* vd = irr_driver->getVideoDriver();
    // A border and the dividers of the 3 views
    core::recti outline = dest;
    outline.UpperLeftCorner -= core::vector2di(border, border);
    outline.LowerRightCorner += core::vector2di(border, border);
    vd->draw2DRectangle(video::SColor(255, 255, 255, 255), outline);
    draw2DImage(m_texture, dest, core::recti(core::position2di(0, 0),
        core::dimension2di(m_texture_size.Width, m_texture_size.Height)),
        NULL/*clip*/, video::SColor(255, 255, 255, 255),
        false/*use_alpha_channel_of_texture*/);
    for (unsigned i = 1; i < 3; i++)
    {
        const int x = dest.UpperLeftCorner.X + (int)(i * m_view_size.Width);
        vd->draw2DRectangle(video::SColor(255, 255, 255, 255),
            core::recti(x - border / 2, dest.UpperLeftCorner.Y,
            x - border / 2 + border, dest.LowerRightCorner.Y));
    }
}   // drawResult

#endif
