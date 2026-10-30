#ifndef SERVER_ONLY

#include "graphics/ge_final_effects.hpp"

#include "graphics/motion_blur.hpp"
#include "config/user_config.hpp"
#include "graphics/camera/camera.hpp"
#include "graphics/irr_driver.hpp"
#include "graphics/speed_streaks.hpp"
#include "io/file_manager.hpp"
#include "modes/world.hpp"

#include <ge_final_pass.hpp>
#include <ge_main.hpp>
#include <ge_vulkan_driver.hpp>

#include <IFileSystem.h>

#include <algorithm>
#include <cstring>

// ----------------------------------------------------------------------------
GEFinalEffects::GEFinalEffects()
{
    // The streaks
    // The same noise as the OpenGL renderer (PostProcessing)
    GE::GEFinalPassInfo info;
    info.m_name = "speed_streaks";
    // Before the motion blur: the OpenGL renderer blurs the streaks too
    info.m_order = GE::GFPO_DEFAULT;
    info.m_needs_depth = false;
    info.m_main_target_only = true;
    info.m_enabled = []() { return UserConfigParams::m_speed_streaks; };

    GE::GEComputePassDesc desc;
    desc.m_shader = "speed_streaks.comp";
    desc.m_push_size = 3 * sizeof(float);
    desc.m_camera_set = false;
    desc.m_texture_paths.push_back(
        file_manager->getFileSystem()->getAbsolutePath(
        file_manager->getAsset(FileManager::SHADER,
        "stk_noise_simplex.jpg").c_str()).c_str());
    desc.m_is_active = [](const GE::GEFinalPassView& view)
    {
        return view.m_user_data.m_values[UD_STREAKS_INTENSITY] > 0.0f;
    };
    desc.m_fill = [](const GE::GEFinalPassView& view, void* out)
    {
        float values[3] =
        {
            view.m_user_data.m_values[UD_STREAKS_INTENSITY],
            view.m_user_data.m_values[UD_STREAKS_RADIAL_SCROLL_OFFSET],
            view.m_user_data.m_values[UD_TIME]
        };
        memcpy(out, values, sizeof(values));
    };
    m_speed_streaks_handle = GE::addComputeFinalPass(info, desc);

    // The motion blur, which reads the depth (so it needs the pbr shaders,
    // GE doesn't enable it without them) and is after the streaks (which it
    // blurs too, like OpenGL)
    GE::GEFinalPassInfo blur_info;
    blur_info.m_name = "motion_blur";
    blur_info.m_order = GE::GFPO_DEFAULT + 20;
    blur_info.m_needs_depth = true;
    blur_info.m_main_target_only = true;
    blur_info.m_enabled = []() { return UserConfigParams::m_motionblur; };

    GE::GEComputePassDesc blur_desc;
    blur_desc.m_shader = "motion_blur.comp";
    // The previous projection view matrix and the boost amount
    blur_desc.m_push_size = 16 * sizeof(float) + sizeof(float);
    blur_desc.m_camera_set = true;
    blur_desc.m_is_active = [](const GE::GEFinalPassView& view)
    {
        return view.m_user_data.m_values[UD_MOTION_BLUR_AMOUNT] > 0.0f;
    };
    blur_desc.m_fill = [](const GE::GEFinalPassView& view, void* out)
    {
        memcpy(out, view.m_prev_pvm.pointer(), 16 * sizeof(float));
        const float amount = view.m_user_data.m_values[UD_MOTION_BLUR_AMOUNT];
        memcpy((char*)out + 16 * sizeof(float), &amount, sizeof(float));
    };
    m_motion_blur_handle = GE::addComputeFinalPass(blur_info, blur_desc);

    // The swapchain was created without them, if one of them is enabled it
    // has to be made again with the chain (nothing is done otherwise)
    GE::getVKDriver()->updateDriver(false/*scale_changed*/,
        false/*pbr_changed*/, false/*ibl_changed*/);
}   // GEFinalEffects

// ----------------------------------------------------------------------------
GEFinalEffects::~GEFinalEffects()
{
    GE::removeFinalPass(m_speed_streaks_handle);
    GE::removeFinalPass(m_motion_blur_handle);
}   // ~GEFinalEffects

// ----------------------------------------------------------------------------
void GEFinalEffects::reset()
{
    m_boost_time.assign(Camera::getNumCameras(), 0.0f);
}   // reset

// ----------------------------------------------------------------------------
void GEFinalEffects::giveBoost(unsigned int camera_index)
{
    if (camera_index >= m_boost_time.size())
        m_boost_time.resize(camera_index + 1, 0.0f);
    // The same as PostProcessing
    m_boost_time[camera_index] = 0.75f;
}   // giveBoost

// ----------------------------------------------------------------------------
void GEFinalEffects::update(float dt)
{
    for (float& boost_time : m_boost_time)
    {
        if (boost_time > 0.0f)
            boost_time = std::max(boost_time - dt, 0.0f);
    }
}   // update

// ----------------------------------------------------------------------------
void GEFinalEffects::prepareCamera(Camera* camera, float dt)
{
    GE::GECameraUserData data;
    World* world = World::getWorld();
    if (world)
    {
        float intensity = 0.0f;
        float radial_scroll_offset = 0.0f;
        // Not set when the streaks are off: the state of the kart is updated
        // anyway
        if (UserConfigParams::m_speed_streaks)
        {
            SpeedStreaks::getCameraParams(camera, &intensity,
                &radial_scroll_offset);
        }
        data.m_values[UD_STREAKS_INTENSITY] = intensity;
        data.m_values[UD_STREAKS_RADIAL_SCROLL_OFFSET] = radial_scroll_offset;
        data.m_values[UD_TIME] = world->getTime();

        // Same as PostProcessing::renderMotionBlur
        const unsigned index = camera->getIndex();
        if (UserConfigParams::m_motionblur && index < m_boost_time.size())
            data.m_values[UD_MOTION_BLUR_AMOUNT] = m_boost_time[index] *
                10.0f * MotionBlur::getFrameRateFactor(dt);
    }
    GE::setFinalPassUserData(camera->getCameraSceneNode(), data);
}   // prepareCamera

#endif   // !SERVER_ONLY
