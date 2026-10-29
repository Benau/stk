#ifndef SERVER_ONLY

#include "graphics/ge_final_effects.hpp"

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

#include <cstring>

// ----------------------------------------------------------------------------
GEFinalEffects::GEFinalEffects()
{
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
        file_manager->getAsset(FileManager::TEXTURE,
        "stk_noise_simplex.png").c_str()).c_str());
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

    // The swapchain was created without them, if one of them is enabled it
    // has to be made again with the chain (nothing is done otherwise)
    GE::getVKDriver()->updateDriver(false/*scale_changed*/,
        false/*pbr_changed*/, false/*ibl_changed*/);
}   // GEFinalEffects

// ----------------------------------------------------------------------------
GEFinalEffects::~GEFinalEffects()
{
    GE::removeFinalPass(m_speed_streaks_handle);
}   // ~GEFinalEffects

// ----------------------------------------------------------------------------
void GEFinalEffects::prepareCamera(Camera* camera)
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
    }
    GE::setFinalPassUserData(camera->getCameraSceneNode(), data);
}   // prepareCamera

#endif   // !SERVER_ONLY
