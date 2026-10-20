#include "states_screens/dialogs/ray_tracing_settings.hpp"

#include "config/user_config.hpp"
#include "guiengine/widgets/check_box_widget.hpp"
#include "guiengine/widgets/label_widget.hpp"
#include "guiengine/widgets/ribbon_widget.hpp"
#include "states_screens/options/options_screen_video.hpp"
#include "states_screens/state_manager.hpp"

#include <IGUIEnvironment.h>
#ifndef SERVER_ONLY
#include <ge_main.hpp>
#include <ge_vulkan_driver.hpp>
#include <ge_vulkan_features.hpp>
#endif

using namespace GUIEngine;
using namespace irr;

// -----------------------------------------------------------------------------

RayTracingSettingsDialog::RayTracingSettingsDialog(const float w,
                                                   const float h)
                        : ModalDialog(w, h)
{
    m_supported = false;
#ifndef SERVER_ONLY
    m_supported = GE::getVKDriver() != NULL &&
        GE::GEVulkanFeatures::supportsRayQuery() &&
        GE::getGEConfig()->m_pbr;
#endif
    loadFromFile("ray_tracing_settings.stkgui");
    updateActivation();
}   // RayTracingSettingsDialog

// -----------------------------------------------------------------------------

RayTracingSettingsDialog::~RayTracingSettingsDialog()
{
}   // ~RayTracingSettingsDialog

// -----------------------------------------------------------------------------

void RayTracingSettingsDialog::beforeAddingWidgets()
{
    getWidget<CheckBoxWidget>("rt_enabled")
        ->setState(UserConfigParams::m_ray_tracing);
    getWidget<CheckBoxWidget>("rt_reflection")
        ->setState(UserConfigParams::m_ray_tracing_reflection);
    getWidget<CheckBoxWidget>("rt_shadow")
        ->setState(UserConfigParams::m_ray_tracing_shadow);
    getWidget<CheckBoxWidget>("rt_dynamic_objects")
        ->setState(UserConfigParams::m_ray_tracing_dynamic_objects);
    getWidget<CheckBoxWidget>("rt_foliage")
        ->setState(UserConfigParams::m_ray_tracing_alphatest);
    getWidget<CheckBoxWidget>("rt_grass")
        ->setState(UserConfigParams::m_ray_tracing_grass);
}   // beforeAddingWidgets

// -----------------------------------------------------------------------------

void RayTracingSettingsDialog::updateActivation()
{
    CheckBoxWidget* enabled = getWidget<CheckBoxWidget>("rt_enabled");
    enabled->setActive(m_supported);
    bool on = m_supported && enabled->getState();
    getWidget<CheckBoxWidget>("rt_reflection")->setActive(on);
    getWidget<CheckBoxWidget>("rt_shadow")->setActive(on);
    getWidget<CheckBoxWidget>("rt_dynamic_objects")->setActive(on);
    getWidget<CheckBoxWidget>("rt_foliage")->setActive(on);
    getWidget<CheckBoxWidget>("rt_grass")->setActive(on);
    // The label explaining why it's not available
    getWidget<LabelWidget>("rt_unsupported")->setVisible(!m_supported);
}   // updateActivation

// -----------------------------------------------------------------------------

GUIEngine::EventPropagation RayTracingSettingsDialog::processEvent(
                                                const std::string& eventSource)
{
    if (eventSource == "rt_enabled")
    {
        updateActivation();
    }
    else if (eventSource == "buttons")
    {
        const std::string& selection = getWidget<RibbonWidget>("buttons")->
            getSelectionIDString(PLAYER_ID_GAME_MASTER);
        if (selection == "apply")
        {
#ifndef SERVER_ONLY
            // Without support the options are kept as they are
            if (m_supported)
            {
                const bool prev_enabled = UserConfigParams::m_ray_tracing;
                const bool prev_reflection =
                    UserConfigParams::m_ray_tracing_reflection;
                const bool prev_shadow =
                    UserConfigParams::m_ray_tracing_shadow;
                UserConfigParams::m_ray_tracing =
                    getWidget<CheckBoxWidget>("rt_enabled")->getState();
                UserConfigParams::m_ray_tracing_reflection =
                    getWidget<CheckBoxWidget>("rt_reflection")->getState();
                UserConfigParams::m_ray_tracing_shadow =
                    getWidget<CheckBoxWidget>("rt_shadow")->getState();
                UserConfigParams::m_ray_tracing_dynamic_objects =
                    getWidget<CheckBoxWidget>("rt_dynamic_objects")
                    ->getState();
                UserConfigParams::m_ray_tracing_alphatest =
                    getWidget<CheckBoxWidget>("rt_foliage")->getState();
                UserConfigParams::m_ray_tracing_grass =
                    getWidget<CheckBoxWidget>("rt_grass")->getState();
                user_config->saveConfig();
                OptionsScreenVideo::setRayTracing();

                // Dynamic objects, foliage and grass are handled by the ray traced
                // scene itself (it's built again if needed). The reflection
                // and the shadow only change which passes are rendered and the
                // pipelines (the shadow is ray traced or uses the shadow map
                // of GEConfig::m_shadow_size), which the draw calls created
                // again have.
                const bool shadow_changed =
                    prev_shadow != UserConfigParams::m_ray_tracing_shadow;
                const bool reflection_changed =
                    prev_reflection != UserConfigParams::m_ray_tracing_reflection;
                if (prev_enabled != UserConfigParams::m_ray_tracing ||
                    reflection_changed || shadow_changed)
                {
                    GE::getVKDriver()->updateDriver(reflection_changed ||
                        shadow_changed);
                }
            }
#endif
            ModalDialog::dismiss();
            return GUIEngine::EVENT_BLOCK;
        }
        else if (selection == "cancel")
        {
            ModalDialog::dismiss();
            return GUIEngine::EVENT_BLOCK;
        }
    }
    return GUIEngine::EVENT_LET;
}   // processEvent
