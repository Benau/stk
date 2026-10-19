#ifndef HEADER_RAY_TRACING_SETTINGS_HPP
#define HEADER_RAY_TRACING_SETTINGS_HPP

#include "guiengine/modaldialog.hpp"

/**
 * \brief Dialog that allows the player to select the ray tracing settings,
 *  which are mirrored to GE::GERayTracingConfig (Vulkan only, and the device
 *  needs to support ray query)
 * \ingroup states_screens
 */
class RayTracingSettingsDialog : public GUIEngine::ModalDialog
{
private:
    /** True if the current renderer and device can do ray tracing. */
    bool m_supported;

public:
    RayTracingSettingsDialog(const float percentWidth,
                             const float percentHeight);
    ~RayTracingSettingsDialog();

    virtual void beforeAddingWidgets();

    /** Enables the options which depend on the main ray tracing one. */
    void updateActivation();

    GUIEngine::EventPropagation processEvent(const std::string& eventSource);
};

#endif
