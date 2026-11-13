//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2009-2015 Marianne Gagnon
//
//  This program is free software; you can redistribute it and/or
//  modify it under the terms of the GNU General Public License
//  as published by the Free Software Foundation; either version 3
//  of the License, or (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

#include "states_screens/dialogs/custom_video_settings.hpp"

#include "config/user_config.hpp"
#include "graphics/graphical_presets.hpp"
#include "guiengine/widgets/check_box_widget.hpp"
#include "guiengine/widgets/ribbon_widget.hpp"
#include "guiengine/widgets/spinner_widget.hpp"
#include "states_screens/options/options_screen_video.hpp"
#include "states_screens/state_manager.hpp"
#include "graphics/central_settings.hpp"
#include "graphics/irr_driver.hpp"
#include "utils/string_utils.hpp"
#include "utils/translation.hpp"

#include <algorithm>

#include <IGUIEnvironment.h>
#ifndef SERVER_ONLY
#include <ge_main.hpp>
#include <ge_vulkan_driver.hpp>
#endif

using namespace GUIEngine;
using namespace irr;
using namespace irr::core;
using namespace irr::gui;

// -----------------------------------------------------------------------------

CustomVideoSettingsDialog::CustomVideoSettingsDialog(const float w, const float h) :
        ModalDialog(w, h)
{
    loadFromFile("custom_video_settings.stkgui");
    getWidget<RibbonWidget>("settings_tabs")->select("tab_lighting",
        PLAYER_ID_GAME_MASTER);
    showTab("tab_lighting");
    updateActivation();
}

// -----------------------------------------------------------------------------

CustomVideoSettingsDialog::~CustomVideoSettingsDialog()
{
}

// -----------------------------------------------------------------------------

void CustomVideoSettingsDialog::beforeAddingWidgets()
{
#ifndef SERVER_ONLY
    getWidget<CheckBoxWidget>("animated_characters")
        ->setState(UserConfigParams::m_animated_characters);
    getWidget<CheckBoxWidget>("dof")->setState(UserConfigParams::m_dof);
    
    SpinnerWidget* particles_effects = getWidget<SpinnerWidget>("particles_effects");
    assert(particles_effects != NULL);
    particles_effects->addLabel(_C("Particle effects", "Disabled"));
    particles_effects->addLabel(_C("Particle effects", "Important only"));
    particles_effects->addLabel(_C("Particle effects", "Enabled"));
    particles_effects->setValue(UserConfigParams::m_particles_effects);

    SpinnerWidget* geometry_level = getWidget<SpinnerWidget>("geometry_detail");
    geometry_level->addLabel(_C("Geometry level", "Very low"));
    geometry_level->addLabel(_C("Geometry level", "Low"));
    geometry_level->addLabel(_C("Geometry level", "Medium"));
    geometry_level->addLabel(_C("Geometry level", "High"));
    geometry_level->addLabel(_C("Geometry level", "Very high"));
    geometry_level->addLabel(_C("Geometry level", "Ultra high"));
    geometry_level->setValue(UserConfigParams::m_geometry_level);

    SpinnerWidget* filtering = getWidget<SpinnerWidget>("image_quality");
    filtering->addLabel(_C("Image quality", "Very low"));
    filtering->addLabel(_C("Image quality", "Low"));
    filtering->addLabel(_C("Image quality", "High"));
    filtering->setValue(GraphicalPresets::getImageQuality());

    SpinnerWidget* shadows = getWidget<SpinnerWidget>("shadows");
    shadows->addLabel(_C("Shadows", "Disabled"));   // 0
    shadows->addLabel(_C("Shadows", "Low"));        // 1
    shadows->addLabel(_C("Shadows", "Medium"));     // 2
    shadows->addLabel(_C("Shadows", "High"));       // 3
    shadows->addLabel(_C("Shadows", "Very high"));  // 4
    shadows->setValue(UserConfigParams::m_shadows_resolution == 2048 ?
                      (UserConfigParams::m_pcss ? 4 : 3) :
                      UserConfigParams::m_shadows_resolution == 1024 ? 2 :
                      UserConfigParams::m_shadows_resolution ==  512 ? 1 : 0);

    fillAntiAliasing(std::string(UserConfigParams::m_render_driver) == "vulkan",
                     UserConfigParams::m_anti_aliasing);

    getWidget<CheckBoxWidget>("dynamiclight")->setState(UserConfigParams::m_dynamic_lights);
    getWidget<CheckBoxWidget>("lightshaft")->setState(UserConfigParams::m_light_shaft);
    getWidget<CheckBoxWidget>("ibl")->setState(!UserConfigParams::m_degraded_IBL);
    getWidget<CheckBoxWidget>("motionblur")->setState(UserConfigParams::m_motionblur);
    getWidget<CheckBoxWidget>("speedstreaks")->setState(UserConfigParams::m_speed_streaks);
    getWidget<CheckBoxWidget>("glow")->setState(UserConfigParams::m_glow);
    getWidget<CheckBoxWidget>("ssao")->setState(UserConfigParams::m_ssao);
    const bool vk = std::string(UserConfigParams::m_render_driver) == "vulkan";
    fillReflection("solid_reflection", vk, UserConfigParams::m_solid_reflection);
    fillReflection("water_reflection", vk, UserConfigParams::m_water_reflection);
    getWidget<CheckBoxWidget>("stochastic_reflection")
        ->setState(UserConfigParams::m_stochastic_reflection);
    getWidget<CheckBoxWidget>("bloom")->setState(UserConfigParams::m_bloom);
    getWidget<CheckBoxWidget>("lightscattering")->setState(UserConfigParams::m_light_scatter);
    getWidget<CheckBoxWidget>("pointlight_shadows")->setState(UserConfigParams::m_pointlight_shadows);

    if (CVS->isEXTTextureCompressionS3TCUsable())
    {
        getWidget<CheckBoxWidget>("texture_compression")->setState(UserConfigParams::m_texture_compression);
    }
    else
    {
        CheckBoxWidget* cb_tex_cmp = getWidget<CheckBoxWidget>("texture_compression");
        cb_tex_cmp->setState(false);
        cb_tex_cmp->setActive(false);
    }

    GUIEngine::SpinnerWidget* rds = getWidget<GUIEngine::SpinnerWidget>("render_driver");
    assert( rds != NULL );

    rds->m_properties[PROP_WRAP_AROUND] = "true";
    rds->clearLabels();
    rds->addLabel("OpenGL");
    rds->addLabel("Vulkan");
#ifndef WIN32
    const int rd_count = 2;
#else
    const int rd_count = 3;
    rds->addLabel("DirectX9");
#endif
    bool found = false;
    for (int i = 0; i < rd_count; i++)
    {
        std::string rd = StringUtils::wideToUtf8(rds->getStringValueFromID(i).make_lower());
        if (std::string(UserConfigParams::m_render_driver) == rd)
        {
            rds->setValue(i);
            found = true;
            break;
        }
    }
    if (!found)
    {
        rds->addLabel(StringUtils::utf8ToWide(UserConfigParams::m_render_driver));
        rds->setValue(rd_count);
    }
    rds->setActive(StateManager::get()->getGameState() != GUIEngine::INGAME_MENU);
#endif
} // beforeAddingWidgets

// -----------------------------------------------------------------------------

void CustomVideoSettingsDialog::fillAntiAliasing(bool vk,
                                                 const std::string& selected)
{
#ifndef SERVER_ONLY
    SpinnerWidget* aa = getWidget<SpinnerWidget>("anti_aliasing");
    assert(aa != NULL);
    aa->m_properties[PROP_WRAP_AROUND] = "true";
    aa->clearLabels();
    m_anti_aliasing_names.clear();

    m_anti_aliasing_names.push_back("off");
    aa->addLabel(_C("Anti-aliasing", "Disabled"));
    if (vk)
    {
        // GE: the cheap one for the low-end devices, then the better one
        m_anti_aliasing_names.push_back("fxaa");
        aa->addLabel(L"FXAA");
        m_anti_aliasing_names.push_back("cmaa2");
        aa->addLabel(L"CMAA2");
    }
    else
    {
        m_anti_aliasing_names.push_back("mlaa");
        aa->addLabel(L"MLAA");
    }

    // The choice of the other renderer (or the generic "mlaa") is the
    // default one of this renderer, an anti-aliasing which was on stays on
    std::string name = selected;
    if (std::find(m_anti_aliasing_names.begin(), m_anti_aliasing_names.end(),
        name) == m_anti_aliasing_names.end())
    {
        name = name == "off" ? "off" : m_anti_aliasing_names.back();
    }
    int value = 0;
    for (unsigned i = 0; i < m_anti_aliasing_names.size(); i++)
    {
        if (m_anti_aliasing_names[i] == name)
            value = i;
    }
    aa->setValue(value);
#endif
}   // fillAntiAliasing

// -----------------------------------------------------------------------------

void CustomVideoSettingsDialog::fillReflection(const std::string& name,
                                               bool vk, int level)
{
#ifndef SERVER_ONLY
    SpinnerWidget* reflection = getWidget<SpinnerWidget>(name.c_str());
    assert(reflection != NULL);
    reflection->clearLabels();
    level = std::max(0, std::min(level, 4));
    if (vk)
    {
        // The value of a label is the level saved in the config
        for (int i = 0; i <= 4; i++)
        {
            reflection->addLabel(
                OptionsScreenVideo::getReflectionLevelName(i));
        }
        reflection->setValue(level);
    }
    else
    {
        // OpenGL only has the trace over the depth buffer (the level 1 of
        // GE), so it's on or off, getReflectionLevel() keeps the level of
        // the config when it stays on
        reflection->addLabel(_C("Reflection", "Disabled"));
        reflection->addLabel(_C("Reflection", "Enabled"));
        reflection->setValue(level != 0 ? 1 : 0);
    }
#endif
}   // fillReflection

// -----------------------------------------------------------------------------

int CustomVideoSettingsDialog::getReflectionLevel(const std::string& name,
                                                  int config_level)
{
    int value = getWidget<SpinnerWidget>(name.c_str())->getValue();
    // The spinner of OpenGL has 2 labels: enabled keeps the level which
    // Vulkan uses (or the lowest one if it was disabled)
    if (getWidget<SpinnerWidget>(name.c_str())->getMax() == 1 && value == 1)
        value = std::max(1, std::min(config_level, 4));
    return value;
}   // getReflectionLevel

// -----------------------------------------------------------------------------

void CustomVideoSettingsDialog::showTab(const std::string& tab)
{
    // The pages are on top of each other in the dialog, a widget which is not
    // visible can't be focused or clicked (EventHandler skips them)
    const char* pages[][2] =
    {
        { "tab_lighting", "page_lighting" },
        { "tab_effects",  "page_effects"  },
        { "tab_quality",  "page_quality"  },
    };
    for (auto& page : pages)
        getWidget(page[1])->setVisible(tab == page[0]);
}   // showTab

// -----------------------------------------------------------------------------

GUIEngine::EventPropagation CustomVideoSettingsDialog::processEvent(const std::string& eventSource)
{
#ifndef SERVER_ONLY
    if (eventSource == "render_driver")
    {
        // We will only update settings if the changed renderer is
        // kept when the players chooses to apply the new config
        // However, we immediately update the GUI to show which
        // advanced settings are available or not with the chosen renderer
        std::string rd = StringUtils::wideToUtf8(
            getWidget<GUIEngine::SpinnerWidget>("render_driver")->getStringValue().make_lower());
        if (rd != std::string(UserConfigParams::m_render_driver))
            m_new_render_driver = rd;
        else
            m_new_render_driver.clear();
        // The choices of anti-aliasing are the ones of the chosen driver
        SpinnerWidget* aa = getWidget<SpinnerWidget>("anti_aliasing");
        const unsigned aa_value = aa->getValue();
        fillAntiAliasing(rd == "vulkan", aa_value < m_anti_aliasing_names.size() ?
            m_anti_aliasing_names[aa_value] : std::string("off"));
        // Same for the levels of the reflections, the choice is kept
        fillReflection("solid_reflection", rd == "vulkan",
            getReflectionLevel("solid_reflection",
            UserConfigParams::m_solid_reflection));
        fillReflection("water_reflection", rd == "vulkan",
            getReflectionLevel("water_reflection",
            UserConfigParams::m_water_reflection));
        updateActivation();
    }
    if (eventSource == "settings_tabs")
    {
        showTab(getWidget<RibbonWidget>("settings_tabs")->
            getSelectionIDString(PLAYER_ID_GAME_MASTER));
        return GUIEngine::EVENT_BLOCK;
    }
    if (eventSource == "buttons")
    {
        const std::string& selection = getWidget<RibbonWidget>("buttons")->
                                    getSelectionIDString(PLAYER_ID_GAME_MASTER);

        if (selection == "apply")
        {
            bool advanced_pipeline = getWidget<CheckBoxWidget>("dynamiclight")->getState();
            bool pbr_changed = false;
            bool ibl_changed = false;
            // What is built from them (shadow maps, the ray traced shadow,
            // volumetric shadow) uses these (the type of the setting is kept
            // when the size is 0), not the raw GEConfig fields
            const unsigned prev_shadow_size = GE::getShadowSize();
            const GE::GEShadowType prev_shadow_type = GE::getShadowType();
            const GE::GEShadowType prev_rt_shadow_type = GE::getRTShadowType();
            if (UserConfigParams::m_dynamic_lights != advanced_pipeline)
            {
                pbr_changed = true;
                GE::getGEConfig()->m_pbr = advanced_pipeline;
            }
            UserConfigParams::m_dynamic_lights = advanced_pipeline;

            bool dof_changed = false;
            bool dof = advanced_pipeline &&
                getWidget<CheckBoxWidget>("dof")->getState();
            if (UserConfigParams::m_dof != dof)
            {
                dof_changed = true;
                GE::getGEConfig()->m_dof = dof;
            }
            UserConfigParams::m_dof = dof;

            // The passes of the final chain of GE Vulkan, which has to make
            // the swapchain again (updateDriver) when one is enabled or
            // disabled, it needs a FBO (and the depth for the motion blur)
            const bool motionblur = advanced_pipeline &&
                getWidget<CheckBoxWidget>("motionblur")->getState();
            const bool speed_streaks = advanced_pipeline &&
                getWidget<CheckBoxWidget>("speedstreaks")->getState();
            const bool final_pass_changed =
                UserConfigParams::m_motionblur != motionblur ||
                UserConfigParams::m_speed_streaks != speed_streaks;
            UserConfigParams::m_motionblur = motionblur;
            UserConfigParams::m_speed_streaks = speed_streaks;

            if (advanced_pipeline)
            {
                UserConfigParams::m_shadows_resolution =
                    getWidget<SpinnerWidget>("shadows")->getValue() == 1 ?  512 :
                    getWidget<SpinnerWidget>("shadows")->getValue() == 2 ? 1024 :
                    getWidget<SpinnerWidget>("shadows")->getValue() >= 3 ? 2048 : 0;
                UserConfigParams::m_pcss =
                    getWidget<SpinnerWidget>("shadows")->getValue() == 4 ? true : false;
                GE::getGEConfig()->m_shadow_size = UserConfigParams::m_shadows_resolution;
                UserConfigParams::m_pointlight_shadows =
                    getWidget<CheckBoxWidget>("pointlight_shadows")->getState();
                GE::getGEConfig()->m_shadow_type =
                    UserConfigParams::m_pointlight_shadows ? GE::GST_COMBINED : GE::GST_SUN;
            }
            else
            {
                UserConfigParams::m_shadows_resolution = 0;
                UserConfigParams::m_pointlight_shadows = false;
                GE::getGEConfig()->m_shadow_size = 0;
            }

            // The config has the name of the choice, the renderer resolves it
            // (OptionsScreenVideo::setAntiAliasing for GE)
            const unsigned aa_value =
                getWidget<SpinnerWidget>("anti_aliasing")->getValue();
            UserConfigParams::m_anti_aliasing =
                aa_value < m_anti_aliasing_names.size() ?
                m_anti_aliasing_names[aa_value] : std::string("off");
            if (CVS->isGLSL() && !advanced_pipeline)
                UserConfigParams::m_anti_aliasing = "off";
            // The FBO the scene is rendered to is different with it
            const GE::GEAntiAliasing prev_anti_aliasing =
                GE::getGEConfig()->m_anti_aliasing;
            OptionsScreenVideo::setAntiAliasing();
            const bool anti_aliasing_changed =
                prev_anti_aliasing != GE::getGEConfig()->m_anti_aliasing;

            bool ssao_changed = false;
            bool ssao = advanced_pipeline &&
                getWidget<CheckBoxWidget>("ssao")->getState();
            if (UserConfigParams::m_ssao != ssao)
            {
                ssao_changed = true;
                GE::getGEConfig()->m_ssao = ssao;
            }
            UserConfigParams::m_ssao = ssao;
            // When a reflection is ray traced its spinner is disabled and
            // keeps the level of the config, which is used again when ray
            // tracing is off
            UserConfigParams::m_solid_reflection = advanced_pipeline ?
                getReflectionLevel("solid_reflection",
                UserConfigParams::m_solid_reflection) : 0;
            UserConfigParams::m_water_reflection = advanced_pipeline ?
                getReflectionLevel("water_reflection",
                UserConfigParams::m_water_reflection) : 0;
            // Kept as it is without the advanced pipeline, it's only an option
            // of the solid reflection
            UserConfigParams::m_stochastic_reflection =
                getWidget<CheckBoxWidget>("stochastic_reflection")->getState();
            bool god_rays_changed = false;
            bool god_rays = advanced_pipeline &&
                getWidget<CheckBoxWidget>("lightshaft")->getState();
            if (UserConfigParams::m_light_shaft != god_rays)
            {
                god_rays_changed = true;
                GE::getGEConfig()->m_god_rays = god_rays;
            }
            UserConfigParams::m_light_shaft = god_rays;

            bool degraded_ibl = !advanced_pipeline || !getWidget<CheckBoxWidget>("ibl")->getState();
            if (UserConfigParams::m_degraded_IBL != degraded_ibl)
            {
                ibl_changed = true;
                GE::getGEConfig()->m_ibl = !degraded_ibl;
                UserConfigParams::m_degraded_IBL = degraded_ibl;
            }

            bool glow_changed = false;
            bool glow = advanced_pipeline && getWidget<CheckBoxWidget>("glow")->getState();
            if (UserConfigParams::m_glow != glow)
            {
                glow_changed = true;
                GE::getGEConfig()->m_glow_outline = glow;
            }
            UserConfigParams::m_glow = glow;

            bool bloom_changed = false;
            bool bloom = advanced_pipeline &&
                getWidget<CheckBoxWidget>("bloom")->getState();
            if (UserConfigParams::m_bloom != bloom)
            {
                bloom_changed = true;
                GE::getGEConfig()->m_bloom = bloom;
            }
            UserConfigParams::m_bloom = bloom;

            bool light_scatter_changed = false;
            bool light_scatter = advanced_pipeline &&
                getWidget<CheckBoxWidget>("lightscattering")->getState();
            if (UserConfigParams::m_light_scatter != light_scatter)
            {
                light_scatter_changed = true;
                GE::getGEConfig()->m_light_scatter = light_scatter;
            }
            UserConfigParams::m_light_scatter = light_scatter;

            bool force_reload_texture = getWidget<CheckBoxWidget>("texture_compression")->getState() !=
                UserConfigParams::m_texture_compression;
            UserConfigParams::m_texture_compression =
                getWidget<CheckBoxWidget>("texture_compression")->getState();
            GE::getGEConfig()->m_texture_compression = UserConfigParams::m_texture_compression;

            UserConfigParams::m_particles_effects =
                getWidget<SpinnerWidget>("particles_effects")->getValue();

            UserConfigParams::m_animated_characters =
                getWidget<CheckBoxWidget>("animated_characters")->getState();

            UserConfigParams::m_geometry_level =
                getWidget<SpinnerWidget>("geometry_detail")->getValue();;
            int quality = getWidget<SpinnerWidget>("image_quality")->getValue();

            std::string rd = StringUtils::wideToUtf8(
                getWidget<GUIEngine::SpinnerWidget>("render_driver")->getStringValue().make_lower());

            bool need_restart = false;
            if (std::string(UserConfigParams::m_render_driver) != rd)
            {
                UserConfigParams::m_render_driver = rd;
                need_restart = true;
            }

            user_config->saveConfig();

            ModalDialog::dismiss();
            OptionsScreenVideo::getInstance()->updateGfxSlider();
            OptionsScreenVideo::getInstance()->updateBlurSlider();
            GE::GESolidReflectionType prev_solid = GE::getGEConfig()->m_solid_reflection;
            GE::GEDisplaceReflectionType prev_displace = GE::getGEConfig()->m_displace_reflection;
            const bool prev_stochastic =
                GE::getGEConfig()->m_stochastic_reflection;
            OptionsScreenVideo::setSSR();
            if (GE::getDriver()->getDriverType() == video::EDT_VULKAN)
            {
                // The stochastic reflection has other passes and a half size
                // target, the render target is created again
                bool need_recreate_swapchain =
                    GE::getGEConfig()->m_solid_reflection != prev_solid ||
                    GE::getGEConfig()->m_displace_reflection != prev_displace ||
                    GE::getGEConfig()->m_stochastic_reflection !=
                    prev_stochastic;
                // Which shadows are ray traced (the FBO has the ray traced
                // shadow only if there is one), also changed by the size 0
                if (prev_rt_shadow_type != GE::getRTShadowType())
                    need_recreate_swapchain = true;
                if (anti_aliasing_changed)
                    need_recreate_swapchain = true;
                // For volumetric shadow
                if ((prev_shadow_size != GE::getShadowSize() ||
                    prev_shadow_type != GE::getShadowType()) &&
                    GE::getGEConfig()->m_light_scatter)
                    need_recreate_swapchain = true;
                if (need_recreate_swapchain || pbr_changed || ibl_changed ||
                    glow_changed || light_scatter_changed || bloom_changed ||
                    ssao_changed || god_rays_changed || dof_changed ||
                    final_pass_changed ||
                    prev_shadow_size != GE::getShadowSize() ||
                    prev_shadow_type != GE::getShadowType())
                    GE::getVKDriver()->updateDriver(need_recreate_swapchain, pbr_changed, ibl_changed);
            }
            // sameRestart will have the same effect
            if (!(CVS->isGLSL() && pbr_changed))
            {
                GraphicalPresets::setImageQuality(quality);
                OptionsScreenVideo::updateImageQuality(force_reload_texture);
            }

            if (need_restart)
                irr_driver->fullRestart();

            return GUIEngine::EVENT_BLOCK;
        }
        else if (selection == "cancel")
        {
            ModalDialog::dismiss();
            return GUIEngine::EVENT_BLOCK;
        }
    }
    else if (eventSource == "dynamiclight" || eventSource == "shadows" ||
             eventSource == "solid_reflection")
    {
        updateActivation();
    }
#endif
    return GUIEngine::EVENT_LET;
}   // processEvent

// -----------------------------------------------------------------------------

void CustomVideoSettingsDialog::updateActivation()
{
#ifndef SERVER_ONLY
    bool light = getWidget<CheckBoxWidget>("dynamiclight")->getState();
    bool real_light = light;
    bool vk = GE::getDriver()->getDriverType() == video::EDT_VULKAN;
    bool modern_gl = CVS->isGLSL();

    // If showing enabled options for a specific renderer has
    // been requested, prioritize that
    if (m_new_render_driver == "vulkan")
    {
        vk = true;
        modern_gl = false;
    }
    else if (m_new_render_driver != "") // OpenGL or DirectX
    {
        vk = false;

        if (m_new_render_driver == "opengl" && !UserConfigParams::m_force_legacy_device)
            modern_gl = true;
        else
            modern_gl = false;
    }

    // Disable the options for advanced lighting if unavailable for this renderer
    if (!vk && !modern_gl)
    {
        getWidget<CheckBoxWidget>("dynamiclight")->setActive(false);
        light = false;
    }

    if (vk)
    {
        getWidget<CheckBoxWidget>("dynamiclight")->setActive(true);
        light = false;
    }
    getWidget<CheckBoxWidget>("motionblur")->setActive(light || (vk && real_light));
    getWidget<CheckBoxWidget>("speedstreaks")->setActive(light || (vk && real_light));
    getWidget<CheckBoxWidget>("dof")->setActive(light || (vk && real_light));
    getWidget<SpinnerWidget>("shadows")->setActive(light || (vk && real_light));
    getWidget<SpinnerWidget>("anti_aliasing")->setActive(light || vk);
    getWidget<CheckBoxWidget>("ssao")->setActive(light || (vk && real_light));
    // The ray traced reflections (RayTracingSettingsDialog) override the
    // screen space ones, their spinners only show it then
    const bool solid_rt = vk && GE::isRayTracing() &&
        GE::getGEConfig()->m_ray_tracing.m_solid_reflection;
    const bool water_rt = vk && GE::isRayTracing() &&
        GE::getGEConfig()->m_ray_tracing.m_displace_reflection;
    SpinnerWidget* solid = getWidget<SpinnerWidget>("solid_reflection");
    SpinnerWidget* water = getWidget<SpinnerWidget>("water_reflection");
    solid->setActive(!solid_rt && (light || (vk && real_light)));
    water->setActive(!water_rt && (light || (vk && real_light)));
    if (solid_rt)
        solid->setCustomText(_C("Reflection", "Ray traced"));
    if (water_rt)
        water->setCustomText(_C("Reflection", "Ray traced"));
    // GE only, it samples the reflection of the solid materials (screen space
    // or ray traced)
    getWidget<CheckBoxWidget>("stochastic_reflection")->setActive(vk &&
        real_light && (solid_rt || solid->getValue() != 0));
    getWidget<CheckBoxWidget>("lightshaft")->setActive(light || (vk && real_light));
    getWidget<CheckBoxWidget>("ibl")->setActive(light || (vk && real_light));
    getWidget<CheckBoxWidget>("glow")->setActive(light || (vk && real_light));
    getWidget<CheckBoxWidget>("bloom")->setActive(light || (vk && real_light));
    getWidget<CheckBoxWidget>("lightscattering")->setActive(light || (vk && real_light));
    getWidget<CheckBoxWidget>("pointlight_shadows")->setActive(vk && real_light &&
        getWidget<SpinnerWidget>("shadows")->getValue() != 0);
#endif
}   // updateActivation
