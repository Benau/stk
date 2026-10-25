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


#ifndef HEADER_CUSTOM_VIDEO_SETTINGS_HPP
#define HEADER_CUSTOM_VIDEO_SETTINGS_HPP

#include "guiengine/modaldialog.hpp"

#include <string>
#include <vector>

/**
 * \brief Dialog that allows the player to select custom video settings
 * \ingroup states_screens
 */
class CustomVideoSettingsDialog : public GUIEngine::ModalDialog
{
private:
    std::string m_new_render_driver;

    /** The names saved in the config (UserConfigParams::m_anti_aliasing) of
     *  the labels of the anti-aliasing spinner, which depend on the render
     *  driver. */
    std::vector<std::string> m_anti_aliasing_names;

    /** Fills the anti-aliasing spinner with the choices of the render driver,
     *  and selects selected (or the default one of the render driver if it
     *  has no such choice). */
    void fillAntiAliasing(bool vk, const std::string& selected);
public:
    /**
     * Creates a modal dialog with given percentage of screen width and height
     */
    CustomVideoSettingsDialog(const float percentWidth, const float percentHeight);
    ~CustomVideoSettingsDialog();

    virtual void beforeAddingWidgets();

    /** Updates the activation state of the advanced pipeline checkboxes. */
    void updateActivation();

    GUIEngine::EventPropagation processEvent(const std::string& eventSource);
    
};

#endif
