//
//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2011-2015  Joerg Henrichs, Marianne Gagnon
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

#ifndef HEADER_WEATHER_HPP
#define HEADER_WEATHER_HPP

#include "utils/singleton.hpp"
#include <vector3d.h>

class Camera;
class SFXBase;

namespace irr
{
    namespace scene { class ISceneNode; }
    namespace video { class ITexture; }
}

class Weather : public AbstractSingleton<Weather>
{
    float m_next_lightning;
    float m_lightning;

    SFXBase* m_thunder_sound;
    SFXBase* m_weather_sound;

    // Lightning for the non-GLSL renderers (GEVulkanDriver and the legacy
    // fixed pipeline): a quad right in front of a camera, shared mesh,
    // additive material and the intensity as the vertex color of the render
    // info of its node. The GLSL renderer does it in PostProcessing.
    irr::scene::ISceneNode* m_lightning_node;
    irr::video::ITexture* m_lightning_texture;

    void createLightningQuad();

public:
             Weather();
    virtual ~Weather();

    void update(float dt);
    void playSound();
    
    /** Set the flag that a lightning should be shown. */
    void startLightning() { m_lightning = 1.0f; }
    bool shouldLightning() { return m_lightning > 0.0f; }
    
    irr::core::vector3df getIntensity();
    /** Called before the scene is drawn for this camera, shows the quad of
     *  this camera only (if there is a lightning now). */
    void prepareLightning(Camera* camera);
    /** Called after the scene is drawn for the camera, hides the quad. */
    void finishLightning();
};

#endif
