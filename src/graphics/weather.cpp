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

#include "audio/sfx_base.hpp"
#include "graphics/camera/camera.hpp"
#include "graphics/central_settings.hpp"
#include "graphics/irr_driver.hpp"
#include "graphics/material.hpp"
#include "graphics/material_manager.hpp"
#include "guiengine/engine.hpp"
#include "audio/sfx_manager.hpp"
#include "graphics/weather.hpp"
#include "modes/world.hpp"
#include "tracks/track.hpp"
#include "utils/random_generator.hpp"

#include <algorithm>
#include <cmath>

#ifndef SERVER_ONLY
#include <ge_main.hpp>
#include <ge_render_info.hpp>
#include <ge_texture.hpp>
#include <ge_vulkan_features.hpp>
#include <ICameraSceneNode.h>
#include <IImage.h>
#include <IMesh.h>
#include <IMeshCache.h>
#include <ISceneManager.h>
#include <ISceneNode.h>
#include <ITexture.h>
#include <IVideoDriver.h>
#include <SMesh.h>
#include <SMeshBuffer.h>
#endif


/**  The weather manager stores information about the weather.
 */
Weather::Weather()
{
    m_thunder_sound = NULL;
    m_weather_sound = NULL;
    m_lightning = 0.0f;
    m_lightning_node = NULL;
    m_lightning_texture = NULL;
    
    if (Track::getCurrentTrack()->getWeatherLightning())
    {
        m_thunder_sound = SFXManager::get()->createSoundSource("thunder");
#ifndef SERVER_ONLY
        if (!GUIEngine::isNoGraphics() && !CVS->isGLSL() &&
            (GE::getDriver()->getDriverType() == video::EDT_VULKAN ||
            GE::getDriver()->getDriverType() == video::EDT_OGLES2))
            createLightningQuad();
#endif
    }

    const std::string &sound = Track::getCurrentTrack()->getWeatherSound();
    if (!sound.empty())
    {
        m_weather_sound = SFXManager::get()->createSoundSource(sound);
    }

    RandomGenerator g;
    m_next_lightning = (float)g.get(35);
}   // Weather

// ----------------------------------------------------------------------------

Weather::~Weather()
{
    if (m_lightning_node)
        irr_driver->removeNode(m_lightning_node);
    scene::IMeshCache* mc = irr_driver->getSceneManager()->getMeshCache();
    scene::IAnimatedMesh* amesh = mc->getMeshByName(GE::getLightningIdent());
    if (amesh && amesh->getReferenceCount() == 1)
        mc->removeMesh(amesh);
    if (m_lightning_texture)
        m_lightning_texture->drop();
    if (m_thunder_sound != NULL)
        m_thunder_sound->deleteSFX();
        
    if (m_weather_sound != NULL)
        m_weather_sound->deleteSFX();
}   // ~Weather

// ----------------------------------------------------------------------------

void Weather::update(float dt)
{
    if (!Track::getCurrentTrack()->getWeatherLightning())
        return;
        
    if (World::getWorld()->getRaceGUI() == NULL)
        return;
        
    m_next_lightning -= dt;

    if (m_next_lightning < 0.0f)
    {
        startLightning();

        if (m_thunder_sound &&
            World::getWorld()->getPhase() != WorldStatus::IN_GAME_MENU_PHASE)
        {
            m_thunder_sound->play();
        }

        RandomGenerator g;
        m_next_lightning = 35 + (float)g.get(35);
    }
    
    if (m_lightning > 0.0f)
    {
        m_lightning -= dt;
    }
}   // update

// ----------------------------------------------------------------------------

void Weather::playSound()
{
    if (m_weather_sound)
    {
        m_weather_sound->setLoop(true);
        m_weather_sound->play();
    }
}

irr::core::vector3df Weather::getIntensity()
{
    irr::core::vector3df value = {0.7f * m_lightning,
                                  0.7f * m_lightning,
                                  0.7f * std::min(1.0f, m_lightning * 1.5f)};
                                 
    return value;
}

#ifndef SERVER_ONLY
// ----------------------------------------------------------------------------
/** Creates the white (with alpha channel, so the additive blending which uses
 *  the alpha of texture * vertex color works in all drivers) texture and the
 *  quad mesh shared by the quads of all cameras.
 */
void Weather::createLightningQuad()
{
    video::IImage* img = irr_driver->getVideoDriver()->createImage(
        video::ECF_A8R8G8B8, core::dimension2du(2, 2));
    img->fill(video::SColor(255, 255, 255, 255));
    // The texture takes the ownership of the image
    m_lightning_texture = GE::createTexture(img, "lightning_white");
    if (!m_lightning_texture)
        return;

    scene::SMeshBuffer* buffer = new scene::SMeshBuffer();
    video::S3DVertex v;
    v.Normal = core::vector3df(0.0f, 0.0f, -1.0f);
    v.Color = video::SColor(255, 255, 255, 255);
    const float pos[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (unsigned i = 0; i < 4; i++)
    {
        v.Pos = core::vector3df(pos[i][0], pos[i][1], 0.0f);
        v.TCoords = core::vector2df((pos[i][0] + 1.0f) * 0.5f,
            1.0f - (pos[i][1] + 1.0f) * 0.5f);
        buffer->Vertices.push_back(v);
    }
    const uint16_t indices[6] = { 0, 1, 2, 0, 2, 3 };
    for (uint16_t i : indices)
        buffer->Indices.push_back(i);

    Material* material = material_manager->getDefaultSPMaterial("additive");
    buffer->getMaterial().TextureLayer[0].Texture = m_lightning_texture;
    material->setMaterialProperties(&buffer->getMaterial(), buffer);
    buffer->Material.setFlag(video::EMF_BACK_FACE_CULLING, false);
    buffer->Material.setFlag(video::EMF_LIGHTING, false);
    buffer->Material.setFlag(video::EMF_COLOR_MATERIAL, true);
    buffer->Material.ColorMaterial = video::ECM_DIFFUSE_AND_AMBIENT;

    scene::SMesh* mesh = new scene::SMesh();
    mesh->addMeshBuffer(buffer);
    mesh->recalculateBoundingBox();
    buffer->drop();
    scene::IMeshCache* mc = irr_driver->getSceneManager()->getMeshCache();
    if (GE::getDriver()->getDriverType() == video::EDT_VULKAN)
    {
        scene::IAnimatedMesh* amesh = GE::convertIrrlichtMeshToSPM(mesh);
        mesh->drop();
        mc->addMesh(GE::getLightningIdent(), amesh);
        m_lightning_node = irr_driver->addMesh(amesh, GE::getLightningIdent());
        m_lightning_node->setVisible(false);
        amesh->drop();
    }
    else
    {
        mc->addMesh(GE::getLightningIdent(), mesh);
        m_lightning_node = irr_driver->addMesh(mesh, GE::getLightningIdent());
        m_lightning_node->setVisible(false);
        mesh->drop();
    }
    m_lightning_node->getMaterial(0).getRenderInfo() =
        std::make_shared<GE::GERenderInfo>();
}   // createLightningQuad

// ----------------------------------------------------------------------------
/** Vertex color (with alpha) of the quad so that after its blending the
 *  display color increases by the intensity (what the GL lightning shader
 *  does after tonemapping, with GL_ONE, GL_ONE).
 *  The additive blending of non-PBR materials and the legacy driver gives
 *  dst + rgb * alpha. The PBR transparent.frag also puts the rgb through the
 *  tonemap curve (convertColor, srgb to linear is done on cpu), so the
 *  inverse of it is needed, otherwise 0.7 would became ~0.95.
 */
static video::SColor getLightningColor(const core::vector3df& intensity)
{
    const float max_i = std::max(intensity.X,
        std::max(intensity.Y, intensity.Z));
    if (max_i <= 0.0f)
        return video::SColor(0, 255, 255, 255);
    float rgb[3] = { intensity.X / max_i, intensity.Y / max_i,
        intensity.Z / max_i };
    float alpha = std::min(max_i, 1.0f);
    if (GE::getGEConfig()->m_pbr &&
        GE::getDriver()->getDriverType() == video::EDT_VULKAN)
    {
        // Same as the ibl specialization constant of GEVulkanDrawCall
        const bool ibl = GE::getGEConfig()->m_ibl &&
            GE::GEVulkanFeatures::supportsComputeInMainQueue();
        const float a = ibl ? 6.5f : 7.0f;
        const float b = ibl ? 0.45f : 0.75f;
        const float c = 5.0f, e = 1.75f;
        // f(x) = x(ax+b) / (x(cx+e)+0.05)
        const float k = (a + b) / (c + e + 0.05f);
        alpha = std::min(max_i / k, 1.0f);
        for (unsigned i = 0; i < 3; i++)
        {
            const float d = std::min(rgb[i] * k, k);
            // (a-dc)x^2 + (b-de)x - 0.05d = 0, the positive root
            const float qa = a - d * c;
            const float qb = b - d * e;
            const float qc = -0.05f * d;
            float x = 0.0f;
            if (std::fabs(qa) > 1e-6f)
            {
                x = (-qb + std::sqrt(qb * qb - 4.0f * qa * qc)) /
                    (2.0f * qa);
            }
            else if (std::fabs(qb) > 1e-6f)
                x = -qc / qb;
            x = std::min(std::max(x, 0.0f), 1.0f);
            // Linear to srgb because it will be converted back on cpu
            rgb[i] = x <= 0.0031308f ? 12.92f * x :
                1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
        }
    }
    return video::SColor((uint32_t)(alpha * 255.0f + 0.5f),
        (uint32_t)(rgb[0] * 255.0f + 0.5f),
        (uint32_t)(rgb[1] * 255.0f + 0.5f),
        (uint32_t)(rgb[2] * 255.0f + 0.5f));
}   // getLightningColor

// ----------------------------------------------------------------------------
void Weather::prepareLightning(Camera* camera)
{
    if (!m_lightning_node || !camera)
        return;
    finishLightning();
    if (!shouldLightning())
        return;

    // Right after the near plane, large enough to cover the whole view
    scene::ICameraSceneNode* cam = camera->getCameraSceneNode();
    const core::vector3df pos = cam->getPosition();
    core::vector3df dir = cam->getTarget() - pos;
    if (dir.getLengthSQ() < 0.000001f)
        return;

    const float dist = cam->getNearValue() * 1.02f + 0.01f;
    const float half_h = dist * tanf(cam->getFOV() * 0.5f) * 1.1f;
    const float half_w = half_h * cam->getAspectRatio();

    dir.normalize();

    // Preserve the camera's roll, including when the kart is upside down
    core::vector3df up = cam->getUpVector();
    up.normalize();

    // Avoid a degenerate look-at matrix if the up vector is parallel
    // to the viewing direction. Match Irrlicht's camera fallback
    if (std::fabs(dir.dotProduct(up)) > 0.9999f)
        up.X += 0.5f;

    // Build the camera's world-space orientation from its viewing direction
    // and up vector, then invert the view matrix to obtain its orientation
    core::matrix4 view;
    view.buildCameraLookAtMatrixLH(pos, cam->getTarget(), up);
    if (!view.makeInverse())
        return;

    m_lightning_node->setPosition(pos + dir * dist);
    m_lightning_node->setRotation(view.getRotationDegrees());
    m_lightning_node->setScale(core::vector3df(half_w, half_h, 1.0f));
    m_lightning_node->getMaterial(0).getRenderInfo()->getVertexColor() =
        getLightningColor(getIntensity());
    m_lightning_node->setVisible(true);
}   // prepareLightning

// ----------------------------------------------------------------------------
void Weather::finishLightning()
{
    if (m_lightning_node)
        m_lightning_node->setVisible(false);
}   // finishLightning
#else
void Weather::createLightningQuad() {}
void Weather::prepareLightning(Camera*) {}
void Weather::finishLightning() {}
#endif
