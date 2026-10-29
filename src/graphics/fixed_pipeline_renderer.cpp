//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2015 SuperTuxKart-Team
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

#ifndef SERVER_ONLY
#include "graphics/fixed_pipeline_renderer.hpp"
#include "config/user_config.hpp"
#include "graphics/camera/camera.hpp"
#include "graphics/ge_final_effects.hpp"
#include "graphics/ge_rtt_test.hpp"
#include "graphics/irr_driver.hpp"
#include "graphics/weather.hpp"
#include "graphics/render_target.hpp"
#include "modes/world.hpp"
#include "physics/physics.hpp"
#include "utils/profiler.hpp"

#include <ge_main.hpp>

#include <ISceneManager.h>
#include <IVideoDriver.h>

void FixedPipelineRenderer::onLoadWorld()
{
    
}

void FixedPipelineRenderer::resetPostProcessing()
{
    if (m_final_effects)
        m_final_effects->reset();
}

void FixedPipelineRenderer::giveBoost(unsigned int cam_index)
{
    if (m_final_effects)
        m_final_effects->giveBoost(cam_index);
}

void FixedPipelineRenderer::onUnloadWorld()
{
    if (GERTTTest::isEnabled())
        GERTTTest::getInstance()->reset();
}

FixedPipelineRenderer::FixedPipelineRenderer()
{
    if (GE::getDriver()->getDriverType() == video::EDT_VULKAN)
        m_final_effects.reset(new GEFinalEffects());
}   // FixedPipelineRenderer

// ----------------------------------------------------------------------------
FixedPipelineRenderer::~FixedPipelineRenderer()
{
}   // ~FixedPipelineRenderer

// ----------------------------------------------------------------------------
void FixedPipelineRenderer::render(float dt, bool is_loading)
{
    World *world = World::getWorld(); // Never NULL.

    if (m_final_effects)
        m_final_effects->update(dt);

    irr_driver->getVideoDriver()->beginScene(/*backBuffer clear*/ true,
                                             /*zBuffer*/ true,
                                             irr_driver->getClearColor());
    
    irr_driver->getVideoDriver()->enableMaterial2D();

    RaceGUIBase *rg = world->getRaceGUI();
    if (rg) rg->update(dt);


    for(unsigned int i=0; i<Camera::getNumCameras(); i++)
    {
        Camera *camera = Camera::getCamera(i);

        std::ostringstream oss;
        oss << "drawAll() for kart " << i;
        PROFILER_PUSH_CPU_MARKER(oss.str().c_str(), (i+1)*60,
                                 0x00, 0x00);
        camera->activate();
        rg->preRenderCallback(camera);   // adjusts start referee
        if (m_final_effects)
            m_final_effects->prepareCamera(camera);

        // Lightning quad of this camera only (GLSL renderer doesn't use this)
        Weather* weather = Weather::getInstance();
        if (weather)
            weather->prepareLightning(camera);

        irr_driver->getSceneManager()->drawAll();

        if (weather)
            weather->finishLightning();

        // Test of several drawAll for a separate RTT (GE Vulkan), only active
        // with STK_GE_RTT_TEST environment variable
        if (GERTTTest::isEnabled() && Camera::getNumCameras() == 1)
            GERTTTest::getInstance()->renderViews(camera);

        PROFILER_POP_CPU_MARKER();

        // Note that drawAll must be called before rendering
        // the bullet debug view, since otherwise the camera
        // is not set up properly. This is only used for
        // the bullet debug view.
        if (UserConfigParams::m_artist_debug_mode)
            Physics::get()->draw();
    }   // for i<world->getNumKarts()

    // For GEVulkanDriver
    irr_driver->getSceneManager()->setActiveCamera(NULL);

    // Set the viewport back to the full screen for race gui
    irr_driver->getVideoDriver()->setViewPort(core::recti(0, 0,
                                              UserConfigParams::m_width,
                                              UserConfigParams::m_height));

    if (GERTTTest::isEnabled())
        GERTTTest::getInstance()->drawResult();

    for(unsigned int i=0; i<Camera::getNumCameras(); i++)
    {
        Camera *camera = Camera::getCamera(i);
        irr_driver->getSceneManager()->setActiveCamera(
            camera->getCameraSceneNode());
        std::ostringstream oss;
        oss << "renderPlayerView() for kart " << i;

        PROFILER_PUSH_CPU_MARKER(oss.str().c_str(), 0x00, 0x00, (i+1)*60);
        rg->renderPlayerView(camera, dt);
        PROFILER_POP_CPU_MARKER();

    }  // for i<getNumKarts

    // Either render the gui, or the global elements of the race gui.
    GUIEngine::render(dt, is_loading);

    if (irr_driver->getRenderNetworkDebug() && !is_loading)
        irr_driver->renderNetworkDebug();

    // Render the profiler
    if(UserConfigParams::m_profiler_enabled)
    {
        PROFILER_DRAW();
    }

#ifdef DEBUG
    drawDebugMeshes();
#endif

    irr_driver->getVideoDriver()->endScene();
    
}

std::unique_ptr<RenderTarget> FixedPipelineRenderer::createRenderTarget(const irr::core::dimension2du &dimension,
                                                                        const std::string &name)
{
    return std::unique_ptr<RenderTarget>(new GL1RenderTarget(dimension, name));
}
#endif
