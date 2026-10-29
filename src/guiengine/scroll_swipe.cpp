//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2026 Dawid Gan <deveee@gmail.com>
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

#include "graphics/irr_driver.hpp"
#include "guiengine/scroll_swipe.hpp"
#include "utils/time.hpp"

ScrollSwipe::ScrollSwipe(gui::IGUIEnvironment *env,
        gui::IGUIElement *parent, OrientationEnum orientation)
    : m_env(env),
      m_parent(parent),
      m_scrollbar(NULL),
      m_orientation(orientation),
      m_swipe_started(false),
      m_is_coasting(false),
      m_swipe_start_px(-1),
      m_last_time(0),
      m_swipe_pos(0),
      m_last_pos(0),
      m_velocity(0),
      m_scrollfactor(1.0f),
      m_velocity_samples{},
      m_sample_count(0),
      m_sample_index(0)
{
}

void ScrollSwipe::calculateCoastingVelocity()
{
    if (m_sample_count < 1)
        return;

    const u64 now = StkTime::getMonoTimeMs();

    u32 last_sample_index = (m_sample_index - 1 + MAX_VELOCITY_SAMPLES) %
                MAX_VELOCITY_SAMPLES;
    VelocitySample last_sample = m_velocity_samples[last_sample_index];

    if (now - last_sample.timestamp > 100)
    {
        m_velocity = 0;
        m_is_coasting = false;
        return;
    }

    VelocitySample first_sample = last_sample;
    for (u32 i = 0; i < m_sample_count; i++)
    {
        VelocitySample &sample = m_velocity_samples[i];

        if (now - sample.timestamp > 100)
            continue;

        if (sample.timestamp < first_sample.timestamp)
            first_sample = sample;
    }

    u64 dt = last_sample.timestamp - first_sample.timestamp;
    if (dt > 0)
    {
        double screen_dpi = getScreenDpi();
        float distance = (last_sample.position - first_sample.position) *
                 m_scrollfactor;
        float distance_in = distance * 1000 / screen_dpi;

        m_velocity = distance_in / dt;
        m_velocity = std::max(std::min(m_velocity, 40.0f), -40.0f);
        m_last_time = now;
        m_is_coasting = true;
    }
}

void ScrollSwipe::updateScrollCoasting()
{
    if (!m_is_coasting || m_velocity == 0)
        return;

    if ((int)m_swipe_pos != m_scrollbar->getPos())
    {
        m_velocity = 0;
        m_is_coasting = false;
        return;
    }

    u64 current_time = StkTime::getMonoTimeMs();
    u64 dt = current_time - m_last_time;
    m_last_time = current_time;

    if (dt == 0)
        return;

    const float SLOWING_FACTOR = 0.97f;
    const float VELOCITY_FACTOR = 1.0f;

    // We want SLOWING_FACTOR to be added every frame assuming 60 FPS
    m_velocity *= std::pow(SLOWING_FACTOR, dt / (1000.0f / 60.0f));
    if (std::abs(m_velocity) < 0.3f)
    {
        m_is_coasting = false;
        return;
    }

    double screen_dpi = getScreenDpi();
    float pixel_velocity = m_velocity * screen_dpi / m_scrollfactor / 1000;

    m_swipe_pos += pixel_velocity * VELOCITY_FACTOR * dt;

    if (m_swipe_pos <= m_scrollbar->getMin())
    {
        m_swipe_pos = m_scrollbar->getMin();
        m_velocity = 0;
        m_is_coasting = false;
    }
    else if (m_swipe_pos >= m_scrollbar->getMax())
    {
        m_swipe_pos = m_scrollbar->getMax();
        m_velocity = 0;
        m_is_coasting = false;
    }

    s32 old_pos = m_scrollbar->getPos();
    m_scrollbar->setPos((int)m_swipe_pos);

    if (old_pos != m_scrollbar->getPos())
    {
        SEvent e;
        e.EventType = EET_GUI_EVENT;
        e.GUIEvent.Caller = m_scrollbar;
        e.GUIEvent.Element = NULL;
        e.GUIEvent.EventType = gui::EGET_SCROLL_BAR_CHANGED;
        m_parent->OnEvent(e);
    }
}

bool ScrollSwipe::onEvent(const SEvent &event)
{
    if (event.EventType != EET_MOUSE_INPUT_EVENT || !m_scrollbar)
        return false;

    const int mouse_pos = m_orientation == HORIZONTAL ? event.MouseInput.X
                              : event.MouseInput.Y;
    if (event.MouseInput.Event == EMIE_LMOUSE_PRESSED_DOWN)
    {
        if (m_parent->isPointInside(core::position2d<s32>(
                    event.MouseInput.X, event.MouseInput.Y)))
        {
            m_swipe_start_px = mouse_pos -
                       m_scrollbar->getPos() * m_scrollfactor;
            m_velocity = 0;
            m_is_coasting = false;
            m_last_pos = m_scrollbar->getPos();
        }
    }
    else if (event.MouseInput.Event == EMIE_LMOUSE_LEFT_UP)
    {
        m_swipe_start_px = -1;
        if (m_swipe_started)
        {
            m_swipe_started = false;
            calculateCoastingVelocity();
            return true;
        }
    }
    else if (event.MouseInput.Event == EMIE_MOUSE_MOVED)
    {
        double screen_dpi = getScreenDpi();

        if (!m_swipe_started && m_orientation != UNDEFINED &&
                m_swipe_start_px != -1 &&
                std::abs(m_swipe_start_px - mouse_pos +
                        m_scrollbar->getPos() * m_scrollfactor) >
                        0.1 * screen_dpi)
        {
            m_swipe_started = true;
            // m_env->setFocus(m_parent);
            m_sample_count = 0;
            m_sample_index = 0;
        }

        if (m_swipe_started)
        {
            s32 old_pos = m_scrollbar->getPos();
            m_swipe_pos = (float)(mouse_pos - m_swipe_start_px) /
                      m_scrollfactor;

            u64 current_time = StkTime::getMonoTimeMs();

            m_velocity_samples[m_sample_index].position = m_swipe_pos;
            m_velocity_samples[m_sample_index].timestamp = current_time;
            m_sample_index = (m_sample_index + 1) % MAX_VELOCITY_SAMPLES;
            if (m_sample_count < MAX_VELOCITY_SAMPLES)
                m_sample_count++;

            m_last_time = current_time;
            m_scrollbar->setPos((int)m_swipe_pos);

            if (old_pos != m_scrollbar->getPos())
            {
                SEvent e;
                e.EventType = EET_GUI_EVENT;
                e.GUIEvent.Caller = m_scrollbar;
                e.GUIEvent.Element = NULL;
                e.GUIEvent.EventType = gui::EGET_SCROLL_BAR_CHANGED;
                m_parent->OnEvent(e);
            }
            return true;
        }
    }
    return false;
}

void ScrollSwipe::cancelSwipe()
{
    m_swipe_started = false;
    m_is_coasting = false;
    m_swipe_start_px = -1;
    m_velocity = 0;
    m_sample_count = 0;
}

double ScrollSwipe::getScreenDpi()
{
    float ddpi = 0.0f, hdpi = 0.0f, vdpi = 0.0f;
    irr_driver->getDevice()->getDisplayDPI(&ddpi, &hdpi, &vdpi);

    float dpi = (m_orientation == HORIZONTAL) ? hdpi : vdpi;
    if (dpi <= 0.0f)
        dpi = ddpi;
    if (dpi <= 0.0f)
        dpi = 96.0f;

    return dpi;
}
