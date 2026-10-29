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

#ifndef HEADER_SCROLL_SWIPE_HPP
#define HEADER_SCROLL_SWIPE_HPP

#include "irrlicht.h"


using namespace irr;
using namespace irr::core;

class ScrollSwipe
{
public:
    enum OrientationEnum
    {
        VERTICAL,
        HORIZONTAL,
        UNDEFINED
    };

    ScrollSwipe(gui::IGUIEnvironment *env, gui::IGUIElement *parent,
            OrientationEnum orientation);

    void calculateCoastingVelocity();
    void updateScrollCoasting();
    bool onEvent(const SEvent &event);
    void cancelSwipe();

    void setScrollBar(gui::IGUIScrollBar *scrollbar) { m_scrollbar = scrollbar; }
    void setScrollFactor(f32 scrollfactor) { m_scrollfactor = scrollfactor; }

private:
    double getScreenDpi();

    gui::IGUIEnvironment *m_env;
    gui::IGUIElement *m_parent;
    gui::IGUIScrollBar *m_scrollbar;

    OrientationEnum m_orientation;
    bool m_swipe_started;
    bool m_is_coasting;
    int m_swipe_start_px;
    u64 m_last_time;
    float m_swipe_pos;
    float m_last_pos;
    float m_velocity;
    f32 m_scrollfactor;

    struct VelocitySample
    {
        float position;
        u64 timestamp;
    };

    static const u32 MAX_VELOCITY_SAMPLES = 10;
    VelocitySample m_velocity_samples[MAX_VELOCITY_SAMPLES];
    u32 m_sample_count;
    u32 m_sample_index;

};

#endif
