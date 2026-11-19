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

#include "guiengine/engine.hpp"
#ifdef MOBILE_STK
#include "guiengine/scalable_font.hpp"
#include "utils/string_utils.hpp"
#include "utils/time.hpp"
#endif
#include "guiengine/modaldialog.hpp"
#include "guiengine/screen_keyboard.hpp"
#include "guiengine/widgets/text_box_widget.hpp"
#include "guiengine/widgets/CGUIEditBox.hpp"
#include "utils/ptr_vector.hpp"
#include "utils/translation.hpp"
#include "utils/utf8/core.h"
#include "utils/utf8.h"

#include <IGUIElement.h>
#include <IGUIEnvironment.h>
#include <IGUIButton.h>
#include <IrrlichtDevice.h>

using namespace irr;

class MyCGUIEditBox : public CGUIEditBox
{
    PtrVector<GUIEngine::ITextBoxWidgetListener, REF> m_listeners;

#ifdef MOBILE_STK
    enum class PasteTip { Hidden, Holding, Visible, Pressed };
    PasteTip m_paste_tip;
    core::position2di m_press_pos;
    uint64_t m_long_press_time;
    core::recti m_paste_rect;
    core::stringw m_paste_label;
    GUIEngine::SkinWidgetContainer m_paste_skin;

    void checkLongPress()
    {
        if (m_paste_tip != PasteTip::Holding ||
            StkTime::getMonoTimeMs() < m_long_press_time)
            return;
        m_paste_label = _("Paste");
        const auto size = GUIEngine::getSmallFont()->getDimension(
            m_paste_label.c_str());
        m_paste_rect = core::recti(0, 0, size.Width + size.Height,
            std::max<s32>(size.Height, AbsoluteClippingRect.getHeight()));
        m_paste_tip = PasteTip::Visible;
    }

    void trackLongPress(const SEvent::SMouseInput& mouse)
    {
        const core::position2di pos(mouse.X, mouse.Y);
        if (mouse.Event == EMIE_LMOUSE_PRESSED_DOWN &&
            AbsoluteClippingRect.isPointInside(pos))
        {
            m_press_pos = pos;
            m_long_press_time = StkTime::getMonoTimeMs() + 500;
            m_paste_tip = PasteTip::Holding;
        }
        else if (MouseMarking && (mouse.Event == EMIE_MOUSE_MOVED ||
                 mouse.Event == EMIE_LMOUSE_LEFT_UP))
        {
            const s32 slop = std::max(1, AbsoluteClippingRect.getHeight() / 4);
            if (pos.getDistanceFromSQ(m_press_pos) > slop * slop)
                m_paste_tip = PasteTip::Hidden;
            if (mouse.Event == EMIE_LMOUSE_LEFT_UP)
            {
                checkLongPress();
                if (m_paste_tip == PasteTip::Holding)
                    m_paste_tip = PasteTip::Hidden;
            }
        }
    }

public:
    void drawPasteTip()
    {
        if (!isVisible() || !isEnabled())
        {
            m_paste_tip = PasteTip::Hidden;
            return;
        }
        checkLongPress();
        if (m_paste_tip != PasteTip::Visible && m_paste_tip != PasteTip::Pressed)
            return;

        auto* device = GUIEngine::getDevice();
        const auto screen = Environment->getRootGUIElement()->getAbsolutePosition();
        const s32 gap = std::max(1, m_paste_rect.getHeight() / 4);
        const s32 moved = device->getMovedHeight();
        const s32 keyboard = device->getOnScreenKeyboardHeight();
        const core::recti bounds(screen.UpperLeftCorner.X + device->getLeftPadding() + gap,
            moved + device->getTopPadding() + gap,
            screen.LowerRightCorner.X - device->getRightPadding() - gap,
            screen.LowerRightCorner.Y + moved - std::max(keyboard, device->getBottomPadding()) - gap);
        const s32 caret_x = CurrentTextRect.UpperLeftCorner.X + m_cursor_distance;
        s32 top = AbsoluteClippingRect.UpperLeftCorner.Y - gap - m_paste_rect.getHeight();
        if (top < bounds.UpperLeftCorner.Y)
            top = AbsoluteClippingRect.LowerRightCorner.Y + gap;
        m_paste_rect = core::recti(core::position2di(caret_x - m_paste_rect.getWidth() / 2, top),
                                   m_paste_rect.getSize());
        m_paste_rect.constrainTo(bounds);
        GUIEngine::Skin* skin = GUIEngine::getSkin();
        skin->drawMessage(&m_paste_skin, m_paste_rect, "tooltip::neutral");
        GUIEngine::getSmallFont()->draw(m_paste_label, m_paste_rect,
            skin->getColor("text::neutral"), true, true, &m_paste_rect);
    }

    bool handlePasteTipMouse(const SEvent::SMouseInput& mouse)
    {
        if (m_paste_tip != PasteTip::Visible && m_paste_tip != PasteTip::Pressed)
            return false;
        const bool inside = m_paste_rect.isPointInside({mouse.X, mouse.Y});
        if (m_paste_tip == PasteTip::Pressed)
        {
            if (mouse.Event == EMIE_LMOUSE_LEFT_UP)
            {
                m_paste_tip = PasteTip::Hidden;
                if (inside && pasteFromClipboard())
                    handleTextUpdated();
            }
            return true;
        }
        if (mouse.Event == EMIE_LMOUSE_PRESSED_DOWN)
            m_paste_tip = inside ? PasteTip::Pressed : PasteTip::Hidden;
        return inside && !MouseMarking;
    }
#endif

public:

    MyCGUIEditBox(const wchar_t* text, bool border, gui::IGUIEnvironment* environment,
                 gui:: IGUIElement* parent, s32 id, const core::rect<s32>& rectangle) :
        CGUIEditBox(text, border, environment, parent, id, rectangle)
    {
#ifdef MOBILE_STK
        m_paste_tip = PasteTip::Hidden;
        m_long_press_time = 0;
#endif
    }

    void addListener(GUIEngine::ITextBoxWidgetListener* listener)
    {
        m_listeners.push_back(listener);
    }

    void clearListeners()
    {
        m_listeners.clearWithoutDeleting();
    }

    void handleTextUpdated()
    {
#ifdef MOBILE_STK
        m_paste_tip = PasteTip::Hidden;
#endif
        for (unsigned n = 0; n < m_listeners.size(); n++)
            m_listeners[n].onTextUpdated();
        GUIEngine::setWithinATextBox(true);
    }

    bool handleEnterPressed()
    {
        bool handled = false;
        for (unsigned n = 0; n < m_listeners.size(); n++)
        {
            if (m_listeners[n].onEnterPressed(Text))
            {
                handled = true;
                setText(L"");
            }
        }
        return handled;
    }

    virtual void inputChar(char32_t c)
    {
        CGUIEditBox::inputChar(c);
        handleTextUpdated();
    }

    virtual bool OnEvent(const SEvent& event)
    {
#ifdef MOBILE_STK
        if (event.EventType == EET_MOUSE_INPUT_EVENT && isEnabled() &&
            Environment->hasFocus(this))
            trackLongPress(event.MouseInput);
        else if (event.EventType == EET_SDL_TEXT_EVENT ||
            (event.EventType == EET_GUI_EVENT &&
             event.GUIEvent.EventType == EGET_ELEMENT_FOCUS_LOST))
            m_paste_tip = PasteTip::Hidden;
#endif
        bool out = CGUIEditBox::OnEvent(event);

        if (event.EventType == EET_KEY_INPUT_EVENT &&
            event.KeyInput.PressedDown)
            handleTextUpdated();

        if (event.EventType == EET_KEY_INPUT_EVENT &&
            event.KeyInput.Key == IRR_KEY_RETURN)
            handleEnterPressed();

        return out;
    }
};

using namespace GUIEngine;
using namespace irr::core;
using namespace irr::gui;
using namespace irr;

#ifdef MOBILE_STK
void TextBoxWidget::drawFocusedPasteTip()
{
    auto* box = dynamic_cast<MyCGUIEditBox*>(getGUIEnv()->getFocus());
    if (box)
        box->drawPasteTip();
}

bool TextBoxWidget::handleFocusedPasteTipMouse(const SEvent::SMouseInput& mouse)
{
    auto* env = getGUIEnv();
    auto* box = env ? dynamic_cast<MyCGUIEditBox*>(env->getFocus()) : NULL;
    return box && box->handlePasteTipMouse(mouse);
}
#endif

// -----------------------------------------------------------------------------

TextBoxWidget::TextBoxWidget() : Widget(WTYPE_TEXTBOX)
{
}

// -----------------------------------------------------------------------------

void TextBoxWidget::add()
{
    rect<s32> widget_size = rect<s32>(m_x, m_y, m_x + m_w, m_y + m_h);

    // Don't call TextBoxWidget::getText(), which assumes that the irrlicht
    // widget already exists.
    const stringw& text = Widget::getText();

    m_element = new MyCGUIEditBox(text.c_str(), true /* border */, GUIEngine::getGUIEnv(),
                                  (m_parent ? m_parent : GUIEngine::getGUIEnv()->getRootGUIElement()),
                                  getNewID(), widget_size);

    if (m_deactivated)
        m_element->setEnabled(false);

    //m_element = GUIEngine::getGUIEnv()->addEditBox(text.c_str(), widget_size,
    //                                               true /* border */, m_parent, getNewID());
    m_id = m_element->getID();
    m_element->setTabOrder(m_id);
    m_element->setTabGroup(false);
    m_element->setTabStop(true);
}

// -----------------------------------------------------------------------------

void TextBoxWidget::addListener(ITextBoxWidgetListener* listener)
{
    ((MyCGUIEditBox*)m_element)->addListener(listener);
}

// -----------------------------------------------------------------------------

void TextBoxWidget::clearListeners()
{
    ((MyCGUIEditBox*)m_element)->clearListeners();
}

// -----------------------------------------------------------------------------

stringw TextBoxWidget::getText() const
{
    const IGUIEditBox* textCtrl =  Widget::getIrrlichtElement<IGUIEditBox>();
    assert(textCtrl != NULL);

    return stringw(textCtrl->getText());
}

// -----------------------------------------------------------------------------

void TextBoxWidget::setPasswordBox(bool passwordBox, wchar_t passwordChar)
{
    IGUIEditBox* textCtrl =  Widget::getIrrlichtElement<IGUIEditBox>();
    assert(textCtrl != NULL);
    textCtrl->setPasswordBox(passwordBox, passwordChar);
    setTextBoxType(TBT_PASSWORD);
}

// -----------------------------------------------------------------------------
void TextBoxWidget::setTextBoxType(TextBoxType t)
{
    ((MyCGUIEditBox*)m_element)->setTextBoxType(t);
}

// -----------------------------------------------------------------------------

EventPropagation TextBoxWidget::focused(const int playerID)
{
    assert(playerID == 0); // No support for multiple players in text areas!

    // special case : to work, the text box must receive "irrLicht focus", STK focus is not enough
    GUIEngine::getGUIEnv()->setFocus(m_element);
    setWithinATextBox(true);
    return EVENT_LET;
}

// -----------------------------------------------------------------------------

void TextBoxWidget::unfocused(const int playerID, Widget* new_focus)
{
    assert(playerID == 0); // No support for multiple players in text areas!

    setWithinATextBox(false);
    
    GUIEngine::getGUIEnv()->removeFocus(m_element);
}

// -----------------------------------------------------------------------------

void TextBoxWidget::elementRemoved()
{
    // If text box is destroyed without moving mouse out of its focus, the
    // reference will be kept so we clear it manually
    GUIEngine::getGUIEnv()->removeHovered(m_element);
    // normally at this point normal widgets have been deleted by irrlicht already.
    // but this is a custom widget and the gui env does not appear to want to
    // manage it. so we free it manually
    m_element->drop();
    Widget::elementRemoved();
}

// -----------------------------------------------------------------------------

void TextBoxWidget::setActive(bool active)
{
    Widget::setActive(active);

    if (m_element != NULL)
    {
        IGUIEditBox* textCtrl = Widget::getIrrlichtElement<IGUIEditBox>();
        assert(textCtrl != NULL);
        textCtrl->setEnabled(active);
    }
}   // setActive

// -----------------------------------------------------------------------------

EventPropagation TextBoxWidget::onActivationInput(const int playerID)
{
    if (GUIEngine::ScreenKeyboard::shouldUseScreenKeyboard())
    {
        ((MyCGUIEditBox*)m_element)->openScreenKeyboard();
    }

    // The onWidgetActivated() wasn't used at all before, so always block
    // event to avoid breaking something
    return EVENT_BLOCK;
}

// -----------------------------------------------------------------------------
EventPropagation TextBoxWidget::rightPressed(const int playerID)
{
    if (((MyCGUIEditBox*)m_element)->getTextCount() ==
        ((MyCGUIEditBox*)m_element)->getCursorPosInBox())
        return EVENT_BLOCK;

    return EVENT_LET;
}   // rightPressed

// -----------------------------------------------------------------------------
EventPropagation TextBoxWidget::leftPressed (const int playerID)
{
    if (((MyCGUIEditBox*)m_element)->getCursorPosInBox() == 0)
        return EVENT_BLOCK;

    return EVENT_LET;
}   // leftPressed

// ============================================================================
/* This callback will allow copying android edittext data directly to editbox,
 * which will allow composing text to be auto updated. */
#ifdef ANDROID
#include "jni.h"

extern "C" JNIEXPORT void JNICALL editText2STKEditbox(JNIEnv* env, jclass cls, jint widget_id, jstring text, jint start, jint end, jint composing_start, jint composing_end)
{
    if (text == NULL)
        return;

    const uint16_t* utf16_text = (const uint16_t*)env->GetStringChars(text, NULL);
    if (utf16_text == NULL)
        return;
    const size_t len = env->GetStringLength(text);
    // Android use 32bit wchar_t and java use utf16 string
    // We should not use the modified utf8 from java as it fails for emoji
    // because it's larger than 16bit
    std::u32string to_editbox;

    std::vector<int> mappings;
    int pos = 0;
    mappings.push_back(pos++);
    for (unsigned i = 0; i < len; i++)
    {
        if (utf8::internal::is_lead_surrogate(utf16_text[i]))
        {
            int duplicated_pos = pos++;
            mappings.push_back(duplicated_pos);
            mappings.push_back(duplicated_pos);
            i++;
        }
        else
            mappings.push_back(pos++);
    }

    // Correct start / end position for utf16
    if (start < (int)mappings.size())
        start = mappings[start];
    if (end < (int)mappings.size())
        end = mappings[end];
    if (composing_start < (int)mappings.size())
        composing_start = mappings[composing_start];
    if (composing_end < (int)mappings.size())
        composing_end = mappings[composing_end];

    try
    {
        utf8::utf16to32(utf16_text, utf16_text + len,
            back_inserter(to_editbox));
    }
    catch (std::exception& e)
    {
        (void)e;
    }
    env->ReleaseStringChars(text, utf16_text);

    GUIEngine::addGUIFunctionBeforeRendering([widget_id, to_editbox, start,
        end, composing_start, composing_end]()
        {
            TextBoxWidget* tb =
                dynamic_cast<TextBoxWidget*>(getFocusForPlayer(0));
            if (!tb || (int)widget_id != tb->getID())
                return;
            MyCGUIEditBox* eb = tb->getIrrlichtElement<MyCGUIEditBox>();
            if (!eb)
                return;
            core::stringw old_text = eb->getText();
            eb->fromAndroidEditText(to_editbox, start, end, composing_start,
                composing_end);
            if (old_text != eb->getText())
                eb->handleTextUpdated();
        });
}

extern void Android_toggleOnScreenKeyboard(bool show, int type, int y);
extern bool Android_isHardwareKeyboardConnected();

extern "C" JNIEXPORT void JNICALL handleActionNext(JNIEnv* env, jclass cls, jint widget_id)
{
    GUIEngine::addGUIFunctionBeforeRendering([widget_id]()
        {
            TextBoxWidget* tb =
                dynamic_cast<TextBoxWidget*>(getFocusForPlayer(0));
            if (!tb || (int)widget_id != tb->getID())
                return;
            MyCGUIEditBox* eb = tb->getIrrlichtElement<MyCGUIEditBox>();
            if (!eb)
                return;

            bool has_hardware_keyboard = Android_isHardwareKeyboardConnected();
            // First test for onEnterPressed, if true then close keyboard
            if (eb->handleEnterPressed())
            {
                // If hardware keyboard connected don't out focus stk edittext
                if (has_hardware_keyboard)
                    tb->setText(L"");
                else
                {
                    // Clear text like onEnterPressed callback
                    Android_toggleOnScreenKeyboard(false, 1/*clear_text*/, 0/*y*/);
                }
                return;
            }
            else
            {
                // Required for some general text field dialog
                eb->sendGuiEvent(EGET_EDITBOX_ENTER);
            }

            // As it's action "next", check if below widget is a text box, if
            // so focus it and keep the screen keyboard, so user can keep
            // typing
            int id = GUIEngine::EventHandler::get()->findIDClosestWidget(
                NAV_DOWN, 0, tb, true/*ignore_disabled*/);
            TextBoxWidget* closest_tb = NULL;
            if (id != -1)
            {
                closest_tb =
                    dynamic_cast<TextBoxWidget*>(GUIEngine::getWidget(id));
            }

            if (closest_tb)
            {
                closest_tb->setFocusForPlayer(0);
            }
            else if (!has_hardware_keyboard)
            {
                // Post an enter event and close the keyboard
                SEvent enter_event;
                enter_event.EventType = EET_KEY_INPUT_EVENT;
                enter_event.KeyInput.Char = 0;
                enter_event.KeyInput.PressedDown = true;
                enter_event.KeyInput.Key = IRR_KEY_RETURN;
                GUIEngine::getDevice()->postEventFromUser(enter_event);
                enter_event.KeyInput.PressedDown = false;
                GUIEngine::getDevice()->postEventFromUser(enter_event);
                Android_toggleOnScreenKeyboard(false, 1/*clear_text*/, 0/*y*/);
            }
        });
}

extern "C" JNIEXPORT void JNICALL handleLeftRight(JNIEnv* env, jclass cls, jboolean left, jint widget_id)
{
    GUIEngine::addGUIFunctionBeforeRendering([left, widget_id]()
        {
            // Handle left, right pressed in android edit box, out focus it if
            // no widget found
            TextBoxWidget* tb =
                dynamic_cast<TextBoxWidget*>(getFocusForPlayer(0));
            if (!tb || (int)widget_id != tb->getID())
                return;
            int id = GUIEngine::EventHandler::get()->findIDClosestWidget(
                left ? NAV_LEFT: NAV_RIGHT, 0, tb, true/*ignore_disabled*/);
            Widget* next_widget = GUIEngine::getWidget(id);
            if (next_widget)
                next_widget->setFocusForPlayer(0);
            else
                tb->unsetFocusForPlayer(0);
        });
}

#endif
