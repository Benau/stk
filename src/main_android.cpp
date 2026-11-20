//  SuperTuxKart - a fun racing game with go-kart
//  Copyright (C) 2016-2017 SuperTuxKart-Team
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

#ifdef MOBILE_STK

#include "config/user_config.hpp"
#include "graphics/irr_driver.hpp"
#include "utils/log.hpp"
#include "utils/string_utils.hpp"

#ifdef ANDROID
#include "SDL_cpuinfo.h"
#include "SDL_loadso.h"
#include "SDL_stdinc.h"
#include "SDL_system.h"
#include <jni.h>
std::string g_android_main_user_agent;

extern int android_main(int argc, char *argv[]);

extern "C" JNIEXPORT void JNICALL debugMsg(JNIEnv* env, jclass cls, jstring msg);
extern "C" JNIEXPORT void JNICALL handlePadding(JNIEnv* env, jclass cls, jboolean val);
extern "C" JNIEXPORT void JNICALL addDNSSrvRecords(JNIEnv* env, jclass cls, jstring name, jint weight);
extern "C" JNIEXPORT void JNICALL surfaceCreatedJNI(JNIEnv* env, jclass cls,
                                                    jlong listener);
extern "C" JNIEXPORT void JNICALL surfaceDestroyedJNI(JNIEnv* env, jclass cls,
                                                      jlong listener);

extern "C" JNIEXPORT void JNICALL editText2STKEditbox(JNIEnv* env, jclass cls, jint widget_id, jstring text, jint start, jint end, jint composing_start, jint composing_end);
extern "C" JNIEXPORT void JNICALL handleActionNext(JNIEnv* env, jclass cls, jint widget_id);
extern "C" JNIEXPORT void JNICALL handleLeftRight(JNIEnv* env, jclass cls, jboolean left, jint widget_id);

#if !defined(ANDROID_PACKAGE_CLASS_NAME)
    #error
#endif

// ----------------------------------------------------------------------------
/** Tells the java activity if the native methods of it can be called, see
 *  m_natives_registered in SuperTuxKartActivity.java.
 */
void set_natives_registered(bool registered)
{
    JNIEnv* env = (JNIEnv*)SDL_AndroidGetJNIEnv();
    if (!env)
        return;
    const char* stkactivity_class =
        ANDROID_PACKAGE_CLASS_NAME "/SuperTuxKartActivity";
    jclass clazz = env->FindClass(stkactivity_class);
    if (clazz == NULL)
    {
        env->ExceptionClear();
        return;
    }
    jfieldID field = env->GetStaticFieldID(clazz, "m_natives_registered", "Z");
    if (field == NULL)
    {
        env->ExceptionClear();
        Log::error("MainAndroid", "Failed to find m_natives_registered.");
    }
    else
        env->SetStaticBooleanField(clazz, field, registered);
    env->DeleteLocalRef(clazz);
}   // set_natives_registered

// ----------------------------------------------------------------------------
void registering_natives()
{
    JNINativeMethod stkactivity_tab[] =
    {
        { "debugMsg",           "(Ljava/lang/String;)V", (void*)&debugMsg },
        { "handlePadding",      "(Z)V", (void*)&handlePadding },
        { "addDNSSrvRecords",   "(Ljava/lang/String;I)V", (void*)&addDNSSrvRecords },
        { "surfaceCreatedJNI",   "(J)V", (void*)&surfaceCreatedJNI },
        { "surfaceDestroyedJNI", "(J)V", (void*)&surfaceDestroyedJNI }
    };
    JNIEnv* env = (JNIEnv*)SDL_AndroidGetJNIEnv();
    assert(env);
    const char* stkactivity_class = ANDROID_PACKAGE_CLASS_NAME "/SuperTuxKartActivity";
    jclass clazz = env->FindClass(stkactivity_class);
    if (clazz == NULL)
    {
        Log::error("MainAndroid", "Failed to find class %s.",
            stkactivity_class);
        return;
    }
    if (env->RegisterNatives(
        clazz, stkactivity_tab, (int)SDL_arraysize(stkactivity_tab)) < 0)
    {
        Log::error("MainAndroid", "Failed to register methods of %s.",
            stkactivity_class);
    }
    else
        set_natives_registered(true);

    JNINativeMethod stkeditbox_tab[] =
    {
        { "editText2STKEditbox", "(ILjava/lang/String;IIII)V", (void*)&editText2STKEditbox },
        { "handleActionNext",    "(I)V", (void*)&handleActionNext },
        { "handleLeftRight",     "(ZI)V", (void*)&handleLeftRight }
    };
    const char* stkeditbox_class = ANDROID_PACKAGE_CLASS_NAME "/STKEditText";
    clazz = env->FindClass(stkeditbox_class);
    if (clazz == NULL)
    {
        Log::error("MainAndroid", "Failed to find class %s.",
            stkeditbox_class);
        return;
    }
    if (env->RegisterNatives(
        clazz, stkeditbox_tab, (int)SDL_arraysize(stkeditbox_tab)) < 0)
    {
        Log::error("MainAndroid", "Failed to register methods of %s.",
            stkeditbox_class);
    }
}

void override_default_params_for_mobile();
// TEMP firebase format probe: race arguments (stk_temp_args in the media dir
// replaces them)
#include <cstring>
#include <fstream>
#include <vector>
static std::vector<std::string> g_temp_args;
static std::vector<char*> g_temp_argv;
static void temp_load_args(int& argc, char**& argv)
{
    const std::string d = "/sdcard/Android/media/org.supertuxkart.stk_dbg/";
    for (int i = 0; i < argc; i++)
        g_temp_args.push_back(argv[i]);
    if (g_temp_args.empty())
        g_temp_args.push_back("stk");
    std::ifstream args(d + "stk_temp_args");
    std::string a;
    bool from_file = false;
    while (args >> a)
    {
        g_temp_args.push_back(a);
        from_file = true;
    }
    if (!from_file)
    {
        // Game loop scenario 2 = OpenGL, anything else Vulkan
        int scenario = 0;
        JNIEnv* env = (JNIEnv*)SDL_AndroidGetJNIEnv();
        jobject activity = (jobject)SDL_AndroidGetActivity();
        if (env && activity)
        {
            jclass ac = env->GetObjectClass(activity);
            jmethodID gi = env->GetMethodID(ac, "getIntent",
                "()Landroid/content/Intent;");
            jobject intent = env->CallObjectMethod(activity, gi);
            if (intent)
            {
                jclass ic = env->GetObjectClass(intent);
                jmethodID ge = env->GetMethodID(ic, "getIntExtra",
                    "(Ljava/lang/String;I)I");
                jstring key = env->NewStringUTF("scenario");
                scenario = env->CallIntMethod(intent, ge, key, 0);
                env->DeleteLocalRef(key);
                env->DeleteLocalRef(ic);
                env->DeleteLocalRef(intent);
            }
            env->DeleteLocalRef(ac);
            env->DeleteLocalRef(activity);
            if (env->ExceptionCheck())
                env->ExceptionClear();
        }
        // 1 Vulkan auto, 2 Vulkan forward, 3 Vulkan split deferred,
        // 4 Vulkan forward + render scale 1, 5 Vulkan without dynamic
        // lights, 6 OpenGL
        // Run 5 (see the table in the notes)
        const char* modes[] = { "auto", "auto", "auto+dn", "auto+dn+ds",
            "auto+cn", "auto+cn+cs", "auto+dn+ds+cn+cs", "forward",
            "forward+dn", "forward+dn+ds", "nopbr+dn", "nopbr+dn+ds", "split" };
        const char* mode = scenario >= 1 && scenario <= 12 ?
            modes[scenario] : "auto";
        setenv("STK_TEMP_GE_MODE", mode, 1);
        Log::info("TEMPSTAT", "game loop scenario %d mode %s", scenario,
            mode);
        g_temp_args.push_back(strcmp(mode, "gl") == 0 ?
            "--render-driver=opengl" : "--render-driver=vulkan");
        g_temp_args.push_back("--track=cornfield_mirror");
        g_temp_args.push_back("--profile-time=60");
        g_temp_args.push_back("--numkarts=4");
    }
    for (auto& s : g_temp_args)
        g_temp_argv.push_back((char*)s.c_str());
    g_temp_argv.push_back(NULL);
    argc = (int)g_temp_args.size();
    argv = g_temp_argv.data();
}

extern "C" int SDL_main(int argc, char *argv[])
{
    temp_load_args(argc, argv); // TEMP firebase
    registering_natives();
    override_default_params_for_mobile();
    int ret = android_main(argc, argv);
    // SDL unloads libmain.so after this returns, the registered methods
    // would point to unloaded code until the next SDL_main registers them
    // again (when the activity is opened again in the same process)
    set_natives_registered(false);
    return ret;
}
#endif

extern "C" void* LoadVulkanDrivers(const char *path);
extern "C" const char* Android_Custom_Vulkan_Driver_In_Use();

void override_default_params_for_mobile()
{
    // It has an effect only on the first run, when config file is created.
    // So that we can still modify these params in STK options and user's
    // choice will be then remembered.
    
    // Set smaller texture size to avoid high RAM usage
    UserConfigParams::m_max_texture_size = 256;
    UserConfigParams::m_high_definition_textures = false;

    // iOS getConfigForDevice will override it later
    UserConfigParams::m_dynamic_lights = false;
    // Enable advanced lighting only for android >= 8
#ifdef ANDROID
    UserConfigParams::m_dynamic_lights = (SDL_GetAndroidSDKVersion() >= 26);

    // Adrenotools requires >= Android 9+ Arm64
#if defined(__aarch64__)
    if (SDL_GetAndroidSDKVersion() >= 28)
    {
        void* obj = LoadVulkanDrivers("libvulkan.so");
        const char* driver = Android_Custom_Vulkan_Driver_In_Use();
        if (driver != NULL)
        {
            Log::info("MainAndroid",
                "Better vulkan driver %s supported.", driver);
            UserConfigParams::m_render_driver = "vulkan";
            UserConfigParams::m_texture_compression = false;
        }
        if (obj)
            SDL_UnloadObject(obj);
    }
#endif

#endif

    // Disable light scattering for better performance
    UserConfigParams::m_light_scatter = false;

    // Enable multitouch race GUI
    UserConfigParams::m_multitouch_draw_gui = true;

#ifdef ANDROID
    // For usage in StringUtils::getUserAgentString
    if (SDL_IsAndroidTV())
    {
        // For some android tv sdl returns a touch screen device even it doesn't
        // have
        UserConfigParams::m_multitouch_draw_gui = false;
        g_android_main_user_agent = " (AndroidTV)";
    }
    else if (SDL_IsChromebook())
        g_android_main_user_agent = " (ChromeOS)";
    else
        g_android_main_user_agent = " (Android)";

    // Get some info about display
    const int SCREENLAYOUT_SIZE_SMALL = 1;
    const int SCREENLAYOUT_SIZE_NORMAL = 2;
    const int SCREENLAYOUT_SIZE_LARGE = 3;
    const int SCREENLAYOUT_SIZE_XLARGE = 4;
    int32_t screen_size = 0;
    int ddpi = 0;
    JNIEnv* env = (JNIEnv*)SDL_AndroidGetJNIEnv();
    assert(env);
    jobject activity = (jobject)SDL_AndroidGetActivity();
    if (activity != NULL)
    {
        jclass clazz = env->GetObjectClass(activity);
        if (clazz != NULL)
        {
            jmethodID method_id = env->GetMethodID(clazz, "getScreenSize",
                "()I");
            if (method_id != NULL)
                screen_size = env->CallIntMethod(activity, method_id);
                
            jmethodID display_dpi_id = env->GetStaticMethodID(clazz,
                "getDisplayDPI", "()Landroid/util/DisplayMetrics;");
                
            if (display_dpi_id != NULL)
            {
                jobject display_dpi_obj = env->CallStaticObjectMethod(clazz,
                                                                display_dpi_id);
                jclass display_dpi_class = env->GetObjectClass(display_dpi_obj);
            
                jfieldID ddpi_field = env->GetFieldID(display_dpi_class,
                                                      "densityDpi", "I");
                ddpi = env->GetIntField(display_dpi_obj, ddpi_field);
            
                env->DeleteLocalRef(display_dpi_obj);
                env->DeleteLocalRef(display_dpi_class);
            }
            
            env->DeleteLocalRef(activity);
            env->DeleteLocalRef(clazz);
        }
    }

    // Set multitouch device scale depending on actual screen size
    switch (screen_size)
    {
    case SCREENLAYOUT_SIZE_SMALL:
    case SCREENLAYOUT_SIZE_NORMAL:
        UserConfigParams::m_multitouch_scale.setDefaultValue(1.3f);
        UserConfigParams::m_multitouch_sensitivity_x.setDefaultValue(0.1f);
        UserConfigParams::m_font_size = 5.0f;
        break;
    case SCREENLAYOUT_SIZE_LARGE:
        UserConfigParams::m_multitouch_scale.setDefaultValue(1.2f);
        UserConfigParams::m_multitouch_sensitivity_x.setDefaultValue(0.15f);
        UserConfigParams::m_font_size = 5.0f;
        break;
    case SCREENLAYOUT_SIZE_XLARGE:
        UserConfigParams::m_multitouch_scale.setDefaultValue(1.1f);
        UserConfigParams::m_multitouch_sensitivity_x.setDefaultValue(0.2f);
        UserConfigParams::m_font_size = 4.0f;
        break;
    default:
        break;
    }
    
    int total_memory = ceilf((float)(SDL_GetSystemRAM()) / 1024);
    
    if ((total_memory >= 8) && (SDL_GetAndroidSDKVersion() >= 31))
    {
        // Use 100% scale on devices with 8GB RAM and Android >= 12
        UserConfigParams::m_scale_rtts_factor = 1.0f;
    }
    else if (ddpi < 1)
    {
        Log::warn("MainAndroid", "Failed to get display DPI.");
        UserConfigParams::m_scale_rtts_factor = 0.7f;
    }
    else
    {
        // Update rtts scale based on display DPI
        if (ddpi > 400)
            UserConfigParams::m_scale_rtts_factor = 0.6f;
        else if (ddpi > 300)
            UserConfigParams::m_scale_rtts_factor = 0.65f;
        else if (ddpi > 200)
            UserConfigParams::m_scale_rtts_factor = 0.7f;
        else if (ddpi > 150)
            UserConfigParams::m_scale_rtts_factor = 0.75f;
        else
            UserConfigParams::m_scale_rtts_factor = 0.8f;

        Log::info("MainAndroid", "Display DPI: %i", ddpi);
    }

    Log::info("MainAndroid", "Render scale: %f",
              (float)UserConfigParams::m_scale_rtts_factor);
#endif

    // Enable screen keyboard
    UserConfigParams::m_screen_keyboard = 1;
    
    // It shouldn't matter, but STK is always run in fullscreen on android
    UserConfigParams::m_fullscreen = true;
    
    // Make sure that user can play every track even if there are installed
    // only few tracks and it's impossible to finish overworld challenges
    UserConfigParams::m_unlock_everything = 1;
    
    // Create default user istead of showing login screen to make life easier
    UserConfigParams::m_enforce_current_player = true;
}

#ifdef IOS_STK
void getConfigForDevice(const char* dev)
{
    // Check browser.geekbench.com/ios-benchmarks metal benchmark
    // https://gist.github.com/adamawolf/3048717 for device name mapping
    std::string device = dev;
    if (device.find("iPhone") != std::string::npos)
    {
        // Normal configuration default
        UserConfigParams::m_multitouch_scale.setDefaultValue(1.3f);
        UserConfigParams::m_multitouch_sensitivity_x.setDefaultValue(0.1f);
        UserConfigParams::m_font_size = 5.0f;
        device.erase(0, 6);
        auto versions = StringUtils::splitToUInt(device, ',');
        if (versions.size() == 2)
        {
            // A11 GPU
            if (versions[0] >= 10)
            {
                UserConfigParams::m_degraded_IBL = false;
            }
            // A12 GPU
            if (versions[0] >= 11)
            {
                UserConfigParams::m_shadows_resolution = 512;
            }
            // A9 GPU
            if (versions[0] >= 8)
            {
                UserConfigParams::m_dynamic_lights = true;
                UserConfigParams::m_high_definition_textures = 1;
            }
            if (versions[0] < 7 || // iPhone 5s
                (versions[0] == 7 && versions[1] == 2) || // iPhone 6
                (versions[0] == 8 && versions[1] == 1) || // iPhone 6S
                (versions[0] == 8 && versions[1] == 4) || // iPhone SE
                (versions[0] == 9 && versions[1] == 1) || // iPhone 7
                (versions[0] == 9 && versions[1] == 3) || // iPhone 7
                (versions[0] == 10 && versions[1] == 1) || // iPhone 8
                (versions[0] == 10 && versions[1] == 4) // iPhone 8
                )
            {
                // Those phones have small screen
                UserConfigParams::m_multitouch_scale.setDefaultValue(1.45f);
                if (UserConfigParams::m_dynamic_lights)
                    UserConfigParams::m_scale_rtts_factor = 0.8f;
            }
        }
    }
    else if (device.find("iPad") != std::string::npos)
    {
        // Normal configuration default
        UserConfigParams::m_multitouch_scale.setDefaultValue(0.95f);
        UserConfigParams::m_multitouch_sensitivity_x.setDefaultValue(0.2f);
        UserConfigParams::m_font_size = 3.0f;
        device.erase(0, 4);
        auto versions = StringUtils::splitToUInt(device, ',');
        if (versions.size() == 2)
        {
            if (versions[0] >= 11)
            {
                UserConfigParams::m_degraded_IBL = false;
                UserConfigParams::m_shadows_resolution = 512;
            }
            if (versions[0] >= 7)
            {
                UserConfigParams::m_dynamic_lights = true;
                UserConfigParams::m_high_definition_textures = 1;
            }
        }
    }
    else if (device.find("iPod") != std::string::npos)
    {
        // All iPod touch has small screen
        UserConfigParams::m_font_size = 5.0f;
        UserConfigParams::m_multitouch_scale.setDefaultValue(1.45f);
        UserConfigParams::m_multitouch_sensitivity_x.setDefaultValue(0.1f);
        device.erase(0, 4);
        auto versions = StringUtils::splitToUInt(device, ',');
        if (versions.size() == 2)
        {
            // iPod Touch 7th Generation (A10)
            if (versions[0] >= 9)
            {
                UserConfigParams::m_dynamic_lights = true;
                UserConfigParams::m_high_definition_textures = 1;
                UserConfigParams::m_scale_rtts_factor = 0.8f;
            }
        }
    }
    UserConfigParams::m_render_driver = "vulkan";
    // Texture compression is slow, and with vulkan we don't load all kart
    // textures anyway
    UserConfigParams::m_texture_compression = false;
}

#endif

#endif
