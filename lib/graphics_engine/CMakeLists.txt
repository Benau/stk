include_directories("${PROJECT_SOURCE_DIR}/lib/graphics_engine/include")
include_directories("${PROJECT_SOURCE_DIR}/lib/graphics_utils")
include_directories("${PROJECT_SOURCE_DIR}/lib/irrlicht/include")
include_directories("${PROJECT_SOURCE_DIR}/lib/bullet/src")
include_directories("${PROJECT_SOURCE_DIR}/lib/simd_wrapper")
find_path(SDL2_INCLUDEDIR NAMES SDL.h PATH_SUFFIXES SDL2 include/SDL2 include PATHS)
if (NOT SDL2_INCLUDEDIR)
    message(FATAL_ERROR "SDL2 not found.")
else()
    include_directories("${SDL2_INCLUDEDIR}")
endif()

if(APPLE AND NOT DLOPEN_MOLTENVK)
    find_path(VULKAN_INCLUDEDIR NAMES vulkan/vulkan.h PATHS)
    if (NOT VULKAN_INCLUDEDIR)
        message(FATAL_ERROR "Vulkan not found.")
    else()
        include_directories("${VULKAN_INCLUDEDIR}")
    endif()
endif()

set(GE_SOURCES
    src/gl.c
    src/ge_compressor_astc_4x4.cpp
    src/ge_compressor_bptc_bc7.cpp
    src/ge_compressor_s3tc_bc3.cpp
    src/ge_culling_tool.cpp
    src/ge_dx9_texture.cpp
    src/ge_main.cpp
    src/ge_material_manager.cpp
    src/ge_occlusion_culling.cpp
    src/ge_texture.cpp
    src/ge_vma.cpp
    src/ge_vulkan_2d_renderer.cpp
    src/ge_vulkan_array_texture.cpp
    src/ge_vulkan_animated_mesh_scene_node.cpp
    src/ge_vulkan_anti_aliasing.cpp
    src/ge_vulkan_attachment_texture.cpp
    src/ge_vulkan_billboard_buffer.cpp
    src/ge_vulkan_bloom.cpp
    src/ge_vulkan_camera_scene_node.cpp
    src/ge_vulkan_combined_shadow_fbo.cpp
    src/ge_vulkan_command_loader.cpp
    src/ge_vulkan_deferred_fbo.cpp
    src/ge_vulkan_deferred_fbo_split.cpp
    src/ge_vulkan_driver.cpp
    src/ge_vulkan_draw_call.cpp
    src/ge_vulkan_dynamic_buffer.cpp
    src/ge_vulkan_dynamic_spm_buffer.cpp
    src/ge_vulkan_environment_map.cpp
    src/ge_vulkan_fbo_texture.cpp
    src/ge_vulkan_features.cpp
    src/ge_final_pass.cpp
    src/ge_vulkan_final_chain.cpp
    src/ge_vulkan_depth_of_field.cpp
    src/ge_vulkan_fxaa.cpp
    src/ge_vulkan_glow_outline.cpp
    src/ge_vulkan_god_rays.cpp
    src/ge_vulkan_hiz_depth.cpp
    src/ge_vulkan_light_handler.cpp
    src/ge_vulkan_light_scatter.cpp
    src/ge_vulkan_low_res_effect.cpp
    src/ge_vulkan_rt_shadow.cpp
    src/ge_vulkan_ssao.cpp
    src/ge_vulkan_mesh_cache.cpp
    src/ge_vulkan_rt_scene.cpp
    src/ge_vulkan_mesh_scene_node.cpp
    src/ge_vulkan_omni_shadow_fbo.cpp
    src/ge_vulkan_post_processing.cpp
    src/ge_vulkan_scene_manager.cpp
    src/ge_vulkan_shader_manager.cpp
    src/ge_vulkan_shadow_draw_call.cpp
    src/ge_vulkan_shadow_fbo.cpp
    src/ge_vulkan_skybox_renderer.cpp
    src/ge_vulkan_texture.cpp
    src/ge_vulkan_texture_descriptor.cpp
    src/ge_gl_texture.cpp
    src/ge_spm.cpp
    src/ge_spm_buffer.cpp
)

if(NOT APPLE OR DLOPEN_MOLTENVK)
    set(GE_SOURCES ${GE_SOURCES} src/vulkan.c)
endif()

option(BASISU_TC "Enable basisu transcoder support (requires c++17)" ON)
if(BASISU_TC)
    add_definitions(-DBASISD_SUPPORT_KTX2_ZSTD=0)
    add_definitions(-DBASISU_TC)
    include_directories("${PROJECT_SOURCE_DIR}/lib/graphics_engine/transcoder")
    set(GE_SOURCES
        ${GE_SOURCES}
        "transcoder/basisu_transcoder.cpp"
    )
endif()

if(UNIX OR MINGW)
    if(BASISU_TC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
    else()
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++0x")
    endif()
endif()

add_library(graphics_engine STATIC ${GE_SOURCES})

target_link_libraries(graphics_engine ${SQUISH_LIBRARY})
