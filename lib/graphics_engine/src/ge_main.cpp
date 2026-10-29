#include "ge_main.hpp"
#include "ge_occlusion_culling.hpp"
#include "ge_spm.hpp"
#include "ge_spm_buffer.hpp"
#include "ge_vulkan_driver.hpp"
#include "ge_vulkan_features.hpp"
#include "mini_glm.hpp"

#include "IMesh.h"
#include "IMeshBuffer.h"
#include "S3DVertex.h"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <memory>

namespace GE
{
irr::video::IVideoDriver* g_driver = NULL;
GEConfig g_config =
{
    false,
    false,
    false,
    true,
    false,
    true,
    0,
    false,
    false,
    false,
    false,
    {},
    "",
    false,
    GADT_DISABLED,
    GSRT_DISABLED,
    GDRT_DISABLED,
    {},
    1.0f,
    0,
    GST_NONE,
    8,
    32,
    0.5f,
    3.0f,
    1.0f,
    GAA_NONE,
    false,
    72.0f,
    0.0f,
    360.0f,
    0.0f,
    3.5f
};
std::string g_shader_folder = "";
std::chrono::steady_clock::time_point g_mono_start =
    std::chrono::steady_clock::now();
std::unique_ptr<GEOcclusionCulling> g_occulsion_culling;
std::array<float, 4> g_displace_direction = {};
GELogCallback g_log_callback = NULL;

void setVideoDriver(irr::video::IVideoDriver* driver)
{
    if (driver != g_driver)
    {
        // Reset everytime driver is recreated
        g_config.m_ondemand_load_texture_paths.clear();
        g_config.m_auto_deferred_type = GADT_DISABLED;
        g_driver = driver;
    }
}

irr::video::IVideoDriver* getDriver()
{
    return g_driver;
}

GE::GEVulkanDriver* getVKDriver()
{
    return dynamic_cast<GE::GEVulkanDriver*>(g_driver);
}

GEConfig* getGEConfig()
{
    return &g_config;
}

void setShaderFolder(const std::string& path)
{
    g_shader_folder = path + "ge_shaders/";
}

const std::string& getShaderFolder()
{
    return g_shader_folder;
}

void deinit()
{
}

uint64_t getMonoTimeMs()
{
    auto duration = std::chrono::steady_clock::now() - g_mono_start;
    auto value =
        std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    return value.count();
}

void mathPlaneNormf(float *p)
{
    float f = 1.0f / sqrtf(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
    p[0] *= f;
    p[1] *= f;
    p[2] *= f;
    p[3] *= f;
}

void mathPlaneFrustumf(float* out, const irr::core::matrix4& pvm)
{
    // return 6 planes, 24 floats
    const float* m = pvm.pointer();

    // near
    out[0] = m[3] + m[2];
    out[1] = m[7] + m[6];
    out[2] = m[11] + m[10];
    out[3] = m[15] + m[14];
    mathPlaneNormf(&out[0]);

    // right
    out[4] = m[3] - m[0];
    out[4 + 1] = m[7] - m[4];
    out[4 + 2] = m[11] - m[8];
    out[4 + 3] = m[15] - m[12];
    mathPlaneNormf(&out[4]);

    // left
    out[2 * 4] = m[3] + m[0];
    out[2 * 4 + 1] = m[7] + m[4];
    out[2 * 4 + 2] = m[11] + m[8];
    out[2 * 4 + 3] = m[15] + m[12];
    mathPlaneNormf(&out[2 * 4]);

    // bottom
    out[3 * 4] = m[3] + m[1];
    out[3 * 4 + 1] = m[7] + m[5];
    out[3 * 4 + 2] = m[11] + m[9];
    out[3 * 4 + 3] = m[15] + m[13];
    mathPlaneNormf(&out[3 * 4]);

    // top
    out[4 * 4] = m[3] - m[1];
    out[4 * 4 + 1] = m[7] - m[5];
    out[4 * 4 + 2] = m[11] - m[9];
    out[4 * 4 + 3] = m[15] - m[13];
    mathPlaneNormf(&out[4 * 4]);

    // far
    out[5 * 4] = m[3] - m[2];
    out[5 * 4 + 1] = m[7] - m[6];
    out[5 * 4 + 2] = m[11] - m[10];
    out[5 * 4 + 3] = m[15] - m[14];
    mathPlaneNormf(&out[5 * 4]);
}

irr::scene::IAnimatedMesh* convertIrrlichtMeshToSPM(irr::scene::IMesh* mesh)
{
    GESPM* spm = new GESPM();
    for (unsigned i = 0; i < mesh->getMeshBufferCount(); i++)
    {
        std::vector<video::S3DVertexSkinnedMesh> vertices;
        scene::IMeshBuffer* mb = mesh->getMeshBuffer(i);
        if (!mb)
            continue;

        GESPMBuffer* spm_mb = new GESPMBuffer();
        assert(mb->getVertexType() == video::EVT_STANDARD);
        video::S3DVertex* v_ptr = (video::S3DVertex*)mb->getVertices();
        for (unsigned j = 0; j < mb->getVertexCount(); j++)
        {
            video::S3DVertexSkinnedMesh sp;
            sp.m_position = v_ptr[j].Pos;
            sp.m_normal = MiniGLM::compressVector3(v_ptr[j].Normal);
            video::SColorf orig(v_ptr[j].Color);
            video::SColorf diffuse(mb->getMaterial().DiffuseColor);
            orig.r = orig.r * diffuse.r;
            orig.g = orig.g * diffuse.g;
            orig.b = orig.b * diffuse.b;
            orig.a = orig.a * diffuse.a;
            sp.m_color = orig.toSColor();
            sp.m_all_uvs[0] = MiniGLM::toFloat16(v_ptr[j].TCoords.X);
            sp.m_all_uvs[1] = MiniGLM::toFloat16(v_ptr[j].TCoords.Y);
            spm_mb->getVerticesVector().push_back(sp);
        }
        uint16_t* idx_ptr = mb->getIndices();
        std::vector<uint16_t> indices(idx_ptr, idx_ptr + mb->getIndexCount());
        std::swap(spm_mb->getIndicesVector(), indices);
        spm_mb->getMaterial() = mb->getMaterial();
        spm_mb->recalculateBoundingBox();
        spm->addMeshBuffer(spm_mb);
    }
    spm->finalize();
    return spm;
}

void copyToMappedBuffer(uint32_t* mapped, GESPMBuffer* spmb, size_t offset)
{
    for (unsigned i = offset; i < spmb->getVertexCount(); i++)
    {
        auto& vv = spmb->getVerticesVector();
        memcpy(mapped, &vv[i], 4 * sizeof(uint32_t));
        mapped += 4;
        if (getGEConfig()->m_pbr)
            *mapped = srgb255ToLinearFromSColor(vv[i].m_color).color;
        else
            memcpy(mapped, &vv[i].m_color, sizeof(video::SColor));
        mapped += 1;
        memcpy(mapped, vv[i].m_all_uvs, 3 * sizeof(uint32_t));
        mapped += 3;
    }
}

GEOcclusionCulling* getOcclusionCulling()
{
    if (!g_occulsion_culling)
    {
        g_occulsion_culling = std::unique_ptr<GEOcclusionCulling>(
            new GEOcclusionCulling());
    }
    return g_occulsion_culling.get();
}

void resetOcclusionCulling()
{
    g_occulsion_culling.reset();
}

bool hasOcclusionCulling()
{
    if (g_occulsion_culling)
        return true;
    return false;
}

bool needsDeferredRendering()
{
    return g_config.m_deferred_split != 0 ||
        (g_config.m_pbr && g_config.m_auto_deferred_type != GADT_DISABLED);
}

std::array<float, 4>& getDisplaceDirection()
{
    return g_displace_direction;
}

void setAbsoluteRotationScale(const irr::core::matrix4& model_matrix,
                              irr::core::quaternion& rotation,
                              irr::core::vector3df& scale)
{
    rotation = irr::core::quaternion(0.0f, 0.0f, 0.0f, 1.0f);
    scale = model_matrix.getScale();
    if (scale.X != 0.0f && scale.Y != 0.0f && scale.Z != 0.0f)
    {
        irr::core::matrix4 local_mat = model_matrix;
        local_mat[0] = local_mat[0] / scale.X / local_mat[15];
        local_mat[1] = local_mat[1] / scale.X / local_mat[15];
        local_mat[2] = local_mat[2] / scale.X / local_mat[15];
        local_mat[4] = local_mat[4] / scale.Y / local_mat[15];
        local_mat[5] = local_mat[5] / scale.Y / local_mat[15];
        local_mat[6] = local_mat[6] / scale.Y / local_mat[15];
        local_mat[8] = local_mat[8] / scale.Z / local_mat[15];
        local_mat[9] = local_mat[9] / scale.Z / local_mat[15];
        local_mat[10] = local_mat[10] / scale.Z / local_mat[15];
        rotation = MiniGLM::getQuaternion(local_mat);
        // Conjugated quaternion in glsl
        rotation.W = -rotation.W;
    }
}

void refreshDeferredSplit()
{
    g_config.m_deferred_split = 0;
    if (!g_config.m_pbr)
        return;
    if (g_config.m_glow_outline)
        g_config.m_deferred_split |= 1;
    if (g_config.m_light_scatter)
        g_config.m_deferred_split |= 1 << 1;
    if (g_config.m_bloom)
        g_config.m_deferred_split |= 1 << 2;
    // The solid materials reflect the output of the tonemap pass of the
    // previous frame (the OpenGL renderer also needs the passes to be
    // separated), and the HiZ depth is generated after the g-buffer. The
    // FAST displace reflection only samples the depth buffer
    if (isSolidSSR() ||
        isDisplaceReflectionHiZ(getDisplaceReflection()))
        g_config.m_deferred_split |= 1 << 3;
    // The ambient occlusion is computed between the g-buffer and the lighting
    // passes, and sampled by the lighting pass
    if (g_config.m_ssao)
        g_config.m_deferred_split |= 1 << 4;
    // The passes of ray tracing (the lighting pass needs the visibility of the
    // ray traced shadows from a compute shader before it), which of them
    // is rendered doesn't change the FBO
    if (isRayTracing())
        g_config.m_deferred_split |= 1 << 5;
    // The march of the rays samples the depth of the g-buffer in the compute
    // shaders before the lighting pass
    if (g_config.m_god_rays)
        g_config.m_deferred_split |= 1 << 6;
    // The depth of field (and the passes of the game which read the depth, see
    // GEFinalPassInfo::m_needs_depth) read the depth which the FBO keeps after
    // its last render pass (a regular sampled texture, not a transient
    // attachment)
    if (needsFinalPassDepth())
        g_config.m_deferred_split |= 1 << 7;
}

void setLogCallback(GELogCallback callback)
{
    g_log_callback = callback;
}

void logMessage(GELogLevel level, const char* format, ...)
{
    if (!format)
        return;

    va_list args;
    va_start(args, format);

    va_list args_copy;
    va_copy(args_copy, args);
    int length = std::vsnprintf(NULL, 0, format, args_copy);
    va_end(args_copy);

    if (length < 0)
    {
        va_end(args);
        return;
    }

    std::string message(length + 1, '\0');
    std::vsnprintf(&message[0], message.size(), format, args);
    va_end(args);
    message.resize(length);

    if (g_log_callback)
        g_log_callback(level, message.c_str());
    else
        std::cerr << "[GE] " << message << std::endl;
}

bool isAntiAliasingEnabled()
{
    return g_config.m_anti_aliasing != GAA_NONE &&
        GEVulkanFeatures::supportsComputeInMainQueue();
}

bool isDepthOfFieldEnabled()
{
    return g_config.m_dof && g_config.m_pbr &&
        GEVulkanFeatures::supportsComputeInMainQueue();
}

GEAntiAliasing getAntiAliasingFromName(const std::string& name)
{
    if (name == "fxaa")
        return GAA_FXAA;
    if (name == "cmaa2")
        return GAA_CMAA2;
    return GAA_NONE;
}

bool isRayTracing()
{
    return g_config.m_ray_tracing.m_enabled &&
        GEVulkanFeatures::supportsRayQuery();
}

bool isSolidRayTracing()
{
    return g_config.m_pbr && g_config.m_ray_tracing.m_solid_reflection &&
        isRayTracing();
}

bool isDisplaceRayTracing()
{
    return g_config.m_pbr && g_config.m_ray_tracing.m_displace_reflection &&
        isRayTracing();
}

GEDisplaceReflectionType getDisplaceReflection()
{
    return isDisplaceRayTracing() ? GDRT_RT : g_config.m_displace_reflection;
}

GESolidReflectionType getSolidReflection()
{
    return isSolidRayTracing() ? GSRT_DISABLED : g_config.m_solid_reflection;
}

GEShadowType getRTShadowType()
{
    if (!g_config.m_pbr || !g_config.m_ray_tracing.m_shadow || !isRayTracing())
        return GST_NONE;
    return GEShadowType(g_config.m_shadow_type & GST_COMBINED);
}

float getRTShadowScale()
{
    return std::min(std::max(g_config.m_ray_tracing.m_shadow_scale, 0.25f),
        1.0f);
}

GEShadowType getShadowType()
{
    return GEShadowType(g_config.m_shadow_type & ~getRTShadowType());
}

unsigned getShadowSize()
{
    return getShadowType() == GST_NONE ? 0 : g_config.m_shadow_size;
}

bool isSolidSSR()
{
    return isSolidReflectionSSR(getSolidReflection());
}

bool isSolidHiZ()
{
    return isSolidReflectionHiZ(getSolidReflection());
}

}
