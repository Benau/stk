#ifndef HEADER_GE_FINAL_PASS_HPP
#define HEADER_GE_FINAL_PASS_HPP

#include "vulkan_wrapper.h"

#include "dimension2d.h"
#include "matrix4.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace irr
{
    namespace scene { class ICameraSceneNode; }
}

namespace GE
{
class GEVulkanCamera;
class GEVulkanDrawCall;
class GEVulkanDriver;

// One step of the final chain (see GEVulkanFinalChain), rendered with compute
// shaders after the FBO is rendered and before it is sampled by anything (the
// 2D renderer, or the users of a separate render target): the depth of field
// and the anti-aliasing of GE are two of them, and the game can add its own
// with addFinalPass()
class GEVulkanFinalStage
{
public:
    // ------------------------------------------------------------------------
    virtual ~GEVulkanFinalStage()                                            {}
    // ------------------------------------------------------------------------
    // True if the stage reads the depth of the FBO (binding 2 of io_set, in
    // shader read only layout), so it only runs for the FBO which keeps it
    // (GEVulkanFBOTexture::hasStoredDepth()), and only for the main one (not
    // the separate render targets, see GEVulkanFinalChain::initTarget). It has
    // to be the same as GEFinalPassInfo::m_needs_depth
    virtual bool needsDepth() const                           { return false; }
    // ------------------------------------------------------------------------
    // False if the stage has nothing to do this frame: it is skipped (apply()
    // isn't called and nothing is copied), and what the stage after it reads
    // is what the stage before it wrote. If no stage is active the FBO is
    // sampled directly, as if there was no chain. It's called once per frame,
    // for every target, in GEVulkanFinalChain::prepare(), which is before the
    // descriptors of the 2D renderer are updated (that's when the image the
    // users of the FBO sample has to be known), so it has to be cheap and can
    // only use what the game set before the draw calls were made (the camera
    // snapshots). p are the same draw calls as apply() gets
    virtual bool isActive(const std::vector<std::pair<GEVulkanDrawCall*,
                          GEVulkanCamera*> >& p) const         { return true; }
    // ------------------------------------------------------------------------
    // Outside of render pass. Reads binding 0 (and binding 2, the depth, if
    // needsDepth()) and writes binding 1 of io_set (set 0 of the pipeline
    // layout of the stage, which must be created with
    // GEVulkanFinalChain::getIOLayout()) and has to write all pixels of the
    // image, also the ones which are not in any viewport. The first one of
    // the chain is called with the layouts (shader read only for the sampled
    // image, general for the storage image) and the memory barriers for the
    // compute shaders of the chain already done, the chain does the barriers
    // between the stages and after the last one
    virtual void apply(VkCommandBuffer cmd, VkDescriptorSet io_set,
        const irr::core::dimension2d<irr::u32>& size,
        const std::vector<std::pair<GEVulkanDrawCall*,
        GEVulkanCamera*> >& p) = 0;
};   // GEVulkanFinalStage

// The order of the passes in the chain, lower runs first. The passes of the
// same order run in the order they were added. The depth of field is first, so
// the anti-aliasing (last) is applied on the edges of the blur too. The passes
// of the game are between them by default, they read what is displayed
// (tonemapped, display referred) and can't be anti-aliased by anything else
enum GEFinalPassOrder : int
{
    GFPO_DEPTH_OF_FIELD = 100,
    GFPO_DEFAULT = 500,
    GFPO_ANTI_ALIASING = 1000
};

// What the chain needs to know about a pass before the pass exists: the
// settings of the game decide if the scene is rendered to a FBO (instead of
// the swapchain image) and if the split deferred FBO (which keeps the depth)
// is used, and both are known before any stage is created
struct GEFinalPassInfo
{
    // For the log
    std::string m_name;

    // The handle addFinalPass() returned, which the copies the chain gets
    // (getEnabledFinalPasses) have, so GE can see which passes the chain was
    // made with. Ignored by addFinalPass()
    unsigned m_handle;

    // See GEFinalPassOrder
    int m_order;

    // True if the pass reads the depth of the FBO, which needs the split
    // deferred FBO (so the pbr shaders), the same as GEVulkanFinalStage::
    // needsDepth()
    bool m_needs_depth;

    // True (default) if the pass only runs for the main FBO of the game, not
    // for the separate render targets (the GUI shows them: the preview of a
    // kart for example)
    bool m_main_target_only;

    // Called when the chain is created, which is when the swapchain is
    // (re)created: false if the pass is not used. It decides if the scene is
    // rendered to a FBO and if the depth is kept, so the swapchain has to be
    // created again if what it returns changes: GEVulkanDriver::updateDriver()
    // does it (it compares the enabled passes with the ones the chain was made
    // with), so call it after the setting changed. NULL means always
    // enabled. Passes need compute in the
    // main queue, if the device can't they are never enabled
    std::function<bool()> m_enabled;

    // Creates the stage when the chain is created (with the size of the
    // biggest FBO), the chain owns it. It can return NULL (nothing is added)
    // or throw (it's logged, the other passes are created anyway)
    std::function<GEVulkanFinalStage*(GEVulkanDriver* vk,
        VkDescriptorSetLayout io_layout,
        const irr::core::dimension2d<irr::u32>& capacity)> m_create;
    // ------------------------------------------------------------------------
    GEFinalPassInfo()
    {
        m_handle = 0;
        m_order = GFPO_DEFAULT;
        m_needs_depth = false;
        m_main_target_only = true;
    }
};   // GEFinalPassInfo

// ----------------------------------------------------------------------------
// Passes written as one compute shader (the most of them)
// ----------------------------------------------------------------------------

// Values of the game for the passes, one set for every camera: it's set on the
// camera node before drawAll() (setFinalPassUserData) and copied in the
// snapshot which drawAll() takes, so the pass sees what the game had when it
// called drawAll() (not what it has when the frame is rendered), also with
// splitscreen and with more than one render target. What the values mean is
// decided by the game and its passes
struct GECameraUserData
{
    float m_values[8];
    // ------------------------------------------------------------------------
    GECameraUserData()
    {
        for (float& v : m_values)
            v = 0.0f;
    }
};   // GECameraUserData

// camera has to be a camera of the Vulkan scene manager (the same one which
// drawAll() uses), the values stay until they are set again
void setFinalPassUserData(irr::scene::ICameraSceneNode* camera,
                          const GECameraUserData& data);
// Forget the previous frame of the camera (GEFinalPassView::m_prev_pvm is the
// current one in the next frame), call it when the camera jumps: a teleport,
// the start of a race...
void resetFinalPassHistory(irr::scene::ICameraSceneNode* camera);

// One viewport (camera) of the target, what the pass sees of it
struct GEFinalPassView
{
    // x0, y0, x1, y1 (exclusive) in pixels of the target. A pass never reads
    // or writes outside of it (splitscreen)
    int32_t m_rect[4];

    // Height of the viewport in pixels of the target, and the same relative to
    // the reference height (1080) which the blur widths are scaled with, so
    // they are the same part of the screen at any resolution, render scale and
    // splitscreen
    float m_height;
    float m_scale;

    // Index of the viewport in the target, in the order of drawAll()
    unsigned m_index;

    // The values of setFinalPassUserData() when drawAll() was called
    GECameraUserData m_user_data;

    // Projection * view of the camera (the same as m_projection_view_matrix of
    // utils/camera.glsl: clip space of the target, reverse Z) and the one of
    // the previous frame of the same camera (the current one if there is no
    // previous frame, see resetFinalPassHistory). A frame is a beginScene(),
    // so more than one drawAll() in a frame (separate render targets) see the
    // same previous matrix
    irr::core::matrix4 m_pvm;
    irr::core::matrix4 m_prev_pvm;
    // ------------------------------------------------------------------------
    GEFinalPassView()
    {
        for (int32_t& v : m_rect)
            v = 0;
        m_height = 0.0f;
        m_scale = 0.0f;
        m_index = 0;
    }
};   // GEFinalPassView

// The push constants of the shader are GEFinalPassHeader (the first 32 bytes,
// see utils/final_pass.glsl) and then the m_push_size bytes the pass fills
static const unsigned GE_FINAL_PASS_MAX_PUSH_SIZE = 96;

// What GE does for a pass which is one compute shader: for every viewport
// (camera) of the target, it binds the camera of the viewport (if wanted) and
// dispatches the shader (local size 8x8, see utils/final_pass.glsl) over the
// rect of the viewport, the pixels which are not in any viewport are copied
// from the input. The shader reads binding 0 (and the depth, binding 2, if
// GEFinalPassInfo::m_needs_depth) and writes binding 1 of set 0, the format of
// the images is rgba8 (display referred, after the tonemap)
struct GEComputePassDesc
{
    // File of the compute shader in the folder of the shaders of GE
    std::string m_shader;

    // Bytes of the push constants after the header, at most
    // GE_FINAL_PASS_MAX_PUSH_SIZE
    unsigned m_push_size;

    // True if the shader uses the camera of the viewport (set 1, see
    // utils/camera.glsl), for example to get the position of the depth
    bool m_camera_set;

    // False for the viewports the pass has nothing to do for, the pass
    // doesn't run at all (and costs nothing) if there are none. NULL means
    // always. It's called for every viewport when the frame is prepared
    // (before the rect is known exactly: don't use the rect) and when the
    // pass runs
    std::function<bool(const GEFinalPassView& view)> m_is_active;

    // Writes the m_push_size bytes of the push constants of the viewport
    std::function<void(const GEFinalPassView& view, void* out)> m_fill;
    // ------------------------------------------------------------------------
    GEComputePassDesc()
    {
        m_push_size = 0;
        m_camera_set = false;
    }
};   // GEComputePassDesc

// Same as addFinalPass(), the stage is made by GE from desc (info.m_create is
// ignored)
unsigned addComputeFinalPass(const GEFinalPassInfo& info,
                             const GEComputePassDesc& desc);

// ----------------------------------------------------------------------------
// Adds a pass to the final chain, returns the handle to remove it. It has to
// be called before the swapchain is created (or the game has to create it
// again, and call refreshDeferredSplit() if m_needs_depth is true) for it to
// be used. It doesn't create anything by itself
unsigned addFinalPass(const GEFinalPassInfo& info);
// Removes a pass added with addFinalPass(), it's not used anymore when the
// chain is created the next time
void removeFinalPass(unsigned handle);

}

#endif
