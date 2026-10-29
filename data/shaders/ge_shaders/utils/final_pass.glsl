// Shared by the compute shaders of the passes of the final chain which are
// added with addComputeFinalPass() (see GEComputePassDesc). Include it before
// anything else, and define FINAL_PASS_DEPTH before it if the pass reads the
// depth of the FBO (GEFinalPassInfo::m_needs_depth). The local size is the one
// GEVulkanComputeFinalPass dispatches with
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

// The image of the previous pass (or the FBO), display referred
layout(set = 0, binding = 0) uniform sampler2D u_src;
// Only written, never loaded with imageLoad (see blur.glsl)
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D u_dst;
#ifdef FINAL_PASS_DEPTH
// Reverse Z depth of the FBO, use texelFetch (nearest) with the pixel
layout(set = 0, binding = 2) uniform sampler2D u_depth;
#endif

// The first 32 bytes of the push constants, the pass puts its own after it:
//
// layout(push_constant) uniform PushConstants
// {
//     FinalPassHeader m_header;
//     float m_boost;
// } pc;
struct FinalPassHeader
{
    // x0, y0, x1, y1 (exclusive) in pixels of the image: the viewport of one
    // camera. Nothing is read or written outside of it
    ivec4 m_rect;
    // x: height of the viewport in pixels, y: the same relative to the
    // reference height (1080), z: index of the viewport, w: 0
    vec4 m_params;
};

// The pixel of this invocation, false if it's outside of the viewport
bool finalPassGetPixel(const FinalPassHeader header, out ivec2 pos)
{
    pos = ivec2(gl_GlobalInvocationID.xy) + header.m_rect.xy;
    return all(lessThan(pos, header.m_rect.zw));
}
