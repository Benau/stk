// The reflection of the glossy solid materials (GEVulkanSolidReflection),
// shared by its passes: solid_reflection_classify.comp writes 0 to every
// pixel of the viewport and adds the glossy ones to the list, then
// solid_reflection_rt.comp (traced in the ray traced scene) or
// solid_reflection_ssr.comp (traced over the depth or the HiZ depth, the
// color is the output of the previous frame) trace only the pixels of the
// list, which is their indirect dispatch (they write every pixel of it). A
// fullscreen pass would take as long as its slowest pixels in every group,
// most of which are not glossy. The lighting pass only reads the output of the
// glossy pixels, so only they are written.
//
// The output is the specular radiance of the pixel which the lighting pass
// uses instead of the specular IBL (handlePBRDeferred): the traced color mixed
// with the specular IBL by the weight of the reflection, the specular IBL if
// nothing is traced (a miss). The lighting pass applies the Fresnel of it, so
// it's never negative and has no alpha (R11G11B10F). A stochastic trace of
// rough reflections, or a mip chain of the output sampled by the roughness,
// would write the same.
//
// The shader which includes it defines SOLID_REFLECTION_SET if it's not 0, and
// includes utils/constants_utils.glsl before it

#ifndef SOLID_REFLECTION_SET
#define SOLID_REFLECTION_SET 0
#endif

#ifdef SHADER_STORAGE_IMAGE_EXTENDED_FORMATS
layout(set = SOLID_REFLECTION_SET, binding = 0, r11f_g11f_b10f) uniform
    writeonly image2D u_reflection;
#else
layout(set = SOLID_REFLECTION_SET, binding = 0, rgba16f) uniform writeonly
    image2D u_reflection;
#endif
// The glossy pixels (packSolidReflectionPixel), m_dispatch is the indirect
// dispatch of the trace (groups of 64 of them)
layout(std430, set = SOLID_REFLECTION_SET, binding = 1) buffer
    SolidReflectionList
{
    uint m_count;
    uint m_dispatch[3];
    uint m_pixels[];
} u_reflection_list;

// ----------------------------------------------------------------------------
uint packSolidReflectionPixel(ivec2 pixel)
{
    return uint(pixel.x) | (uint(pixel.y) << 16u);
}   // packSolidReflectionPixel

// ----------------------------------------------------------------------------
ivec2 unpackSolidReflectionPixel(uint packed)
{
    return ivec2(packed & 0xFFFFu, packed >> 16u);
}   // unpackSolidReflectionPixel

// ----------------------------------------------------------------------------
// The pixel of the thread of a trace, false if it has none (the last group
// isn't full)
bool getSolidReflectionPixel(out ivec2 pixel)
{
    pixel = ivec2(0);
    if (gl_GlobalInvocationID.x >= u_reflection_list.m_count)
        return false;
    pixel = unpackSolidReflectionPixel(
        u_reflection_list.m_pixels[gl_GlobalInvocationID.x]);
    return true;
}   // getSolidReflectionPixel
