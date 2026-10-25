// CMAA2 (Conservative Morphological Anti-Aliasing 2.3) of GEVulkanAntiAliasing,
// a port of the compute shaders of https://github.com/GameTechDev/CMAA2
//
// Copyright (c) 2018, Intel Corporation, Author: Filip Strugar
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy
// of the License at http://www.apache.org/licenses/LICENSE-2.0
//
// Differences from the reference (single sample, luma edge detection, HIGH
// preset, no extra sharpness):
// - The input is the display referred (gamma encoded) color of the FBO, so
//   luma is computed without the sqrt which approximates gamma of linear
//   input, and the output is stored as it is.
// - The input and the output are different images (the output has been filled
//   with the input by cmaa2_copy.comp), only the pixels of the edges are
//   written to it.
// - Everything is restricted to the rect of one viewport (splitscreen), the
//   colors are clamped inside of it, there are no edges outside of it and no
//   pixel outside of it is ever written, so nothing is blended over the
//   border between viewports.
// - The working data is in storage buffers only (no storage images and no
//   image atomics), and the sizes of the lists are checked.
//
// The including shader defines one of CMAA2_COPY, CMAA2_EDGES, CMAA2_ARGS,
// CMAA2_CANDIDATES and CMAA2_APPLY

layout(set = 0, binding = 0) uniform sampler2D u_src;
// Only written, never loaded with imageLoad (see blur.glsl)
layout(set = 0, binding = 1, rgba8) uniform writeonly image2D u_dst;

layout(set = 1, binding = 0, std430) buffer EdgesBuffer
{
    // Two pixels per uint, 4 bits of edges for each: bit 0 = right, 1 = below,
    // 2 = left, 3 = above, in rows of m_buf.x uints
    uint g_edges[];
};
layout(set = 1, binding = 1, std430) buffer HeadsBuffer
{
    // Head of the list of deferred colors of every 2x2 quad, in rows of
    // m_buf.y uints, 0xFFFFFFFF if none
    uint g_heads[];
};
layout(set = 1, binding = 2, std430) buffer CandidatesBuffer
{
    uint g_candidates[];
};
layout(set = 1, binding = 3, std430) buffer LocationsBuffer
{
    // Quads which have items, (x << 16) | y
    uint g_locations[];
};
layout(set = 1, binding = 4, std430) buffer ItemsBuffer
{
    // Two uints per item: the next item (with the header) and the color
    uint g_items[];
};
layout(set = 1, binding = 5, std430) buffer ControlBuffer
{
    // 3 = number of items to process (candidates or locations), 4 = counter
    // of candidates, 8 = counter of locations, 12 = counter of items. The
    // counters are zero again after the last step
    uint g_control[16];
};
layout(set = 1, binding = 6, std430) buffer IndirectBuffer
{
    uint g_indirect[4];
};

layout(push_constant) uniform PushConstants
{
    // x0, y0, x1, y1 (exclusive) of the viewport in pixels of the image
    ivec4 m_rect;
    // x = uints per row of g_edges, y = uints per row of g_heads, z = size of
    // g_candidates, w = size of g_locations
    uvec4 m_buf;
    // Number of items of g_items
    uint m_max_items;
} pc;

const float EDGE_THRESHOLD = 0.07;
const float LOCAL_CONTRAST_ADAPTATION = 0.10;
const float SIMPLE_SHAPE_BLURINESS = 0.10;
const float MAX_LINE_LENGTH = 86.0;
const float SYMMETRY_CORRECTION_OFFSET = 0.22;
const float DAMPENING_EFFECT = 0.15;

const uint PROCESS_CANDIDATES_THREADS = 128u;
const uint DEFERRED_APPLY_THREADS = 32u;
const uint BLEND_ITEM_SLM_SIZE = 768u;

bool inRect(ivec2 p)
{
    return all(greaterThanEqual(p, pc.m_rect.xy)) &&
        all(lessThan(p, pc.m_rect.zw));
}

// Clamped inside of the rect of the viewport
vec3 loadColor(ivec2 p)
{
    return texelFetch(u_src, clamp(p, pc.m_rect.xy, pc.m_rect.zw - 1),
        0).rgb;
}

float lumaForEdges(vec3 c)
{
    return dot(c, vec3(0.299, 0.587, 0.114));
}

uint packEdges(vec4 edges)
{
    return uint(dot(edges, vec4(1.0, 2.0, 4.0, 8.0)));
}

vec4 unpackEdges(uint v)
{
    return vec4(float((v & 1u) != 0u), float((v & 2u) != 0u),
        float((v & 4u) != 0u), float((v & 8u) != 0u));
}

// No edges outside of the rect of the viewport, so the lines stop at its border
uint loadEdge(ivec2 p)
{
    if (!inRect(p))
        return 0u;
    uint v = g_edges[uint(p.y) * pc.m_buf.x + uint(p.x >> 1)];
    return (v >> ((uint(p.x) & 1u) * 4u)) & 0xFu;
}

// The colors are display referred values in [0, 1], 11 + 11 + 10 bits
uint packColor(vec3 c)
{
    uvec3 u = uvec3(clamp(c, vec3(0.0), vec3(1.0)) *
        vec3(2047.0, 2047.0, 1023.0) + 0.5);
    return u.x | (u.y << 11) | (u.z << 22);
}

vec3 unpackColor(uint p)
{
    return vec3(float(p & 2047u) / 2047.0, float((p >> 11) & 2047u) / 2047.0,
        float((p >> 22) & 1023u) / 1023.0);
}

#if defined(CMAA2_CANDIDATES)

void storeColorSample(ivec2 pixel, vec3 color, bool is_complex_shape)
{
    // Never write a pixel of another viewport
    if (!inRect(pixel))
        return;
    uint counter_index = atomicAdd(g_control[12], 1u);
    if (counter_index >= pc.m_max_items)
        return;
    ivec2 quad = pixel >> 1;
    uint offset_xy = uint((pixel.y & 1) * 2 + (pixel.x & 1));
    uint header = (offset_xy << 30) | (is_complex_shape ? (1u << 26) : 0u);
    uint original_index = atomicExchange(
        g_heads[uint(quad.y) * pc.m_buf.y + uint(quad.x)],
        counter_index | header);
    g_items[counter_index * 2u] = original_index;
    g_items[counter_index * 2u + 1u] = packColor(color);
    if (original_index == 0xFFFFFFFFu)
    {
        uint location_counter = atomicAdd(g_control[8], 1u);
        if (location_counter < pc.m_buf.w)
        {
            g_locations[location_counter] =
                (uint(quad.x) << 16) | uint(quad.y);
        }
    }
}

shared uint g_blend_item_count;
shared uvec2 g_blend_items[BLEND_ITEM_SLM_SIZE];

vec4 computeSimpleShapeBlendValues(vec4 edges, vec4 edges_left,
    vec4 edges_right, vec4 edges_top, vec4 edges_bottom,
    bool dont_test_shape_validity)
{
    float from_right = edges.r;
    float from_below = edges.g;
    float from_left = edges.b;
    float from_above = edges.a;

    float blur_coeff = SIMPLE_SHAPE_BLURINESS;

    float number_of_edges = dot(edges, vec4(1.0));
    float number_of_edges_all_around = dot(edges_left.bga +
        edges_right.rga + edges_top.rba + edges_bottom.rgb, vec3(1.0));

    if (!dont_test_shape_validity)
    {
        if (number_of_edges == 1.0)
            blur_coeff = 0.0;
        if (number_of_edges == 2.0)
        {
            blur_coeff *= (1.0 - from_below * from_above) *
                (1.0 - from_right * from_left);
        }
    }

    if (number_of_edges == 2.0)
    {
        blur_coeff *= 0.75;
        float k = 0.9;
        from_right += k * (edges.g * edges_top.r * (1.0 - edges_left.g) +
            edges.a * edges_bottom.r * (1.0 - edges_left.a));
        from_below += k * (edges.b * edges_right.g * (1.0 - edges_top.b) +
            edges.r * edges_left.g * (1.0 - edges_top.r));
        from_left += k * (edges.a * edges_bottom.b * (1.0 - edges_right.a) +
            edges.g * edges_top.b * (1.0 - edges_right.g));
        from_above += k * (edges.r * edges_left.a * (1.0 - edges_bottom.r) +
            edges.b * edges_right.a * (1.0 - edges_bottom.b));
    }

    blur_coeff *= clamp(1.30 - number_of_edges_all_around / 10.0, 0.0, 1.0);

    return vec4(from_left, from_above, from_right, from_below) * blur_coeff;
}

void findZLineLengths(out float line_length_left, out float line_length_right,
    ivec2 screen_pos, bool horizontal, bool inverted_z_shape,
    ivec2 step_right)
{
    uint mask_trace_left, mask_trace_right;
    if (horizontal)
    {
        mask_trace_left = 0x08u;   // tracing top edge
        mask_trace_right = 0x02u;  // tracing bottom edge
    }
    else
    {
        mask_trace_left = 0x04u;   // tracing left edge
        mask_trace_right = 0x01u;  // tracing right edge
    }
    if (inverted_z_shape)
    {
        uint temp = mask_trace_left;
        mask_trace_left = mask_trace_right;
        mask_trace_right = temp;
    }
    const uint mask_left = mask_trace_left;
    const uint bits_continue_left = mask_trace_left;
    const uint mask_right = mask_trace_right;
    const uint bits_continue_right = mask_trace_right;

    bool continue_left = true;
    bool continue_right = true;
    line_length_left = 1.0;
    line_length_right = 1.0;
    for (;;)
    {
        uint edge_left = loadEdge(screen_pos -
            step_right * int(line_length_left));
        uint edge_right = loadEdge(screen_pos +
            step_right * (int(line_length_right) + 1));

        continue_left = continue_left &&
            ((edge_left & mask_left) == bits_continue_left);
        continue_right = continue_right &&
            ((edge_right & mask_right) == bits_continue_right);
        line_length_left += continue_left ? 1.0 : 0.0;
        line_length_right += continue_right ? 1.0 : 0.0;

        float max_lr = max(line_length_right, line_length_left);

        // Both stopped: finish
        if (!continue_left && !continue_right)
            max_lr = MAX_LINE_LENGTH;

        // Stop when one side is much longer than the other (or at the limit)
        if (max_lr >= min(MAX_LINE_LENGTH, 1.25 *
            min(line_length_right, line_length_left) - 0.25))
            break;
    }
}

bool collectBlendZs(ivec2 screen_pos, bool horizontal, bool inverted_z_shape,
    float shape_quality_score, float line_length_left,
    float line_length_right)
{
    float left_odd = SYMMETRY_CORRECTION_OFFSET * mod(line_length_left, 2.0);
    float right_odd = SYMMETRY_CORRECTION_OFFSET * mod(line_length_right, 2.0);

    float dampen_effect = clamp((line_length_left + line_length_right -
        shape_quality_score) * DAMPENING_EFFECT, 0.0, 1.0);

    float loop_from = -floor((line_length_left + 1.0) / 2.0) + 1.0;
    float loop_to = floor((line_length_right + 1.0) / 2.0);

    const uint blend_item_count = uint(loop_to - loop_from + 1.0);
    uint item_index = atomicAdd(g_blend_item_count, blend_item_count);
    if ((item_index + blend_item_count) > BLEND_ITEM_SLM_SIZE)
        return false;

    float total_length = (loop_to - loop_from) + 1.0 - left_odd - right_odd;
    float lerp_step = 1.0 / total_length;
    float lerp_from_k = (0.5 - left_odd - loop_from) * lerp_step;

    uint item_header = (uint(screen_pos.x) << 18) | uint(screen_pos.y);
    uint item_val_static = (horizontal ? (1u << 31) : 0u) |
        (inverted_z_shape ? (1u << 30) : 0u);

    for (float i = loop_from; i <= loop_to; i += 1.0)
    {
        float second_part = (i > 0.0) ? 1.0 : 0.0;
        float src_offset = 1.0 - second_part * 2.0;

        float lerp_k = (lerp_step * i + lerp_from_k) * src_offset +
            second_part;
        lerp_k *= dampen_effect;

        uvec2 encoded_item;
        encoded_item.x = item_header;
        encoded_item.y = item_val_static | (uint(i + 256.0) << 20) |
            (uint(src_offset + 256.0) << 10) |
            uint(clamp(lerp_k, 0.0, 1.0) * 1023.0 + 0.5);
        g_blend_items[item_index++] = encoded_item;
    }
    return true;
}

// Fallback when the shared memory is full: the same, stored directly
void blendZs(ivec2 screen_pos, bool horizontal, bool inverted_z_shape,
    float shape_quality_score, float line_length_left, float line_length_right,
    ivec2 step_right)
{
    ivec2 blend_dir = horizontal ? ivec2(0, -1) : ivec2(-1, 0);
    if (inverted_z_shape)
        blend_dir = -blend_dir;

    float left_odd = SYMMETRY_CORRECTION_OFFSET * mod(line_length_left, 2.0);
    float right_odd = SYMMETRY_CORRECTION_OFFSET * mod(line_length_right, 2.0);

    float dampen_effect = clamp((line_length_left + line_length_right -
        shape_quality_score) * DAMPENING_EFFECT, 0.0, 1.0);

    float loop_from = -floor((line_length_left + 1.0) / 2.0) + 1.0;
    float loop_to = floor((line_length_right + 1.0) / 2.0);

    float total_length = (loop_to - loop_from) + 1.0 - left_odd - right_odd;
    float lerp_step = 1.0 / total_length;
    float lerp_from_k = (0.5 - left_odd - loop_from) * lerp_step;

    for (float i = loop_from; i <= loop_to; i += 1.0)
    {
        float second_part = (i > 0.0) ? 1.0 : 0.0;
        float src_offset = 1.0 - second_part * 2.0;

        float lerp_k = (lerp_step * i + lerp_from_k) * src_offset +
            second_part;
        lerp_k *= dampen_effect;

        ivec2 pixel = screen_pos + step_right * int(i);
        vec3 color_center = loadColor(pixel);
        vec3 color_from = loadColor(pixel + blend_dir * int(src_offset));
        storeColorSample(pixel, mix(color_center, color_from, lerp_k), true);
    }
}

void detectZsHorizontal(vec4 edges, vec4 edges_m1_p0, vec4 edges_p1_p0,
    vec4 edges_p2_p0, out float inverted_z_score, out float normal_z_score)
{
    inverted_z_score = edges.r * edges.g * edges_p1_p0.a;
    inverted_z_score *= 2.0 + (edges_m1_p0.g + edges_p2_p0.a) -
        (edges.a + edges_p1_p0.g) -
        0.7 * (edges_p2_p0.g + edges_m1_p0.a + edges.b + edges_p1_p0.r);

    normal_z_score = edges.r * edges.a * edges_p1_p0.g;
    normal_z_score *= 2.0 + (edges_m1_p0.a + edges_p2_p0.g) -
        (edges.g + edges_p1_p0.a) -
        0.7 * (edges_p2_p0.a + edges_m1_p0.g + edges.b + edges_p1_p0.r);
}

#endif   // CMAA2_CANDIDATES
