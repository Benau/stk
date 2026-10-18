// A2B10G10R10 snorm, x is the lowest bits, w is the 2 bits at the top
vec4 unpackSnorm1010102(uint p)
{
    return max(vec4(float(int(p << 22u) >> 22), float(int(p << 12u) >> 22),
        float(int(p << 2u) >> 22), float(int(p) >> 30)) /
        vec4(511.0, 511.0, 511.0, 1.0), vec4(-1.0));
}

// w_bits are the 2 bits at the top, kept as they are
uint packSnorm1010102(vec3 v, uint w_bits)
{
    ivec3 i = ivec3(round(clamp(v, vec3(-1.0), vec3(1.0)) * 511.0));
    return (uint(i.x) & 0x3FFu) | ((uint(i.y) & 0x3FFu) << 10u) |
        ((uint(i.z) & 0x3FFu) << 20u) | w_bits;
}
