uint packSurfelGIRGB(vec3 light)
{
    light = clamp(light, 0.0, 65408.0);
    float peak = max(light.x, max(light.y, light.z));
    int exponent = clamp(int(floor(log2(max(peak, exp2(-16.0))))) + 16, 0, 31);
    if (round(peak * exp2(float(24 - exponent))) >= 512.0)
        exponent = min(exponent + 1, 31);
    uvec3 mantissa = uvec3(min(round(light * exp2(float(24 - exponent))), 511.0));
    return mantissa.r | (mantissa.g << 9u) | (mantissa.b << 18u) | (uint(exponent) << 27u);
}

vec3 unpackSurfelGIRGB(uint bits)
{
    return vec3(uvec3(bits, bits >> 9u, bits >> 18u) & 511u) * exp2(float(bits >> 27u) - 24.0);
}
