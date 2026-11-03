// Random numbers of the passes which need a dither or a low discrepancy
// sample: the steps of the marches of the volumetric light (light_scatter.comp
// and god_rays.comp), the samples of the specular reflections
// (utils/specular_gi.glsl) and the rotation of the filter of their spatial
// resolve (solid_reflection_resolve.frag).

// The hash of an integer to another one (lowbias32, Wellons)
uint hashUint(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

// A random angle of [0, 2π) of a pixel of the screen, decorrelated from the
// pixels next to it: the hash of its coordinates
float randomAngle(vec2 pixel)
{
    ivec2 coordinate = ivec2(pixel);
    return float(hashUint(uint(coordinate.x) | (uint(coordinate.y) << 16))) *
        (6.28318530718 / 4294967296.0);
}

// Interleaved gradient noise of Jimenez (Call of Duty: Advanced Warfare, "Next
// Generation Post Processing"), a dither in [0, 1) which is decorrelated from
// the neighboring pixels
float interleavedGradientNoise(vec2 pixel)
{
    return fract(52.9829189 * fract(dot(pixel, vec2(0.06711056, 0.00583715))));
}

// The index-th point of the R2 sequence (Roberts 2018, "The Unreasonable
// Effectiveness of Quasirandom Sequences") in [0, 1)²: a low discrepancy
// sequence, a few points already cover the square evenly
vec2 r2Sequence(uint index)
{
    const vec2 r2 = vec2(0.7548776662466927, 0.5698402909980532);
    return fract(r2 * (float(index) + 1.0));
}
