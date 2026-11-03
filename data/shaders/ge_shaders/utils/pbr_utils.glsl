vec2 F_AB(float perceptual_roughness, float NdotV)
{
    vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = perceptual_roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * NdotV)) * r.x + r.y;
    // Keep F_ab positive to avoid divide-by-zero in downstream BRDF terms.
    return max(vec2(-1.04, 1.04) * a004 + r.zw, vec2(0.00005));
}

// Lambert model
float F_Schlick(float f0, float f90, float VdotH)
{
    return mix(f0, f90, pow(1.0 - VdotH, 5.0));
}

// Burley diffuse of bevy, roughness is alpha (perceptualRoughnessToRoughness)
float Fd_Burley(float roughness, float NdotV, float NdotL, float LdotH)
{
    float f90 = 0.5 + 2.0 * roughness * LdotH * LdotH;
    float lightScatter = F_Schlick(1.0, f90, NdotL);
    float viewScatter = F_Schlick(1.0, f90, NdotV);
    return lightScatter * viewScatter * (1.0 / 3.14159265359);
}

// Calculate distribution.
// Based on https://google.github.io/filament/Filament.html#citation-walter07
// D_GGX(h,α) = α^2 / { π ((n⋅h)^2 (α2−1) + 1)^2 }
// Simple implementation, has precision problems when using fp16 instead of fp32
// see https://google.github.io/filament/Filament.html#listing_speculardfp16
float D_GGX(float roughness, float NdotH)
{
    float oneMinusNdotHSquared = 1.0 - NdotH * NdotH;
    float a = NdotH * roughness;
    float k = roughness / (oneMinusNdotHSquared + a * a);
    return k * k * (1.0 / 3.14159265359);
}

// Calculate visibility. Exact Smith-GGX correlated of bevy, roughness is alpha
// (perceptualRoughnessToRoughness)
// https://google.github.io/filament/Filament.md.html#listing_approximatedspecularv
float V_Smith_GGX_Correlated(float roughness, float NdotV, float NdotL)
{
    float a2 = roughness * roughness;
    float lambdaV = NdotL * sqrt((NdotV - a2 * NdotV) * NdotV + a2);
    float lambdaL = NdotV * sqrt((NdotL - a2 * NdotL) * NdotL + a2);
    return 0.5 / (lambdaV + lambdaL);
}

// Fresnel function
// see https://google.github.io/filament/Filament.html#citation-schlick94
// F_Schlick(v,h,f_0,f_90) = f_0 + (f_90 − f_0) (1 − v⋅h)^5
vec3 fresnel(vec3 f0, float f90, float VdotH)
{
    return f0 + (f90 - f0) * pow(1.0 - VdotH, 5.0);
}

vec3 envBRDFApprox(vec3 F0, vec2 F_ab)
{
    return F0 * F_ab.x + F_ab.y;
}

// Returns alpha, the roughness squared
float perceptualRoughnessToRoughness(float perceptual_roughness)
{
    float roughness = clamp(perceptual_roughness, 0.089, 1.0);
    return roughness * roughness;
}

// Bevy's compute_multiscatter
// Approximation: https://www.jcgt.org/published/0008/01/03/paper.pdf
void computeMultiscatter(vec3 F0, vec2 F_ab, float Ems,
                         float specular_occlusion, out vec3 FssEss,
                         out vec3 FmsEms, out vec3 Edss)
{
    FssEss = (F0 * F_ab.x + F_ab.y) * specular_occlusion;
    vec3 Favg = F0 + (1.0 - F0) / 21.0;
    FmsEms = FssEss * Favg / (1.0 - Ems * Favg) * Ems;
    Edss = 1.0 - (FssEss + FmsEms);
}

// Indirect light of the environment, same as bevy's environment_map_light:
// the dielectric and the metallic multiscatter are calculated separately and
// mixed, because the compensation is not linear in F0.
vec3 environmentLight(
    vec3 irradiance,
    vec3 radiance,
    vec3 diffuse_color,
    vec3 F0_dielectric,
    vec3 F0_metallic,
    float metallic,
    vec2 F_ab,
    vec3 ao_diffuse,
    vec3 ao_specular)
{
    // No real world material has specular values under 0.02, so we use this
    // range as a "pre-baked specular occlusion" that extinguishes the fresnel
    // term, for artistic control.
    // See: https://google.github.io/filament/Filament.md.html#specularocclusion
    vec3 F0_surface = mix(F0_dielectric, F0_metallic, metallic);
    float specular_occlusion = clamp(dot(F0_surface, vec3(50.0 * 0.33)),
        0.0, 1.0);

    float Ems = 1.0 - (F_ab.x + F_ab.y);
    vec3 dielectric_FssEss, dielectric_FmsEms, dielectric_Edss;
    vec3 metallic_FssEss, metallic_FmsEms, metallic_Edss;
    computeMultiscatter(F0_dielectric, F_ab, Ems, specular_occlusion,
        dielectric_FssEss, dielectric_FmsEms, dielectric_Edss);
    computeMultiscatter(F0_metallic, F_ab, Ems, specular_occlusion,
        metallic_FssEss, metallic_FmsEms, metallic_Edss);

    vec3 FssEss = mix(dielectric_FssEss, metallic_FssEss, metallic);
    vec3 FmsEms = mix(dielectric_FmsEms, metallic_FmsEms, metallic);
    vec3 kD = diffuse_color * dielectric_Edss;

    vec3 diffuse = (FmsEms + kD) * irradiance * ao_diffuse;
    vec3 specular = FssEss * radiance * ao_specular;
    return diffuse + specular;
}

// "Moving Frostbite to Physically Based Rendering 3.0", listing 22
// https://seblagarde.wordpress.com/wp-content/uploads/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf#page=70
vec3 radianceSampleDirection(vec3 N, vec3 R, float roughness)
{
    float smoothness = clamp(1.0 - roughness, 0.0, 1.0);
    float lerp_factor = smoothness * (sqrt(smoothness) + roughness);
    return mix(N, R, lerp_factor);
}

// ---------------------------------------------------------------------------
// Sampling of the specular lobe of a reflection, shared by the screen space
// reflection (solid_ssr.frag) and the ray traced one (solid_rt.frag): a half
// vector of the visible normal distribution (Heitz 2018, bounded VNDF) is
// sampled and the reflected ray is traced along it. The radiance of the hit is
// stored raw, like the trace of FidelityFX SSSR: the sample has no BRDF weight
// (the Fresnel of every sample would be noise in the reflection), the
// split-sum BRDF of the surface is applied once by the lighting calculation,
// like the composite of FidelityFX SSSR.
//
// The basis is orthonormalize of Duff et al. 2017, "Building an Orthonormal
// Basis, Revisited"
// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
// ---------------------------------------------------------------------------
mat3 orthonormalBasis(vec3 n)
{
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    return mat3(vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3(b, s + n.y * n.y * a, -n.y), n);
}

// Bounded VNDF sampling of the visible normal distribution, xi is a uniform
// random point in [0, 1)², roughness is alpha (perceptualRoughnessToRoughness)
vec3 sampleVisibleGGX(vec2 xi, float roughness, vec3 n, vec3 view)
{
    float alpha = roughness;
    float wi_n = dot(view, n);
    vec3 wi_z = -n * wi_n;
    vec3 wi_xy = view + wi_z;
    vec3 wi_std = -normalize(alpha * wi_xy + wi_z);
    float wi_std_z = dot(wi_std, n);

    // The bound of the spherical cap of the sample (eq. 5 and 6 of the paper)
    float s = 1.0 + length(wi_xy);
    float a = clamp(alpha, 0.0, 1.0);
    float a2 = a * a;
    float s2 = s * s;
    float k = (1.0 - a2) * s2 / (s2 + a2 * wi_n * wi_n);
    float b = wi_n > 0.0 ? k * wi_std_z : wi_std_z;

    float z = 1.0 - xi.y * (1.0 + b);
    float sin_theta = sqrt(max(0.0, 1.0 - z * z));
    float phi = 2.0 * 3.14159265359 * xi.x - 3.14159265359;
    vec3 c_std = vec3(sin_theta * cos(phi), sin_theta * sin(phi), z);
    vec3 c = orthonormalBasis(n) * c_std;

    vec3 wm_std = c + wi_std;
    vec3 wm_std_z = n * dot(n, wm_std);
    vec3 wm_std_xy = wm_std_z - wm_std;
    vec3 h = normalize(alpha * wm_std_xy + wm_std_z);
    return reflect(-view, h);
}

// The world space direction of a sample of the specular lobe: view is the
// direction from the surface to the eye (world space), xi a uniform point of
// [0, 1)²
vec3 sampleSpecularLobe(vec3 view, float perceptual_roughness, vec2 xi,
                        vec3 n)
{
    return sampleVisibleGGX(xi, perceptualRoughnessToRoughness(
        perceptual_roughness), n, view);
}

// The F0 of a surface, the same as the direct lighting (0.04 of the
// dielectrics, the albedo of the metals)
vec3 getReflectionF0(vec3 diffuse_color, float metallic)
{
    return mix(vec3(0.04), diffuse_color, metallic);
}
