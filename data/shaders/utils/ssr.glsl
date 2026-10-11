vec3 CalcCoordFromPosition(vec3 pos, mat4 projection_matrix,
                           vec2 viewport_scale, vec2 viewport_offset)
{
    vec4 projectedCoord = projection_matrix * vec4(pos, 1.0);
    projectedCoord.xyz /= projectedCoord.w;
#if defined(VULKAN)
    projectedCoord.xy   = projectedCoord.xy * 0.5 + 0.5;  // map X,Y from -1 -> +1 into 0 -> 1
    // no Z remap here because vulkan projection matrix already gave us Z in [0..1]
#else
    projectedCoord.xyz  = projectedCoord.xyz * 0.5 + 0.5;
#endif
    // scale and offset by viewport
    projectedCoord.xy   = projectedCoord.xy * viewport_scale + viewport_offset;
    return projectedCoord.xyz;
}

// Fade out edges of screen buffer tex
// 1 means full render tex, 0 means full IBL tex
float GetEdgeFade(vec2 coords, vec2 viewport_scale, vec2 viewport_offset)
{
    // transform coords to viewport space
    vec2 viewport_coords = (coords - viewport_offset) / viewport_scale;
    float gradL = smoothstep(0.0, 0.4, viewport_coords.x);
    float gradR = 1.0 - smoothstep(0.6, 1.0, viewport_coords.x);
    float gradT = smoothstep(0.0, 0.4, viewport_coords.y);
    float gradB = 1.0 - smoothstep(0.6, 1.0, viewport_coords.y);
    return min(min(gradL, gradR), min(gradT, gradB));
}

vec2 RayCast(vec3 dir, vec3 hitCoord, mat4 projection_matrix,
             vec2 viewport_scale, vec2 viewport_offset, sampler2DShadow depth)
{
    dir *= 0.5;
    hitCoord += dir;

    vec3 projectedCoord = CalcCoordFromPosition(hitCoord, projection_matrix,
                          viewport_scale, viewport_offset);
    float factor = 1.0;

    for (int i = 0; i < 32; i++)
    {
        float direction = texture(depth, projectedCoord);
        factor *= direction;
        dir = dir * (0.5 + 0.5 * factor);
        hitCoord += dir * (2. * direction - 1.);
        projectedCoord = CalcCoordFromPosition(hitCoord, projection_matrix,
                         viewport_scale, viewport_offset);
    }

    return projectedCoord.xy;
}


// Only calculate reflections if the reflectivity value is high enough (the
// gloss stored in the normal texture), and never towards the camera
bool SSRIsReflective(float specval, float cosine)
{
    return specval >= 0.5 && cosine <= 0.2;
}

// How much of the traced color replaces the fallback (the specular IBL), the
// product of the fades below (the same as chaining a mix() for each of them)
// edge   : GetEdgeFade() of the hit
// cosine : dot(reflected, eyedir), fades out the rays towards the camera
// mirror : gloss of the hit, disables raycasts onto another reflective surface
// specval: gloss of the surface, brought from 0.5-1.0 to 0.0-1.0
float SSRBlendWeight(float edge, float cosine, float mirror, float specval)
{
    return edge * (1. - max(cosine * 5., 0.)) * (4. - max(mirror * 4., 3.)) *
        ((specval - 0.5) * 2.0);
}

#ifdef SSR_HIZ
// The includer declares u_hiz_depth (sampler2D, reverse-Z max depth levels of
// GEVulkanHiZDepth, its level 0 is the viewport), u_camera and
// u_hiz_iterations (constants_utils.glsl) before including this file

// Start tracing in this level.
#define HIZ_START_LEVEL      0
// Stop tracing if current level is higher than this. (higher level means lower value)
#define HIZ_STOP_LEVEL       0
#define HIZ_MAX_LEVEL        6

// Set to 1 to disable HiZ and perform naive linear search.
#define DEBUG_LINEAR_SEARCH  0
// Thickness of a surface, in view space units (meters), a ray that is behind
// a surface by more than (ABS + REL * surface_view_depth) passes behind it
// instead of hitting it. The old test used raw reverse-Z depth, whose world
// size grows with the square of the distance (and 10x with near = 0.1)
#ifndef SSR_THICKNESS_ABS
#define SSR_THICKNESS_ABS    0.3
#endif
#ifndef SSR_THICKNESS_REL
#define SSR_THICKNESS_REL    0.01
#endif

// ---------------------------------------------------------------------------
// SSR debug visualisation, replaces o_displace_ssr with a debug color.
// 0 : off
// 1 : exit reason of the trace
//     green   = hit
//     red     = iteration limit (u_hiz_iterations) reached
//     blue    = ray reached the far plane (z axis limited the trace)
//     cyan    = ray reached the screen edge (x or y axis limited the trace)
//     yellow  = early exit in main (normal.z < -0.75, no trace done)
//     magenta = back facing (NdotV <= 0, no trace done)
//     white   = hit, but the hit point is outside of viewport (fallback used)
// 2 : iteration heat map (black = 0, red = u_hiz_iterations)
// 3 : deepest HiZ level reached (black = 0, white = HIZ_MAX_LEVEL)
// 4 : per-iteration cause counters (normalized by u_hiz_iterations)
//     R = level 0, ray behind surface by more than the thickness, step 1 px
//     G = level > 0, ray in front, skipped to the next cell (good, fast)
//     B = level > 0, ray behind the max depth of a coarse cell, descended
//     R and B both high = ray is crawling behind a foreground occluder
// 5 : hit uv (r = x, g = y), black if no hit
// 6 : depth discontinuity under the hit: linear depth min/max ratio in the
//     3x3 texels around the hit, black = flat, white = big jump (>= 2x)
// 7 : trace start depth vs depth buffer at the start pixel, ratio of the
//     linear depths. green = ray starts in front, red = starts behind
// ---------------------------------------------------------------------------
#ifndef SSR_DEBUG_MODE
#define SSR_DEBUG_MODE       0
#endif

#if SSR_DEBUG_MODE != 0
// Exit reason: 0 = hit, 1 = iteration limit, 2 = far plane, 3 = screen edge
int dbg_exit_reason = 0;
uint dbg_iterations = 0;
int dbg_deepest_level = 0;
uint dbg_cnt_behind_l0 = 0;
uint dbg_cnt_skip_hi = 0;
uint dbg_cnt_descend_hi = 0;
#endif

#if SSR_DEBUG_MODE != 0
float dbgLinearDepth(float d)
{
    vec4 v = u_camera.m_inverse_projection_matrix * vec4(0.0, 0.0, d, 1.0);
    return abs(v.z) / max(abs(v.w), 1e-12);
}

// hit_uv is in viewport space [0, 1], start_ss is the trace start (x, y, depth)
vec4 dbgColor(bool hit, bool hit_outside, vec2 hit_uv, vec3 start_ss)
{
#if SSR_DEBUG_MODE == 1
    if (hit_outside)
        return vec4(1.0);
    if (dbg_exit_reason == 0) return vec4(0.0, 1.0, 0.0, 1.0);
    if (dbg_exit_reason == 1) return vec4(1.0, 0.0, 0.0, 1.0);
    if (dbg_exit_reason == 2) return vec4(0.0, 0.0, 1.0, 1.0);
    return vec4(0.0, 1.0, 1.0, 1.0);
#elif SSR_DEBUG_MODE == 2
    float t = float(dbg_iterations) / float(max(u_hiz_iterations, 1u));
    return vec4(t, 0.0, 0.0, 1.0);
#elif SSR_DEBUG_MODE == 3
    return vec4(vec3(float(dbg_deepest_level) / float(HIZ_MAX_LEVEL)), 1.0);
#elif SSR_DEBUG_MODE == 4
    float n = float(max(u_hiz_iterations, 1u));
    return vec4(float(dbg_cnt_behind_l0) / n,
        float(dbg_cnt_skip_hi) / n, float(dbg_cnt_descend_hi) / n, 1.0);
#elif SSR_DEBUG_MODE == 5
    return hit ? vec4(hit_uv, 0.0, 1.0) : vec4(0.0, 0.0, 0.0, 1.0);
#elif SSR_DEBUG_MODE == 6
    if (!hit || hit_outside)
        return vec4(0.0, 0.0, 0.0, 1.0);
    ivec2 size = textureSize(u_hiz_depth, 0);
    ivec2 c = ivec2(hit_uv * vec2(size));
    float lo = 1e20;
    float hi = 0.0;
    for (int y = -1; y <= 1; y++)
    {
        for (int x = -1; x <= 1; x++)
        {
            ivec2 q = clamp(c + ivec2(x, y), ivec2(0), size - 1);
            float l = dbgLinearDepth(texelFetch(u_hiz_depth, q, 0).x);
            lo = min(lo, l);
            hi = max(hi, l);
        }
    }
    return vec4(vec3(clamp(hi / max(lo, 1e-6) - 1.0, 0.0, 1.0)), 1.0);
#elif SSR_DEBUG_MODE == 7
    ivec2 size = textureSize(u_hiz_depth, 0);
    ivec2 c = clamp(ivec2(start_ss.xy * vec2(size)), ivec2(0), size - 1);
    float surf = dbgLinearDepth(texelFetch(u_hiz_depth, c, 0).x);
    float ray_l = dbgLinearDepth(start_ss.z);
    // > 0 : ray starts in front of the depth buffer (ok), < 0 : behind it
    float r = clamp(log2(surf / max(ray_l, 1e-6)) * 0.25, -1.0, 1.0);
    return r >= 0.0 ? vec4(0.0, r, 0.0, 1.0) : vec4(-r, 0.0, 0.0, 1.0);
#else
    return vec4(1.0, 0.0, 1.0, 1.0);
#endif
}
#endif

vec3 intersectDepthPlane(vec3 o, vec3 d, float z)
{
    return o + d * z;
}

// Index of the cell that contains the given 2D position.
ivec2 getCell(vec2 screenUV, ivec2 cellCount)
{
    return ivec2(screenUV * cellCount);
}

// The number of cells in the quad tree at the given level.
ivec2 getCellCount(int level)
{
    return textureSize(u_hiz_depth, level);
}

// Returns screen space position of the intersection
// between o + d*t and the closest cell boundary at current HiZ level.
vec3 intersectCellBoundary(
    vec3 pos, vec3 dir,
    ivec2 cell, ivec2 cellCount,
    vec2 crossStep, vec2 crossOffset)
{
    vec3 intersection = vec3(0.0);

    vec2 index = cell + crossStep;
    vec2 boundary = index / vec2(cellCount); // Screen space position of the boundary
    boundary += crossOffset;

    vec2 delta = boundary - pos.xy;
    delta /= dir.xy;
    float t = min(delta.x, delta.y);

    intersection = intersectDepthPlane(pos, dir, t);
    return intersection;
}

bool crossedCellBoundary(ivec2 oldCellIx, ivec2 newCellIx)
{
    return any(notEqual(oldCellIx, newCellIx));
}

// Reverse Z: HiZ stores MAX depth (closest to camera = highest value)
float getMaxDepthPlane(ivec2 cellIx, int level)
{
    return texelFetch(u_hiz_depth, cellIx, level).x;
}

float getMaxTraceDistance(vec3 p, vec3 v)
{
    vec3 traceDistances;
    if (v.x < 0.0)
        traceDistances.x = p.x / (-v.x);
    else
        traceDistances.x = (1.0 - p.x) / v.x;

    if (v.y < 0.0)
        traceDistances.y = p.y / (-v.y);
    else
        traceDistances.y = (1.0 - p.y) / v.y;

    if (v.z < 0.0)
        traceDistances.z = p.z / (-v.z);
    else
        traceDistances.z = (1.0 - p.z) / v.z;

    return min(traceDistances.x, min(traceDistances.y, traceDistances.z));
}


// Is the ray (reverse-Z depth rayD) behind the surface (surfaceD) by more than
// its thickness, in view space. The view space depth of a reverse-Z depth value
// d (0 is the far plane) is abs(vz) / abs(vw), with
// vz = inv_proj[2][2] * d + inv_proj[3][2], vw = inv_proj[2][3] * d + inv_proj[3][3]
// so  rayZ - surfaceZ > ABS + REL * surfaceZ  is tested multiplied by
// abs(vw_ray) * abs(vw_surface) (positive), which has no division
bool isTooThick(float surfaceD, float rayD)
{
    mat4 inv_proj = u_camera.m_inverse_projection_matrix;
    float vz_s = abs(inv_proj[2][2] * surfaceD + inv_proj[3][2]);
    float vw_s = max(abs(inv_proj[2][3] * surfaceD + inv_proj[3][3]), 1e-12);
    float vz_r = abs(inv_proj[2][2] * rayD + inv_proj[3][2]);
    float vw_r = max(abs(inv_proj[2][3] * rayD + inv_proj[3][3]), 1e-12);
    return vz_r * vw_s - (1.0 + SSR_THICKNESS_REL) * vz_s * vw_r >
        SSR_THICKNESS_ABS * vw_r * vw_s;
}

// p            : Screen space position
// v            : Screen space reflection direction
// hitPointSS   : Returns screen space hit point
// Return value : Whether RT actually hit a surface
bool traceHiZ(vec3 p, vec3 v, out vec2 hitPointSS)
{
    const int maxLevel = min(HIZ_MAX_LEVEL, textureQueryLevels(u_hiz_depth) - 1); // Last mip level
    float maxTraceDistance = getMaxTraceDistance(p, v);

    // Get the cell cross direction and a small offset to enter
    // the next cell when doing cell crossing.
    vec2 crossStep = vec2(v.x >= 0 ? 1 : -1, v.y >= 0 ? 1 : -1);
    vec2 crossOffset = crossStep / u_camera.m_viewport.zw / 128.;
    crossStep = clamp(crossStep, 0.0, 1.0);

    // Set current ray to the original screen coordinate and depth.
    vec3 ray = p;
    float minZ = ray.z;
    float maxZ = ray.z + v.z * maxTraceDistance;
    float deltaZ = maxZ - minZ;

    vec3 o = ray;
    vec3 d = v * maxTraceDistance;

    int level = HIZ_START_LEVEL;
    int deepestLevel = level;
#if DEBUG_LINEAR_SEARCH
    level = 0;
#endif
    uint iterations = 0;
    bool isBackwardRay = v.z > 0;
#if SSR_DEBUG_MODE != 0
    // Is the trace ended by the far plane (z) or by the screen edge (xy)?
    float dbgTraceZ = v.z < 0.0 ? p.z / (-v.z) : (v.z > 0.0 ? (1.0 - p.z) / v.z : 1e20);
    bool dbgLimitedByZ = dbgTraceZ <= maxTraceDistance + 1e-6;
#endif
    float rayDir = isBackwardRay ? 1.0 : -1.0;

    // Cross to next cell s.t. we don't get a self-intersection immediately.
    ivec2 startCellCount = getCellCount(level);
    ivec2 rayCell = getCell(ray.xy, startCellCount);
    ray = intersectCellBoundary(o, d, rayCell, startCellCount, crossStep, crossOffset * 64.);

    while (level >= HIZ_STOP_LEVEL && ray.z * rayDir <= maxZ * rayDir &&
        iterations < u_hiz_iterations)
    {
        // Get the cell number of our current ray.
        ivec2 cellCount = getCellCount(level);
        ivec2 oldCellIx = getCell(ray.xy, cellCount);

        // Get the maximum depth plane in which the current ray resides.
        float cellMaxZ = getMaxDepthPlane(oldCellIx, level);

        // Intersect only if ray depth is above the maximum depth plane.
        vec3 tempRay;
        if (cellMaxZ < ray.z && !isBackwardRay)
            tempRay = intersectDepthPlane(o, d, (cellMaxZ - minZ) / deltaZ);
        else
            tempRay = ray;

        ivec2 newCellIx = getCell(tempRay.xy, cellCount);

        // Level 0 only: ray is behind the surface by more than its thickness
        // so it passes behind it, thickness is measured in view space
        bool tooThick = false;
        if (level == 0 && cellMaxZ > 0.0 && ray.z < cellMaxZ)
            tooThick = isTooThick(cellMaxZ, max(ray.z, 0.0));
        // Nothing was drawn there (sky, cleared to 0), never a hit
        bool noGeometry = cellMaxZ <= 0.0;

        bool crossed = (isBackwardRay && (cellMaxZ < ray.z))
                    || tooThick || noGeometry
                    || crossedCellBoundary(oldCellIx, newCellIx);

#if SSR_DEBUG_MODE != 0
        if (level == 0 && tooThick)
            dbg_cnt_behind_l0 += 1;
        else if (level > 0 && crossed)
            dbg_cnt_skip_hi += 1;
        else if (level > 0 && !crossed)
            dbg_cnt_descend_hi += 1;
#endif

        if (crossed)
        {
            ray = intersectCellBoundary(o, d, oldCellIx, cellCount, crossStep, crossOffset);
            // When passing behind a surface at level 0 stay at level 0, going
            // up and then immediately back down costs 2 iterations per pixel
            level = tooThick ? level : min(maxLevel, level + 1);
            deepestLevel = max(deepestLevel, level);
#if DEBUG_LINEAR_SEARCH
            level = 0;
#endif
        }
        else
        {
            ray = tempRay;
            level = level - 1;
        }

        iterations += 1;
    }

    // Results
    hitPointSS = ray.xy;
    bool traceHit = level < HIZ_STOP_LEVEL && iterations < u_hiz_iterations;
#if SSR_DEBUG_MODE != 0
    dbg_iterations = iterations;
    dbg_deepest_level = deepestLevel;
    if (traceHit)
        dbg_exit_reason = 0;
    else if (iterations >= u_hiz_iterations)
        dbg_exit_reason = 1;
    else
        dbg_exit_reason = dbgLimitedByZ ? 2 : 3;
#endif
    return traceHit;
}

// Traces the reflection of the view space position xpos, in the direction
// reflected (view space) over the HiZ, hitPointSS is in the viewport (0 to 1),
// startSS is the start of the trace (x, y, depth), for debugging
bool traceHiZView(vec3 xpos, vec3 reflected, mat4 projection_matrix,
                  out vec2 hitPointSS, out vec3 startSS)
{
    vec3 positionSS = CalcCoordFromPosition(xpos, projection_matrix,
        vec2(1.0), vec2(0.0));
    startSS = positionSS;
    vec3 position2VS = xpos + 1000.0 * reflected;
    vec4 position2CS = projection_matrix * vec4(position2VS, 1.0);
    position2CS /= position2CS.w;
    vec3 position2SS = position2CS.xyz;
    position2SS.xy = vec2(0.5) + 0.5 * position2SS.xy;
    vec3 reflectionDirSS = normalize(position2SS - positionSS);
    return traceHiZ(positionSS, reflectionDirSS, hitPointSS);
}
#endif
