struct ProbeWork
{
    uint counts[SURFEL_GI_BANDS];
    uint slots[SURFEL_GI_BANDS * SURFEL_GI_BATCH];
};

struct SurfelGIShadow
{
    vec3 origin;
    uint target;
    vec3 segment;
    uint light;
};

layout(std430, set = SURFEL_SET, binding = 11) buffer ProbeQueue
{
    uvec4 shadow_dispatch[2];
    vec4 ray_results[SURFEL_GI_PRIMARY_BUDGET];
    SurfelGIShadow shadow_rays[SURFEL_GI_PRIMARY_BUDGET];
    ProbeWork probe_work[];
};

uint probeRayCount()
{
    return pc.m_camera_count * SURFEL_GI_BATCH * SURFEL_GI_RAYS;
}

uint surfelRayBase(uint list)
{
    uint young_rays = list == 0u ? 0u : u_trace_lists.m_selected[0] * getSurfelTraceThreads(0u) * SURFEL_RAYS_PER_THREAD;
    return probeRayCount() + young_rays;
}

uint scheduledSurfel(uint list, uint entry)
{
    uint count = u_trace_lists.m_count[list];
    uint offset = pc.m_frame * 8191u % count;
    return u_trace_lists.m_ids[list * pc.m_surfel_count + (offset + entry) % count];
}

uint scheduledProbe(uint camera, uint rank)
{
    for (int band = SURFEL_GI_BANDS - 1; band >= 0; --band)
    {
        uint count = min(probe_work[camera].counts[band], SURFEL_GI_BATCH);
        if (rank < count)
            return probe_work[camera].slots[uint(band) * SURFEL_GI_BATCH + rank];
        rank -= count;
    }
    return ~0u;
}

vec3 probeRay(uint slot, uint lane)
{
    uint seed = pc.m_frame * 1487u + slot * 5323u;
    float azimuth = (getSurfelRandom(seed) + float(lane) * 0.618033989) * 6.283185307;
    float height = 1.0 - 2.0 * (float(lane) + getSurfelRandom(seed)) / float(SURFEL_GI_RAYS);
    vec2 circle = sqrt(max(1.0 - height * height, 0.0)) * vec2(cos(azimuth), sin(azimuth));
    return vec3(circle.x, height, circle.y);
}
