// The polygons of the visible objects of each workgroup, always written (even
// if zero) so it needs no clearing, the CPU sums it a few frames later
layout(std430, set = 0, binding = 4) writeonly buffer Stats
{
    uint group_polys[];
};

shared uint s_polys;

// ----------------------------------------------------------------------------
void statsBegin()
{
    if (gl_LocalInvocationIndex == 0u)
        s_polys = 0u;
    memoryBarrierShared();
    barrier();
}   // statsBegin

// ----------------------------------------------------------------------------
void statsAddVisible(uint polygons)
{
    atomicAdd(s_polys, polygons);
}   // statsAddVisible

// ----------------------------------------------------------------------------
void statsEnd()
{
    memoryBarrierShared();
    barrier();
    if (gl_LocalInvocationIndex == 0u)
        group_polys[gl_WorkGroupID.x] = s_polys;
}   // statsEnd
