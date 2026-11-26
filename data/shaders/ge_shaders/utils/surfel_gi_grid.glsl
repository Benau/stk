#ifndef SURFEL_GI_GRID_GLSL
#define SURFEL_GI_GRID_GLSL

const float SURFEL_CELL_SIZE = 2.0;

float getSurfelCellSize(uint level)
{
    return SURFEL_CELL_SIZE * float(1u << level);
}

ivec3 surfelGICellAt(vec3 position, float spacing)
{
    return ivec3(floor(position / spacing));
}

vec3 surfelGICenterGrid(vec3 position, float spacing)
{
    return position / spacing - 0.5;
}

vec3 surfelGIProbePosition(ivec3 cell, vec3 offset, float spacing)
{
    return (vec3(cell) + 0.5 + offset) * spacing;
}

#endif
