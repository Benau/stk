// The wind of the grass, in the model space of the mesh (before the
// transformation of the node). It's shared by grass.vert and the shader which
// skins the BLAS of the grass for ray tracing (the vertices of the BLAS are
// in the model space too, the instance transform moves them to the world), so
// both are the same
//
// wind_direction is the push constant of grass.vert (the wind direction of
// GEMaterialManager::getWindDirection), model_y is v_position.y (the rest
// pose) and the result is multiplied by the red of the vertex color (the
// weight) by the caller
vec3 getGrassWindOffset(vec3 wind_direction, float model_y)
{
    vec3 offset = sin(wind_direction * (model_y * 0.1));
    offset += vec3(cos(wind_direction) * 0.7);
    return offset;
}
