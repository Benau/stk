// The ray traced reflection of the displace materials, see displace_mask.frag
// (this is the same shader, the reflected ray is traced in the ray traced
// scene), used by GEVulkanDrawCall if isDisplaceRayTracing()
#extension GL_EXT_ray_query : require

#ifdef PBR_ENABLED
#define DISPLACE_RT
// The pipeline layout of solid_rt.frag: the scene is set 0 (RT_SCENE_SET is the
// default one), the camera and the global light are set 1, the environment is
// set 3 and the mesh textures are set 4. The mesh textures are all bound
// at once with the ray traced scene, so the displace mask is sampled from them
// too and the mesh textures of the draw (the default set 0) are not used
#define DISPLACE_ENV_SET 3
#define MESH_TEXTURES_SET 4
#endif

#include "displace_mask.frag"
