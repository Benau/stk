// The shading of the hit of a ray traced reflection, shared by the passes
// which trace a reflected ray in the ray traced scene (GEVulkanRTScene): the
// reflection of the solid materials (solid_rt.frag, a fullscreen draw of the
// lighting pass) and the one of the displace materials (displace_mask_rt.frag,
// which is displace_mask.frag with DISPLACE_RT defined).
//
// The hit is lit like solid.frag + deferred_pbr.frag: albedo of the mesh
// textures (with the hue, vertex color and uv shift of the node, which are
// refreshed every frame by GEVulkanRTScene::update), PBR data of its third
// texture, the sun with a shadow ray, the ambient from the diffuse (irradiance)
// and specular cubemaps of the environment (skytop * ambient color without
// IBL). Point lights are not done yet.
//
// The index of the geometry of a hit is the custom index of its instance (the
// first GERTGeometry of the instance) plus the geometry index of the hit in
// the BLAS of the instance. Vertices of an instance which is not in world space
// are in the space of its BLAS, so the normal is transformed to world space by
// the world to object matrix of the hit.
//
// The vertices and indices of a geometry are in the data buffer (GERTVertex, 32
// bit indices) or, if the geometry is of a dynamic node, in the mesh cache of
// the driver (GEVulkanMeshCache, the same as the raster pass uses, 16 bit
// indices), see m_flags. A vertex is the same 8 words in both: position,
// packed normal, linear vertex color, 2 half uvs, 2 half uvs of the second uv
// and packed tangent (in world space, the ones of the mesh cache are in the
// space of the BLAS of the instance).
//
// The shader which includes this file must have, before it (the sets are the
// ones of its pipeline layout, RT_SCENE_SET and MESH_TEXTURES_SET are defined
// by it if they're not 0):
// - #extension GL_EXT_ray_query : require in its own file
// - u_diffuse and u_specular (cubemaps of the environment) and u_shadow_map (only
//   to share pbr_light.glsl, the shadow of a hit is a ray)
// - the includes of utils/camera.glsl, utils/constants_utils.glsl,
//   utils/pbr_utils.glsl, utils/get_vertex_color.glsl,
//   utils/sample_mesh_texture.glsl, ../utils/rgb_conversion.frag,
//   utils/global_light_data.glsl, utils/pbr_light.glsl, utils/rt_utils.glsl and
//   utils/rt_scene.glsl

// The albedo of solid.frag and normalmap.frag (hue, vertex color and the
// texture)
vec3 getSolidAlbedo(int base, vec2 uv, float lod, float hue,
                    vec4 vertex_color)
{
    vec4 tex = sampleHitTexture(base, 0, uv, lod);
    if (hue > 0.0)
    {
        float mask = tex.a;
        vec3 old_hsv = rgbToHsv(tex.rgb);
        float mask_step = step(mask, 0.5);
        float saturation = mask * 2.5;
        vec2 new_xy = mix(vec2(old_hsv.x, old_hsv.y),
            vec2(hue, max(old_hsv.y, saturation)),
            vec2(mask_step, mask_step));
        tex = vec4(hsvToRgb(vec3(new_xy.x, new_xy.y, old_hsv.z)), 1.0);
    }
    return tex.rgb * vertex_color.rgb;
}

// splatting.frag, the 4 detail layers (2 to 5) are mixed by the splatting map
// (layer 1, second uv). The low resolution sampling of the far distance
// (mitigation of repetitive patterns) uses the distance to the camera, and as
// its uv is the half one, it's one level less blurry
vec3 getSplattingAlbedo(int base, vec2 uv, vec2 uv_two, float lod,
                        float camera_distance)
{
    float mitigation = clamp(pow(camera_distance * 0.01, 2.0), 0.0, 1.0);
    float low_lod = max(lod - 1.0, 0.0);
    vec4 splatting = sampleHitTexture(base, 1, uv_two, lod);
    vec4 detail0 = mix(sampleHitTexture(base, 2, uv, lod),
        sampleHitTexture(base, 2, uv * 0.5, low_lod), mitigation);
    vec4 detail1 = mix(sampleHitTexture(base, 3, uv, lod),
        sampleHitTexture(base, 3, uv * 0.5, low_lod), mitigation);
    vec4 detail2 = mix(sampleHitTexture(base, 4, uv, lod),
        sampleHitTexture(base, 4, uv * 0.5, low_lod), mitigation);
    vec4 detail3 = mix(sampleHitTexture(base, 5, uv, lod),
        sampleHitTexture(base, 5, uv * 0.5, low_lod), mitigation);
    return (splatting.r * detail0 + splatting.g * detail1 +
        splatting.b * detail2 +
        max(0.0, (1.0 - splatting.r - splatting.g - splatting.b)) *
        detail3).xyz;
}

// The material of the hit like the fragment shader of the geometry writes it to
// the g-buffer: the albedo, x: gloss, y: metallic, z: emissive of pbr, and the
// (not two sided yet) normal, which the shader can change. uv is the first
// one (with the uv shift of the node), uv_two the second one, which has no
// shift
// tangent is in world space, w is the sign of the bitangent
void getHitMaterial(RTGeometry geometry, int base, vec2 uv, vec2 uv_two,
                    vec4 vertex_color, vec4 tangent, float lod,
                    float camera_distance, out vec3 diffuse_color,
                    out vec3 pbr, inout vec3 normal)
{
    if (geometry.m_shading_type == RT_SHADING_ALPHATEST ||
        geometry.m_shading_type == RT_SHADING_UNLIT ||
        geometry.m_shading_type == RT_SHADING_GRASS)
    {
        // alphatest.frag and unlit.frag (the alpha was tested by the ray),
        // the hue only changes the hue of the color
        vec3 tex_color = sampleHitTexture(base, 0, uv, lod).rgb;
        if (geometry.m_hue > 0.0)
        {
            vec3 old_hsv = rgbToHsv(tex_color);
            tex_color = hsvToRgb(vec3(geometry.m_hue, old_hsv.y, old_hsv.z));
        }
        // The vertex color of grass.vert is white
        if (geometry.m_shading_type == RT_SHADING_GRASS)
            diffuse_color = tex_color;
        else
            diffuse_color = tex_color * vertex_color.rgb;
        // unlit.frag has only a little emissive, no gloss and no metallic
        if (geometry.m_shading_type == RT_SHADING_UNLIT)
            pbr = vec3(0.0, 0.0, 0.4);
        else
            pbr = sampleHitTexture(base, 2, uv, lod).xyz;
        return;
    }
    if (geometry.m_shading_type == RT_SHADING_DECAL)
    {
        // decal.frag, the second layer is over the first one, no vertex
        // color and no hue
        vec4 color = sampleHitTexture(base, 0, uv, lod);
        vec4 layer_two = sampleHitTexture(base, 1, uv_two, lod);
        diffuse_color = layer_two.a * layer_two.rgb +
            color.rgb * (1.0 - layer_two.a);
        pbr = sampleHitTexture(base, 2, uv, lod).xyz;
        return;
    }
    if (geometry.m_shading_type == RT_SHADING_SPLATTING)
    {
        // No gloss, metallic and emissive
        diffuse_color = getSplattingAlbedo(base, uv, uv_two, lod,
            camera_distance);
        pbr = vec3(0.0);
        return;
    }
    // solid.frag and normalmap.frag
    diffuse_color = getSolidAlbedo(base, uv, lod, geometry.m_hue,
        vertex_color);
    pbr = sampleHitTexture(base, 2, uv, lod).xyz;
    if (geometry.m_shading_type == RT_SHADING_NORMALMAP &&
        dot(tangent.xyz, tangent.xyz) > 0.0)
    {
        // normalmap.frag, the same tangent space (not orthogonalized)
        vec3 tangent_space_normal =
            2.0 * sampleHitTexture(base, 3, uv, lod).xyz - 1.0;
        vec3 frag_tangent = normalize(tangent.xyz);
        vec3 frag_normal = normalize(normal);
        vec3 frag_bitangent =
            normalize(cross(frag_normal, tangent.xyz) * tangent.w);
        normal = normalize(mat3(frag_tangent, frag_bitangent, frag_normal) *
            tangent_space_normal);
    }
}

// The lit (hdr) color of the closest hit, the values of the hit are
// parameters because a ray query can't be one
vec3 shadeHit(uint geometry_index, uint primitive, vec2 b, float hit_distance,
              mat3 world_to_object, mat3 object_to_world, vec3 ray_origin,
              vec3 ray_direction)
{
    vec3 bary = vec3(1.0 - b.x - b.y, b.x, b.y);

    RTGeometry geometry = u_rt_geometries[geometry_index];
    uvec3 idx = uvec3(getRTIndex(geometry, primitive * 3u),
        getRTIndex(geometry, primitive * 3u + 1u),
        getRTIndex(geometry, primitive * 3u + 2u));

    vec2 uv = getRTVertexUV(geometry, idx.x) * bary.x +
        getRTVertexUV(geometry, idx.y) * bary.y +
        getRTVertexUV(geometry, idx.z) * bary.z;
    vec2 uv_two = getRTVertexUVTwo(geometry, idx.x) * bary.x +
        getRTVertexUVTwo(geometry, idx.y) * bary.y +
        getRTVertexUVTwo(geometry, idx.z) * bary.z;
    uv += vec2(geometry.m_uv_shift_x, geometry.m_uv_shift_y);
    vec4 vertex_color =
        getRTVertexColor(geometry, idx.x) * bary.x +
        getRTVertexColor(geometry, idx.y) * bary.y +
        getRTVertexColor(geometry, idx.z) * bary.z;
    vertex_color *= getVertexColor(geometry.m_vertex_color);
    vec3 normal = normalize(
        getRTVertexNormal(geometry, idx.x) * bary.x +
        getRTVertexNormal(geometry, idx.y) * bary.y +
        getRTVertexNormal(geometry, idx.z) * bary.z);
    // The inverse transpose of object to world (row vector times matrix)
    normal = normalize(normal * world_to_object);
    vec3 eyedir = -ray_direction;
    // The tangents of a dynamic geometry are in the space of its BLAS, the sign
    // of the bitangent is the one of the first vertex
    vec4 tangent = getRTVertexTangent(geometry, idx.x) * bary.x +
        getRTVertexTangent(geometry, idx.y) * bary.y +
        getRTVertexTangent(geometry, idx.z) * bary.z;
    tangent.xyz = object_to_world * tangent.xyz;
    tangent.w = getRTVertexTangent(geometry, idx.x).w >= 0.0 ? 1.0 : -1.0;

    // No derivatives in a ray query, the farther the hit the blurrier
    float lod = clamp(log2(1.0 + hit_distance * 0.5), 0.0, 8.0);
    int base = int(geometry.m_material_id) * TOTAL_MESH_TEXTURE_LAYER;
    vec3 diffuse_color;
    vec3 pbr;
    getHitMaterial(geometry, base, uv, uv_two, vertex_color, tangent, lod,
        distance(ray_origin + ray_direction * hit_distance,
        u_camera.m_inverse_view_matrix[3].xyz), diffuse_color, pbr, normal);
    // Two sided
    if (dot(normal, eyedir) < 0.0)
        normal = -normal;
    float perceptual_roughness = 1.0 - pbr.x;

    vec3 world_position = ray_origin + ray_direction * hit_distance;
    vec3 sun_direction = normalize(u_global_light.m_sun_direction);
    float shadow = 1.0;
    // Same as the lighting pass: the sun has a shadow if it has a shadow map or
    // it's ray traced
    const bool sun_shadow =
        ((u_shadow_type & GST_SUN) != 0 && u_shadow_size != 0) ||
        (u_rt_shadow_type & GST_SUN) != 0;
    if (sun_shadow && dot(normal, sun_direction) > 0.0)
    {
        float view_depth =
            (u_camera.m_view_matrix * vec4(world_position, 1.0)).z;
        shadow = getSunShadow(world_position, normal, sun_direction,
            view_depth);
    }

    // Same as handlePBRDeferred, in world space (no sun disk, no ssao)
    vec3 irradiance = vec3(0.0);
    vec3 radiance = vec3(0.0);
    if (u_ibl)
    {
        vec3 reflection = reflect(ray_direction, normal);
        irradiance = texture(u_diffuse, normal).rgb;
        radiance = textureLod(u_specular, reflection,
            perceptual_roughness * u_specular_levels_minus_one).rgb;
    }
    vec3 hdr = PBRSunAmbientEmitLight(normal, eyedir, sun_direction,
        diffuse_color, irradiance, radiance, u_global_light.m_sun_color,
        u_global_light.m_ambient_color, perceptual_roughness, pbr.y, pbr.z,
        shadow, vec3(1.0), vec3(1.0));

    // TODO: point lights (accumulateLights of the lighting pass, in world
    // space, with the shadows of the lights as a ray too)
    return hdr;
}

// Traces a reflected ray, returns false if it has no hit (the caller decides
// what a miss is, the environment or the lit color which is already there),
// hdr is the lit color of the closest hit otherwise. direction is normalized
bool traceReflection(vec3 origin, vec3 direction, out vec3 hdr)
{
    rayQueryEXT ray_query;
    rayQueryInitializeEXT(ray_query, u_tlas,
        gl_RayFlagsCullBackFacingTrianglesEXT, 0xFF, origin, RT_RAY_MIN,
        direction, RT_RAY_MAX);
    RT_PROCEED(ray_query)
    if (rayQueryGetIntersectionTypeEXT(ray_query, true) ==
        gl_RayQueryCommittedIntersectionNoneEXT)
    {
        hdr = vec3(0.0);
        return false;
    }
    uint geometry_index =
        uint(rayQueryGetIntersectionInstanceCustomIndexEXT(ray_query, true)) +
        uint(rayQueryGetIntersectionGeometryIndexEXT(ray_query, true));
    hdr = shadeHit(geometry_index,
        uint(rayQueryGetIntersectionPrimitiveIndexEXT(ray_query, true)),
        rayQueryGetIntersectionBarycentricsEXT(ray_query, true),
        rayQueryGetIntersectionTEXT(ray_query, true),
        mat3(rayQueryGetIntersectionWorldToObjectEXT(ray_query, true)),
        mat3(rayQueryGetIntersectionObjectToWorldEXT(ray_query, true)),
        origin, direction);
    return true;
}
