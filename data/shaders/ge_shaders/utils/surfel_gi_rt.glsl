// How the surfels of GEVulkanSurfelGI are attached to the triangles of the ray
// traced scene (GEVulkanRTScene) and how they follow them between the frames:
//
// A surfel stores the key of the node of the triangle it was created on, the
// index of the geometry in the instance of the node, the index of the triangle
// and the two barycentrics of the point in it (surfelAttach). Every frame
// surfelFollow recomputes the world position and the normal of the surfel from
// the vertices of the triangle and the transformation of the instance of the
// key (the TLAS instance buffer, see utils/rt_scene.glsl), so a surfel of a
// kart follows it when it moves, it's not left behind (attached to a triangle
// which left its position) and it doesn't have to be created again. A triangle
// which turns under the surfel (a wheel) is not the surface its samples came
// from: surfel_gi_update.comp sees the change of the normal and resets the
// history of the surfel.
//
// The static geometry (the key 0) is already in world space and doesn't move.
// The dynamic nodes are in the model space of their BLAS and the skinned ones
// are in world space relative to the origin of their node, so the transform of
// the node of the two of them is the model to world matrix. A node which is
// gone (or hidden) leaves its slot with the static instance and the key 0, and
// a slot reused by another node has a new key, so the check fails and the
// surfel is recycled.
//
// The shader which includes this file must have, before it, the include of
// utils/surfel_gi.glsl, utils/camera.glsl and the ones which
// utils/rt_scene.glsl needs, and #extension GL_EXT_ray_query : require in its
// own file (surfelAttach traces a ray)

// Transformation of a world position and a normal of the node (row major 3x4,
// the translation is the w of the rows)
vec3 surfelNodePoint(RTInstance instance, vec3 position)
{
    return vec3(dot(instance.m_transform[0].xyz, position) +
        instance.m_transform[0].w,
        dot(instance.m_transform[1].xyz, position) + instance.m_transform[1].w,
        dot(instance.m_transform[2].xyz, position) +
        instance.m_transform[2].w);
}

// A normal doesn't transform like a direction: the surface of a node with a
// non uniform scale is skewed by it, and the normal of the skewed surface is
// the inverse transpose of the matrix times the normal. The cofactor matrix is
// the inverse transpose times the determinant (the cross products of the rows,
// sign corrected by the determinant), so the result has the direction of the
// normal and the callers normalize it. For a rotation (the wheels of a kart)
// and for a uniform scale it's the matrix itself, so this only changes the
// result where the old plain direction transform was wrong
vec3 surfelNodeNormal(RTInstance instance, vec3 normal)
{
    const vec3 r0 = instance.m_transform[0].xyz;
    const vec3 r1 = instance.m_transform[1].xyz;
    const vec3 r2 = instance.m_transform[2].xyz;
    const float det = dot(r0, cross(r1, r2));
    return sign(det) * vec3(dot(cross(r1, r2), normal),
        dot(cross(r2, r0), normal), dot(cross(r0, r1), normal));
}

// The world position and normal of the surfel of the triangle it's attached to
// in this frame. Returns false if the node of the triangle is gone or the slot
// of its key is used by another node, so the surfel can't be followed
bool surfelFollow(Surfel surfel, out vec3 position, out vec3 normal)
{
    position = surfel.m_position;
    normal = surfel.m_normal;
    // The static geometry is in world space
    if (surfel.m_node_key == 0u)
        return true;

    const uint slot = surfel.m_node_key & (RT_INSTANCE_CAPACITY - 1u);
    const RTInstance instance = u_rt_instances[slot];
    // The slot is reused by another node, or the node is not traced anymore
    // (the slot has the static instance, which has the key 0)
    if ((instance.m_sbt_offset_flags & RT_INSTANCE_BITS_MASK) !=
        surfel.m_node_key)
        return false;

    const uint geometry_index =
        (instance.m_custom_index_mask & RT_INSTANCE_BITS_MASK) +
        surfel.m_geometry;
    const RTGeometry geometry = u_rt_geometries[geometry_index];
    const uint primitive = surfel.m_primitive;
    const uvec3 indices = uvec3(getRTIndex(geometry, primitive * 3u),
        getRTIndex(geometry, primitive * 3u + 1u),
        getRTIndex(geometry, primitive * 3u + 2u));
    // The 2 16 bit unorms of the barycentrics of the attachment
    const vec2 barycentrics = vec2(float(surfel.m_bary & 0xFFFFu),
        float(surfel.m_bary >> 16u)) / 65535.0;
    const vec3 bary = vec3(1.0 - barycentrics.x - barycentrics.y,
        barycentrics.x, barycentrics.y);
    // The vertices are in the space of the BLAS (see getRTVertexPosition)
    const vec3 local_position = getRTVertexPosition(geometry, indices.x) *
        bary.x + getRTVertexPosition(geometry, indices.y) * bary.y +
        getRTVertexPosition(geometry, indices.z) * bary.z;
    const vec3 local_normal = getRTVertexNormal(geometry, indices.x) *
        bary.x + getRTVertexNormal(geometry, indices.y) * bary.y +
        getRTVertexNormal(geometry, indices.z) * bary.z;

    position = surfelNodePoint(instance, local_position);
    normal = surfelNodeNormal(instance, local_normal);
    if (normal == vec3(0.0))
        return false;
    normal = normalize(normal);
    return true;
}

// The triangle of the surface of a world position, traced from the camera of
// the viewport (the same surface which the g-buffer of the pixel shows). The
// position, the geometry and the triangle of the hit are returned so a surfel
// can be attached to it (surfelFollow). Returns false if there is no hit near
// the position (the surface is not in the ray traced scene)
bool surfelAttach(vec3 world_position, out uint node_key, out uint geometry,
                  out uint primitive, out uint bary)
{
    node_key = 0u;
    geometry = 0u;
    primitive = 0u;
    bary = 0u;
    const vec3 camera_position = u_camera.m_inverse_view_matrix[3].xyz;
    const vec3 to_position = world_position - camera_position;
    const float distance_to_position = length(to_position);
    if (distance_to_position < 1.0e-4)
        return false;
    const vec3 direction = to_position / distance_to_position;
    // A bit farther than the surface, the barycenter of a triangle is farther
    // than its plane. The tolerance is small enough that a surfel is only
    // attached to the triangle it was spawned on and not to the one behind it,
    // whatever the distance
    const float tolerance = max(0.1, distance_to_position * 0.005);

    rayQueryEXT query;
    rayQueryInitializeEXT(query, u_tlas, 0, 0xFF, camera_position, 0.0,
        direction, distance_to_position + tolerance);
    RT_PROCEED(query)
    if (rayQueryGetIntersectionTypeEXT(query, true) !=
        gl_RayQueryCommittedIntersectionTriangleEXT)
        return false;
    const vec3 hit = camera_position + direction *
        rayQueryGetIntersectionTEXT(query, true);
    if (distance(hit, world_position) > tolerance)
        return false;

    // The key of the node of the instance of the hit (the slot of the
    // instance), see rt_scene.glsl
    node_key = rayQueryGetIntersectionInstanceShaderBindingTableRecordOffsetEXT(
        query, true);
    geometry = uint(rayQueryGetIntersectionGeometryIndexEXT(query, true));
    primitive = uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true));
    const vec2 barycentrics =
        rayQueryGetIntersectionBarycentricsEXT(query, true);
    if (node_key == 0u)
    {
        // The static geometry: the hit is in the one BLAS of the scene, so the
        // index of the geometry is the custom index of its instance
        geometry = uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query,
            true)) + geometry;
    }
    bary = uint(round(clamp(barycentrics.x, 0.0, 1.0) * 65535.0)) |
        (uint(round(clamp(barycentrics.y, 0.0, 1.0) * 65535.0)) << 16u);
    return true;
}
