# SuperTuxKart bubblegum shields: skeletal pre-burst animation.
#
# Run from Blender (3.6 or newer, tested with 5.1) on bubblegum_shield.blend
# or bubblegum_shield_nolok.blend, either in the Text Editor or headless:
#   blender -b bubblegum_shield.blend --python build_bubblegum_rig.py
# Pass "-- --export path/bubblegum_shield.spm" (headless) to export the SPM
# with the STK SPM exporter (io_scene_spm) right after building the rig, and
# "--save path.blend" to save the rigged file.
#
# The shield is a closed bubble (a sphere of radius 1 around the kart, a bit
# flatter along one horizontal axis), so the rig is 12 bones pointing from its
# centre to the vertices of an icosahedron. Every vertex follows the bones
# closest to its direction, scaling a bone along its own axis pushes the part
# of the surface around it in or out. The mesh topology and UVs are not
# changed.
#
# The bubble is the mesh with the most vertices. Every other mesh (the spikes
# of Nolok's shield) moves as a whole with the part of the bubble under it:
# all its vertices get the weights of the direction of its centre, so a spike
# stays on the surface and keeps its shape (it only follows the stretch).
#
# Meshes without a material get one (the 2.66 Nolok file lost them when it
# was opened in a new Blender): the bubble an image texture with the first
# bubblegum_shield*.png image of the file, the others their vertex colours,
# which is what the exporter needs to write the same materials as the SPM in
# stk-assets.
#
# The action is the last 3 seconds of the shield, the game maps the time left
# onto it (frame 1 = 3 s left, the last frame = the pop):
# - frame 1 is the rest pose, so the shield is unchanged until then and the
#   static (bind) mesh of the SPM is the original one,
# - a jelly like squash and stretch around a slowly turning axis grows in and
#   speeds up from 2 to 6 wobbles per second,
# - in the last second a lumpier wobble joins, the gum is getting thin,
# - in the last 0.4 s the bubble swells by ~20% and then the game pops it
#   (explosion sound, gum on the road).

import bpy
import math
import sys
from mathutils import Vector

ARMATURE_OBJECT_NAME = "Bubblegum_PreBurst_Armature"
ARMATURE_DATA_NAME = "Bubblegum_PreBurst_Rig"
ACTION_NAME = "Bubblegum_PreBurst_3s"
MODIFIER_NAME = "Bubblegum_PreBurst_Skin"
GROUP_PREFIX = "Gum_Radial_"
DURATION = 3.0
# 50 samples per second: the fastest wobble (6 Hz) still has 8 samples per
# cycle, STK interpolates linearly between them.
FRAME_START = 1
FRAME_END = 151
# Higher means each bone moves a smaller part of the surface.
WEIGHT_POWER = 6.0


def smoothstep(e0, e1, x):
    x = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return x * x * (3.0 - 2.0 * x)


def icosahedron_directions():
    g = (1.0 + math.sqrt(5.0)) / 2.0
    dirs = []
    for a in (-1.0, 1.0):
        for b in (-g, g):
            dirs.append(Vector((0.0, a, b)))
            dirs.append(Vector((a, b, 0.0)))
            dirs.append(Vector((b, 0.0, a)))
    return [d.normalized() for d in dirs]


def find_meshes():
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes:
        raise RuntimeError("No mesh in the scene")
    bubble = max(meshes, key=lambda o: len(o.data.vertices))
    return bubble, [o for o in meshes if o is not bubble]


def fix_materials(bubble, others):
    if not bubble.material_slots or bubble.material_slots[0].material is None:
        images = [i for i in bpy.data.images
                  if i.name.startswith("bubblegum_shield")]
        if not images:
            raise RuntimeError("The bubble has no material and there is no "
                               "bubblegum_shield*.png image in the file")
        mat = bpy.data.materials.new(images[0].name.rsplit(".", 1)[0])
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        bsdf = nodes["Principled BSDF"]
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = images[0]
        tex.location = (-300.0, 300.0)
        mat.node_tree.links.new(tex.outputs["Color"],
                                bsdf.inputs["Base Color"])
        bubble.data.materials.clear()
        bubble.data.materials.append(mat)
        print("Added material", mat.name, "to", bubble.name)

    vcol_mat = None
    for obj in others:
        if obj.material_slots and obj.material_slots[0].material is not None:
            continue
        if vcol_mat is None:
            vcol_mat = bpy.data.materials.new("Bubblegum_Vertex_Color")
            vcol_mat.use_nodes = True
            nodes = vcol_mat.node_tree.nodes
            col = nodes.new("ShaderNodeVertexColor")
            col.location = (-300.0, 300.0)
            vcol_mat.node_tree.links.new(col.outputs["Color"],
                nodes["Principled BSDF"].inputs["Base Color"])
        if obj.data.color_attributes:
            vcol_mat.node_tree.nodes["Color Attribute"].layer_name = \
                obj.data.color_attributes[0].name
        obj.data.materials.clear()
        obj.data.materials.append(vcol_mat)
        print("Added material", vcol_mat.name, "to", obj.name)


def remove_previous_rig(mesh_objs):
    # Safe to run again, only removes what this script creates.
    for mesh_obj in mesh_objs:
        for mod in list(mesh_obj.modifiers):
            if mod.name == MODIFIER_NAME:
                mesh_obj.modifiers.remove(mod)
        for group in list(mesh_obj.vertex_groups):
            if group.name.startswith(GROUP_PREFIX):
                mesh_obj.vertex_groups.remove(group)
    old_obj = bpy.data.objects.get(ARMATURE_OBJECT_NAME)
    if old_obj is not None:
        old_data = old_obj.data
        bpy.data.objects.remove(old_obj, do_unlink=True)
        if old_data is not None and old_data.users == 0:
            bpy.data.armatures.remove(old_data)
    old_action = bpy.data.actions.get(ACTION_NAME)
    if old_action is not None:
        bpy.data.actions.remove(old_action)


def get_weights(n, dirs):
    # At most 4 influences per vertex (the SPM vertex format), normalized.
    weights = []
    for i, d in enumerate(dirs):
        w = max(0.0, n.dot(d)) ** WEIGHT_POWER
        if w > 1.0e-4:
            weights.append((w, i))
    weights.sort(reverse=True)
    weights = weights[:4]
    total = sum(w for w, _ in weights)
    return [(w / total, i) for w, i in weights]


def add_armature_modifier(obj, arm_obj):
    mod = obj.modifiers.new(MODIFIER_NAME, 'ARMATURE')
    mod.object = arm_obj
    mod.use_vertex_groups = True
    mod.use_bone_envelopes = False


def make_rig(mesh_obj, others, dirs):
    # Work in world space (the mesh object is rotated by 90 degrees): the
    # armature object has the identity transform and the bones start at the
    # world centre of the mesh.
    mw = mesh_obj.matrix_world
    world_verts = [mw @ v.co for v in mesh_obj.data.vertices]
    center = sum(world_verts, Vector()) / len(world_verts)
    radius = max((v - center).length for v in world_verts)

    arm_data = bpy.data.armatures.new(ARMATURE_DATA_NAME)
    arm_obj = bpy.data.objects.new(ARMATURE_OBJECT_NAME, arm_data)
    mesh_obj.users_collection[0].objects.link(arm_obj)
    arm_data.display_type = 'STICK'

    bpy.ops.object.select_all(action='DESELECT')
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    bpy.ops.object.mode_set(mode='EDIT')
    names = []
    for i, d in enumerate(dirs):
        bone = arm_data.edit_bones.new("{}{:02d}".format(GROUP_PREFIX, i))
        bone.head = center
        bone.tail = center + d * radius * 1.2
        bone.use_deform = True
        names.append(bone.name)
    bpy.ops.object.mode_set(mode='OBJECT')

    groups = [mesh_obj.vertex_groups.new(name=n) for n in names]
    for v, wv in zip(mesh_obj.data.vertices, world_verts):
        for w, i in get_weights((wv - center).normalized(), dirs):
            groups[i].add([v.index], w, 'REPLACE')
    add_armature_modifier(mesh_obj, arm_obj)

    for obj in others:
        verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
        middle = sum(verts, Vector()) / len(verts)
        weights = get_weights((middle - center).normalized(), dirs)
        all_ids = [v.index for v in obj.data.vertices]
        obj_groups = [obj.vertex_groups.new(name=n) for n in names]
        for w, i in weights:
            obj_groups[i].add(all_ids, w, 'REPLACE')
        add_armature_modifier(obj, arm_obj)
    return arm_obj, names


def p2(x):
    return 0.5 * (3.0 * x * x - 1.0)


def p3(x):
    return 0.5 * (5.0 * x * x * x - 3.0 * x)


def axis(theta, phi):
    return Vector((math.cos(theta) * math.sin(phi),
                   math.sin(theta) * math.sin(phi), math.cos(phi)))


def get_fcurves(action, arm_obj):
    if hasattr(action, "fcurves"):
        return action.fcurves
    # Blender 4.4+ layered actions
    slot = arm_obj.animation_data.action_slot
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(slot)
            if bag is not None:
                return bag.fcurves
    return []


def animate_rig(arm_obj, names, dirs):
    scene = bpy.context.scene
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END

    action = bpy.data.actions.new(ACTION_NAME)
    arm_obj.animation_data_create()
    arm_obj.animation_data.action = action
    bones = [arm_obj.pose.bones[n] for n in names]

    frames = FRAME_END - FRAME_START
    dt = DURATION / frames
    phase = 0.0
    for frame in range(FRAME_START, FRAME_END + 1):
        t = (frame - FRAME_START) / frames
        sec = t * DURATION
        if frame > FRAME_START:
            # Integrate the frequency so speeding up doesn't jump the phase.
            freq = 2.0 + 4.0 * t * t
            phase += 2.0 * math.pi * freq * dt

        # Squash and stretch (P2) around a slowly turning axis, the lumpier
        # P3 wobble only in the last second. Both start at 0 on frame 1.
        ramp_in = smoothstep(0.0, 0.15, t)
        amp2 = ramp_in * (0.14 + 0.12 * t)
        amp3 = 0.10 * smoothstep(0.55, 0.9, t)
        a2 = axis(0.7 * sec, 0.9 + 0.35 * math.sin(1.3 * sec))
        a3 = axis(-1.1 * sec + 2.0, 1.9 + 0.4 * math.cos(0.8 * sec))
        s2 = math.sin(phase)
        s3 = math.sin(1.7 * phase + 1.1)
        # The bubble is only 10% bigger than the kart: grow it by 6% while
        # the wobble comes in, so its dents don't let the driver poke out.
        inflate = (1.0 + 0.06 * ramp_in + 0.04 * t * t +
                   0.16 * smoothstep(0.87, 1.0, t))

        for bone, d in zip(bones, dirs):
            radial = (1.0 + amp2 * s2 * p2(d.dot(a2)) +
                      amp3 * s3 * p3(d.dot(a3)))
            bone.location = (0.0, 0.0, 0.0)
            bone.rotation_mode = 'QUATERNION'
            bone.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)
            # Local Y is the bone (radial) axis, X / Z are tangential.
            bone.scale = (inflate, inflate * radial, inflate)
            bone.keyframe_insert(data_path="scale", frame=frame,
                                 group=bone.name)

    # Every frame is keyed, linear interpolation avoids overshoot between.
    for curve in get_fcurves(action, arm_obj):
        for point in curve.keyframe_points:
            point.interpolation = 'LINEAR'
    scene.frame_set(FRAME_START)


def export_spm(mesh_objs, path):
    import addon_utils
    addon_utils.enable("io_scene_spm")
    from io_scene_spm import export_spm as spm
    bpy.ops.object.select_all(action='DESELECT')
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    spm.writeSPMFile(path, {
        "selection-type": "selected",
        "local-space": False,
        "apply-modifiers": True,
        "keyframes-only": True,
        "export-normal": True,
        "export-vcolor": True,
        "export-tangent": True,
        # Frame 1 is the rest pose, also the static mesh.
        "static-mesh-frame": FRAME_START,
    })
    print("Exported", path)


def main():
    mesh_obj, others = find_meshes()
    fix_materials(mesh_obj, others)
    remove_previous_rig([mesh_obj] + others)
    dirs = icosahedron_directions()
    arm_obj, names = make_rig(mesh_obj, others, dirs)
    animate_rig(arm_obj, names, dirs)

    # Leave every shield mesh selected for the SPM exporter.
    bpy.ops.object.select_all(action='DESELECT')
    for obj in [mesh_obj] + others:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_obj
    print("Bubblegum rig created:", mesh_obj.name, "+", len(others),
          "other meshes, bones:", len(names),
          "frames:", FRAME_START, "..", FRAME_END)

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--export" in argv:
        export_spm([mesh_obj] + others, argv[argv.index("--export") + 1])
    if "--save" in argv:
        bpy.ops.wm.save_as_mainfile(filepath=argv[argv.index("--save") + 1])


main()
