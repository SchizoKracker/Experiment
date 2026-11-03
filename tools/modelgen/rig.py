"""Rig a base body: clean remesh -> game decimation -> armature from the shared skeleton ->
automatic (bone-heat) weights -> pose tests -> optional glb export.

usage: python3 rig.py <preset> <out_prefix> [--export path.glb] [--tris 40000]
"""
from __future__ import annotations

import math
import sys
import time

import numpy as np

import body
import humanoid as hu
import mesher
import races
import bpy  # noqa: E402
from mathutils import Euler, Matrix, Vector  # noqa: E402
import blendutil as bu  # noqa: E402

# bone hierarchy: name -> (head joint, tail joint, parent)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def bone_table(J):
    t = {
        "hips": ("hips", "spine", None),
        "spine": ("spine", "spine1", "hips"),
        "spine1": ("spine1", "chest", "spine"),
        "chest": ("chest", "neck", "spine1"),
        "neck": ("neck", "head", "chest"),
        "head": ("head", "head_top", "neck"),
    }
    for s in ("L", "R"):
        t[f"clavicle.{s}"] = (f"clavicle.{s}", f"upper_arm.{s}", "chest")
        t[f"upper_arm.{s}"] = (f"upper_arm.{s}", f"forearm.{s}", f"clavicle.{s}")
        t[f"forearm.{s}"] = (f"forearm.{s}", f"hand.{s}", f"upper_arm.{s}")
        t[f"hand.{s}"] = (f"hand.{s}", f"middle1.{s}", f"forearm.{s}")
        for f in FINGERS:
            for i in (1, 2, 3):
                tail = f"{f}{i + 1}.{s}" if i < 3 else f"{f}_end.{s}"
                parent = f"hand.{s}" if i == 1 else f"{f}{i - 1}.{s}"
                t[f"{f}{i}.{s}"] = (f"{f}{i}.{s}", tail, parent)
        t[f"thigh.{s}"] = (f"thigh.{s}", f"shin.{s}", "hips")
        t[f"shin.{s}"] = (f"shin.{s}", f"foot.{s}", f"thigh.{s}")
        t[f"foot.{s}"] = (f"foot.{s}", f"toe.{s}", f"shin.{s}")
        t[f"toe.{s}"] = (f"toe.{s}", f"toe_end.{s}", f"foot.{s}")
    return t


def build_armature(J, name="rig"):
    arm = bpy.data.armatures.new(name)
    ob = bpy.data.objects.new(name, arm)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    table = bone_table(J)
    for bname, (h, tl, par) in table.items():
        b = eb.new(bname)
        b.head = Vector(J[h].tolist())
        b.tail = Vector(J[tl].tolist())
        if (b.tail - b.head).length < 1e-4:
            b.tail = b.head + Vector((0, 0, 0.02))
    for bname, (h, tl, par) in table.items():
        if par:
            eb[bname].parent = eb[par]
            eb[bname].use_connect = False
    bpy.ops.armature.select_all(action="SELECT")
    bpy.ops.armature.calculate_roll(type="GLOBAL_NEG_Y")
    # thumbs/fingers: roll so they curl toward the palm with +X rotation (consistent per side)
    bpy.ops.object.mode_set(mode="OBJECT")
    return ob


def clean_mesh(v, t, voxel, target_tris, J=None):
    ob = bu.mesh_object("body", v, t)
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
    # NOTE: Blender's voxel remesh silently collapses below ~5 mm (voxel-count cap), so instead we
    # clean the surface-nets mesh directly: weld near-duplicates and drop degenerate faces.
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=voxel * 0.05)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=1e-6)
    # close sliver holes and make all normals point outward
    bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=12)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(ob.data)
    bm.free()
    # protect the face, hands and feet from decimation (they carry the detail people look at)
    vg = ob.vertex_groups.new(name="detail")
    if J is not None:
        co = np.empty(len(ob.data.vertices) * 3, np.float32)
        ob.data.vertices.foreach_get("co", co)
        co = co.reshape(-1, 3)
        w = np.zeros(len(co), np.float32)
        hh = J["_head_h"]
        w = np.maximum(w, np.clip((co[:, 2] - (J["head_top"][2] - 1.05 * hh)) / (0.1 * hh), 0, 1))
        for s in ("L", "R"):
            for key, rad in ((f"hand.{s}", 0.13), (f"foot.{s}", 0.16)):
                c = J[key] + (J[f"middle_end.{s}"] - J[key]) * 0.5 if key.startswith("hand") else J[key]
                d = np.linalg.norm(co - c, axis=1)
                w = np.maximum(w, np.clip((rad - d) / (rad * 0.4), 0, 1))
        for i, wi in enumerate(w):
            if wi > 0:
                vg.add([i], float(wi), "REPLACE")
    n = len(ob.data.polygons)
    d = ob.modifiers.new("dec", "DECIMATE")
    d.ratio = min(1.0, target_tris / max(n * 2, 1))
    d.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier="dec")
    ob.vertex_groups.remove(ob.vertex_groups["detail"])
    ob.data.shade_smooth()
    return ob


def skin(mesh_ob, rig_ob):
    bpy.ops.object.select_all(action="DESELECT")
    mesh_ob.select_set(True)
    rig_ob.select_set(True)
    bpy.context.view_layer.objects.active = rig_ob
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    # soften bone-heat transitions (hips/shoulders deform smoothly instead of creasing)
    bpy.ops.object.select_all(action="DESELECT")
    mesh_ob.select_set(True)
    bpy.context.view_layer.objects.active = mesh_ob
    bpy.ops.object.mode_set(mode="WEIGHT_PAINT")
    bpy.ops.object.vertex_group_smooth(group_select_mode="ALL", factor=0.5, repeat=4, expand=0.0)
    # the hip/glute region needs a much broader hips<->thigh transition than bone heat gives
    for g in ("hips", "thigh.L", "thigh.R", "spine"):
        mesh_ob.vertex_groups.active_index = mesh_ob.vertex_groups[g].index
        bpy.ops.object.vertex_group_smooth(group_select_mode="ACTIVE", factor=0.6, repeat=10, expand=0.15)
    bpy.ops.object.vertex_group_normalize_all(group_select_mode="ALL", lock_active=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    return len(mesh_ob.vertex_groups)


POSES = {
    "rest": {},
    "elbows_knees": {"forearm.L": (0, 0, -95), "forearm.R": (0, 0, 95), "shin.L": (100, 0, 0), "shin.R": (100, 0, 0),
                     "thigh.L": (-60, 0, 0), "thigh.R": (-20, 0, 0)},
    "arms_up": {"upper_arm.L": (0, 0, 70), "upper_arm.R": (0, 0, -70), "forearm.L": (0, 0, 20), "forearm.R": (0, 0, -20)},
    "stride": {"thigh.L": (-35, 0, 0), "thigh.R": (25, 0, 0), "shin.R": (40, 0, 0), "upper_arm.L": (0, 25, -35),
               "upper_arm.R": (0, -25, 35), "spine1": (0, 8, 0), "fist_L": 1},
    "squat": {"thigh.L": (-100, 0, -10), "thigh.R": (-100, 0, 10), "shin.L": (120, 0, 0), "shin.R": (120, 0, 0),
              "foot.L": (-25, 0, 0), "foot.R": (-25, 0, 0), "spine": (25, 0, 0), "upper_arm.L": (0, 0, -30), "upper_arm.R": (0, 0, 30)},
}


def apply_pose(rig_ob, pose):
    for pb in rig_ob.pose.bones:
        pb.rotation_mode = "XYZ"
        pb.rotation_euler = (0, 0, 0)
    fist = pose.get("fist_L")
    for bname, rot in pose.items():
        if bname.startswith("fist"):
            continue
        pb = rig_ob.pose.bones[bname]
        pb.rotation_euler = Euler([math.radians(a) for a in rot], "XYZ")
    if fist:
        for f in FINGERS[1:]:
            for i in (1, 2, 3):
                rig_ob.pose.bones[f"{f}{i}.L"].rotation_euler = Euler((math.radians(70), 0, 0), "XYZ")
    bpy.context.view_layer.update()


def main():
    preset = sys.argv[1]
    out = sys.argv[2]
    export = sys.argv[sys.argv.index("--export") + 1] if "--export" in sys.argv else None
    tris = int(sys.argv[sys.argv.index("--tris") + 1]) if "--tris" in sys.argv else 80000
    t0 = time.time()
    p = races.preset(preset)
    J = hu.build_skeleton(p)
    s = body.build_body(p, J)
    v, t = mesher.mesh_shape(s, h=0.004, verbose=False)
    import anatomy
    ev, et = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0012, verbose=False)
    print("meshed", round(time.time() - t0, 1), len(t))
    bu.reset()
    mesh_ob = clean_mesh(v, t, 0.0035, tris, J)
    print("cleaned", round(time.time() - t0, 1), "tris", len(mesh_ob.data.polygons))
    eyes = bu.mesh_object("eyes", ev, et)
    eyes.data.materials.append(bu.material("eye", (0.9, 0.9, 0.88), rough=0.15))
    rig_ob = build_armature(J)
    ngroups = skin(mesh_ob, rig_ob)
    # eyes ride the head bone rigidly
    eyes.parent = rig_ob
    eyes.parent_type = "BONE"
    eyes.parent_bone = "head"
    eyes.matrix_parent_inverse = (rig_ob.matrix_world @ rig_ob.pose.bones["head"].matrix @ Matrix.Translation((0, rig_ob.data.bones["head"].length, 0))).inverted()
    # verify every vertex has weight
    me = mesh_ob.data
    unweighted = sum(1 for vx in me.vertices if len(vx.groups) == 0)
    print("skinned", round(time.time() - t0, 1), "groups", ngroups, "unweighted verts", unweighted)
    mesh_ob.data.materials.append(bu.material("clay", (0.55, 0.45, 0.4), rough=0.5, sss=0.05))
    bu.setup_render((520, 760), 12)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.03, 0.03, 0.035, 1)
    H = p["height"]
    bu.add_light("AREA", (-2.2, -2.6, H * 2.0), 90 * H, (1, 0.97, 0.94), 1.4, (0, 0, H * 0.6))
    bu.add_light("AREA", (2.2, 1.5, H * 1.4), 40 * H, (0.8, 0.85, 1.0), 2.0, (0, 0, H * 0.6))
    cam = bu.camera((2.4 * H, -3.2 * H, H * 0.7), (0, 0, H * 0.48), lens=50)
    shots = []
    for name, pose in POSES.items():
        apply_pose(rig_ob, pose)
        path = f"/tmp/claude-0/prev/_pose_{name}.png"
        bu.render(path)
        shots.append(path)
    apply_pose(rig_ob, {})
    from PIL import Image
    ims = [Image.open(x).convert("RGB") for x in shots]
    sheet = Image.new("RGB", (sum(i.width for i in ims), ims[0].height))
    x = 0
    for im in ims:
        sheet.paste(im, (x, 0))
        x += im.width
    sheet.save(out + "_poses.png")
    bpy.ops.wm.save_as_mainfile(filepath=out + ".blend")
    if export:
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.export_scene.gltf(filepath=export, export_format="GLB", use_selection=True, export_skins=True,
                                  export_apply=False, export_animations=False)
        print("exported", export)
    print("done", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
