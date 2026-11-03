"""Render a dressed character in a warm, game-like light setup.
usage: python3 preview2.py <out_prefix> <preset> [outfit] [hair] [beard]
"""
import sys, time
import numpy as np
import races, character
import bpy
import blendutil as bu

PALETTE = {
    "cloth_a": ((0.25, 0.55, 0.75), 0.8), "cloth_b": ((0.78, 0.7, 0.55), 0.85), "cloth_c": ((0.85, 0.83, 0.78), 0.9),
    "leather": ((0.18, 0.11, 0.07), 0.55), "leather_dark": ((0.12, 0.08, 0.06), 0.5), "hair": ((0.09, 0.05, 0.03), 0.45),
}

def main():
    out, preset = sys.argv[1], sys.argv[2]
    outfit = sys.argv[3] if len(sys.argv) > 3 else "commoner"
    hair = sys.argv[4] if len(sys.argv) > 4 else "short"
    beard = sys.argv[5] if len(sys.argv) > 5 else "none"
    t = time.time()
    p = races.preset(preset)
    import os
    head_only = os.environ.get("HEADONLY") == "1"
    J, parts = character.build_parts(p, outfit, hair, beard, h=0.0028 if head_only else 0.0045, head_only=head_only)
    print("built", round(time.time() - t, 1), {k: len(v["t"]) for k, v in parts.items()})
    bu.reset()
    skin = np.array(races.skin_color(p))
    mats = {}
    for name, part in parts.items():
        ob = bu.mesh_object(name, part["v"], part["t"])
        if name == "body":
            cols = np.tile(skin, (len(part["v"]), 1))
            tags = part["tags"]
            cols[tags == "lips"] = skin * np.array([1.0, 0.62, 0.62])
            cols[tags == "nail"] = skin * 1.15 + 0.05
            cols[tags == "tusk"] = (0.85, 0.8, 0.65)
            cols[tags == "horn"] = (0.12, 0.09, 0.08)
            bu.set_vertex_colors(ob, np.clip(cols, 0, 1))
            ob.data.materials.append(bu.vc_material("skin", rough=0.5, sss=0.12))
        elif name == "eyes":
            bu.set_vertex_colors(ob, part["colors"])
            ob.data.materials.append(bu.vc_material("eye", rough=0.08))
        else:
            col, rough = PALETTE[part["mat"]]
            if part["mat"] not in mats:
                mats[part["mat"]] = bu.material(part["mat"], col, rough=rough, sss=0.0)
            ob.data.materials.append(mats[part["mat"]])
    floor = bu.mesh_object("floor", np.array([[-4, -4, 0], [4, -4, 0], [4, 4, 0], [-4, 4, 0]], float), np.array([[0, 1, 2], [0, 2, 3]]), smooth=False)
    floor.data.materials.append(bu.material("floor", (0.32, 0.22, 0.14), rough=0.7))
    bu.setup_render((700, 900), 24)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.42, 0.5, 1)
    bpy.context.scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    H = p["height"]
    sun = bpy.data.lights.new("sun", "SUN"); sun.energy = 3.2; sun.color = (1.0, 0.93, 0.82); sun.angle = 0.1
    so = bpy.data.objects.new("sun", sun); bpy.context.scene.collection.objects.link(so)
    so.rotation_euler = (0.85, 0.25, -0.6)
    cam = bu.camera((1.6 * H, -3.0 * H, H * 0.75), (0, 0, H * 0.5), lens=55)
    if not head_only:
        bu.render(out + "_a.png")
        cam.location = (-2.2 * H, -2.4 * H, H * 0.7); bu.look_at(cam, (0, 0, H * 0.5)); bu.render(out + "_b.png")
    hz = float(J["head"][2]); k = J["_head_h"] / 0.23
    cam.data.lens = 85; cam.location = (0.25 * k, -0.95 * k, hz + 0.03 * k); bu.look_at(cam, (0, -0.03 * k, hz - 0.02 * k)); bu.render(out + "_c.png")
    cam.location = (0.9 * k, -0.35 * k, hz + 0.02 * k); bu.look_at(cam, (0, 0, hz - 0.02 * k)); bu.render(out + "_d.png")
    print("done", round(time.time() - t, 1))

main()
