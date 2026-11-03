"""Render front/side/face previews of a humanoid parameter set.

usage: python3 preview.py <out_prefix> [preset]
"""
import sys
import time

import numpy as np

import humanoid as hu
import mesher
import races

import bpy  # noqa: E402  (needs PYTHONPATH to the bpy wheel)
import blendutil as bu


def main():
    out = sys.argv[1]
    preset = sys.argv[2] if len(sys.argv) > 2 else "human_m"
    h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0035
    p = races.preset(preset)
    t = time.time()
    J = hu.build_skeleton(p)
    body = hu.build_body(p, J)
    v, f = mesher.mesh_shape(body, h=h)
    ev, ef = mesher.mesh_shape(hu.eye_shape(p, J), h=0.0015)
    print("meshed", time.time() - t)
    bu.reset()
    skin = races.skin_color(p)
    ob = bu.mesh_object("body", v, f)
    ob.data.materials.append(bu.material("skin", skin, rough=0.45, sss=0.15))
    eo = bu.mesh_object("eyes", ev, ef)
    eo.data.materials.append(bu.material("eye", (0.85, 0.82, 0.78), rough=0.1))
    floor = bu.mesh_object("floor", np.array([[-3, -3, 0], [3, -3, 0], [3, 3, 0], [-3, 3, 0]], float), np.array([[0, 1, 2], [0, 2, 3]]), smooth=False)
    floor.data.materials.append(bu.material("floor", (0.15, 0.13, 0.12), rough=0.8))
    bu.setup_render((900, 900), 20)
    H = p["height"]
    bu.add_light("AREA", (-2.0, -3.0, H * 1.6), 380 * H, (1.0, 0.9, 0.8), 2.0, (0, 0, H * 0.6))
    bu.add_light("AREA", (2.5, 1.0, H * 1.2), 160 * H, (0.6, 0.7, 1.0), 2.0, (0, 0, H * 0.6))
    bu.add_light("AREA", (0.5, 3, H * 1.5), 260 * H, (1.0, 0.8, 0.6), 1.5, (0, 0, H * 0.8))
    cam = bu.camera((0, -4.2 * H, H * 0.55), (0, 0, H * 0.5), ortho=H * 1.12)
    bu.render(out + "_front.png")
    cam.location = (4.2 * H, -0.3, H * 0.55)
    bu.look_at(cam, (0, 0, H * 0.5))
    bu.render(out + "_side.png")
    hz = float(J["head"][2])
    cam.data.type = "PERSP"
    cam.data.lens = 85
    cam.location = (0.35 * H * 0.5, -1.2 * J["_head_h"] / 0.23 * 1.0, hz + 0.02)
    bu.look_at(cam, (0, float(J["head"][1]) - 0.03, hz - 0.01))
    bu.render(out + "_face.png")
    print("done", time.time() - t)


main()
