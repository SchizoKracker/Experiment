"""Clay render (front/back/side/face) to compare against reference sheets.
usage: python3 clay.py <out_prefix> <preset> [h]"""
import sys, time
import numpy as np
import races, humanoid as hu, anatomy, mesher
import bpy
import blendutil as bu
out, preset = sys.argv[1], sys.argv[2]
h = float(sys.argv[3]) if len(sys.argv) > 3 else 0.004
t = time.time()
p = races.preset(preset)
J = hu.build_skeleton(p)
body = anatomy.build_body(p, J)
v, f = mesher.mesh_shape(body, h=h)
ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0012)
print("meshed", round(time.time() - t, 1), len(f))
bu.reset()
ob = bu.mesh_object("body", v, f); ob.data.materials.append(bu.material("clay", (0.55, 0.55, 0.55), rough=0.55))
eo = bu.mesh_object("eyes", ev, ef); eo.data.materials.append(bu.material("eye", (0.9, 0.9, 0.9), rough=0.2))
bu.setup_render((700, 1000), 16)
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.03, 0.03, 0.03, 1)
H = p["height"]
bu.add_light("AREA", (-1.5, -3.0, H * 1.5), 90 * H, (1, 1, 1), 2.5, (0, 0, H * 0.6))
bu.add_light("AREA", (2.5, -1.0, H * 1.0), 35 * H, (1, 1, 1), 2.0, (0, 0, H * 0.6))
bu.add_light("AREA", (0.5, 3, H * 1.4), 60 * H, (1, 1, 1), 2.0, (0, 0, H * 0.7))
cam = bu.camera((0, -5 * H, H * 0.5), (0, 0, H * 0.5), ortho=H * 1.08)
bu.render(out + "_front.png")
cam.location = (0, 5 * H, H * 0.5); bu.look_at(cam, (0, 0, H * 0.5)); bu.render(out + "_back.png")
cam.location = (5 * H, 0, H * 0.5); bu.look_at(cam, (0, 0, H * 0.5)); bu.render(out + "_side.png")
hz = J["head_top"][2] - 0.5 * J["_head_h"]; hh = J["_head_h"]
bu.bpy.context.scene.render.resolution_x = 900; bpy.context.scene.render.resolution_y = 700
cam.data.ortho_scale = hh * 2.2
cam.location = (0, -5 * H, hz); bu.look_at(cam, (0, 0, hz)); bu.render(out + "_hfront.png")
cam.location = (-5 * H, 0, hz); bu.look_at(cam, (0, 0, hz)); bu.render(out + "_hside.png")
print("done", round(time.time() - t, 1))
