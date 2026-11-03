"""Render a lineup of NPC variants generated from the same base by parameter edits."""
import sys, time
import numpy as np
import races, humanoid as hu, body, mesher, anatomy
import bpy
import blendutil as bu

VARIANTS = [
    ("human_m", dict()),
    ("human_m", dict(height=1.70, muscle=0.35, fat=0.55, jaw=1.1, nose_len=1.1, brow=1.1)),
    ("human_m", dict(height=1.88, muscle=0.85, fat=0.15, shoulder=1.06, jaw=1.15, chin=1.1)),
    ("human_f", dict()),
    ("human_f", dict(height=1.62, fat=0.42, muscle=0.3, hip=1.06, cheek=1.1, nose_w=0.9)),
    ("human_f", dict(height=1.76, muscle=0.6, fat=0.12, shoulder=1.04, jaw=1.0)),
]
out = sys.argv[1]
bu.reset()
clay = bu.material("clay", (0.55, 0.45, 0.4), rough=0.5, sss=0.05)
eye = bu.material("eye", (0.9, 0.9, 0.88), rough=0.15)
x = 0.0
for i, (pre, kw) in enumerate(VARIANTS):
    t = time.time()
    p = races.preset(pre)
    p.update(kw)
    J = hu.build_skeleton(p)
    s = body.build_body(p, J)
    v, f = mesher.mesh_shape(s, h=0.0055, verbose=False)
    ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0015, verbose=False)
    for nm, vv, ff, mat in (("b", v, f, clay), ("e", ev, ef, eye)):
        ob = bu.mesh_object(f"{nm}{i}", vv + np.array([x, 0, 0], np.float32), ff)
        ob.data.materials.append(mat)
    print(i, pre, round(time.time() - t, 1))
    x += 0.62
bu.setup_render((1800, 700), 14)
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.03, 0.03, 0.035, 1)
cx = x / 2 - 0.31
bu.add_light("AREA", (cx - 2.0, -3.5, 3.2), 260, (1, 0.97, 0.94), 3.0, (cx, 0, 0.9))
bu.add_light("AREA", (cx + 3.0, 1.5, 2.2), 90, (0.8, 0.85, 1.0), 3.0, (cx, 0, 0.9))
bu.camera((cx + 0.6, -6.5, 1.2), (cx, 0, 0.88), ortho=4.1)
bu.render(out)
print("saved", out)
