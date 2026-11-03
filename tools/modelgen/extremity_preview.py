"""Close-up clay renders of the left hand and left foot next to the user's references."""
import sys, time
import numpy as np
from PIL import Image
import races, humanoid as hu, body, mesher
import bpy
import blendutil as bu

out = sys.argv[1]
p = races.preset(sys.argv[2] if len(sys.argv) > 2 else "human_m")
J = hu.build_skeleton(p)
s = body.build_body(p, J)
H = p["height"]
t = time.time()
wr, fwd, n, ac = hu.hand_frame(J, "L")
hc = wr + fwd * 0.09
hv, ht = mesher.mesh_shape(s, h=0.0014, clip=(hc - 0.15, hc + 0.15), verbose=False)
an = J["foot.L"]
fc = an + np.array([0, -0.06, -0.03], np.float32)
fv, ft = mesher.mesh_shape(s, h=0.0016, clip=(fc - np.array([0.12, 0.18, 0.06], np.float32), fc + np.array([0.12, 0.18, 0.14], np.float32)), verbose=False)
print("meshed", round(time.time() - t, 1))
bu.reset()
clay = bu.material("clay", (0.55, 0.42, 0.34), rough=0.5, sss=0.1)
for nm, v, f in (("hand", hv, ht), ("foot", fv, ft)):
    ob = bu.mesh_object(nm, v, f); ob.data.materials.append(clay)
bu.setup_render((640, 640), 16)
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.06, 0.06, 1)
bu.add_light("AREA", tuple(hc + np.array([-0.5, -0.6, 0.8])), 40, (1, 0.97, 0.93), 0.6, tuple(hc))
bu.add_light("AREA", tuple(fc + np.array([-0.6, -0.8, 0.9])), 40, (1, 0.97, 0.93), 0.6, tuple(fc))
bu.add_light("AREA", (0, 2, 2), 120, (0.8, 0.85, 1.0), 2.0, (0, 0, 0.5))
shots = []
def shot(name, cam_pos, target, ortho):
    cam = bpy.context.scene.camera
    cam.location = tuple(cam_pos); bu.look_at(cam, tuple(target)); cam.data.ortho_scale = ortho
    path = f"/tmp/claude-0/prev/_ext_{name}.png"; bu.render(path); shots.append(path)
bu.camera(tuple(hc - n * 1.0), tuple(hc), ortho=0.24)
shot("hand_back", hc - n * 1.0, hc, 0.24)
shot("hand_palm", hc + n * 1.0, hc, 0.24)
shot("hand_side", hc + ac * 1.0, hc, 0.24)
shot("foot_med", fc + np.array([-1.0, 0, 0.05]), fc, 0.3)
shot("foot_front", fc + np.array([-0.5, -0.8, 0.45]), fc, 0.3)
shot("foot_lat", fc + np.array([1.0, -0.1, 0.1]), fc, 0.3)
ims = [Image.open(pth).convert("RGB").resize((320, 320)) for pth in shots]
sheet = Image.new("RGB", (320 * 3, 640))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % 3) * 320, (i // 3) * 320))
sheet.save(out)
print("saved", out)
