"""Side-by-side + outline overlay against the user's reference sheet (same pose, same scale).
usage: python3 compare.py <out.png> <preset> <ref_image> <ref_crop x0,y0,x1,y1> [arm_drop_deg]"""
import sys, time
import numpy as np
from PIL import Image, ImageFilter, ImageChops
import humanoid as hu
import races, body, anatomy, mesher
import bpy
import blendutil as bu

out, preset, ref_path, crop = sys.argv[1], sys.argv[2], sys.argv[3], [int(v) for v in sys.argv[4].split(",")]
hu.ARM_DROP_DEG = float(sys.argv[5]) if len(sys.argv) > 5 else 70.0
p = races.preset(preset)
J = hu.build_skeleton(p)
s = body.build_body(p, J)
t = time.time()
v, f = mesher.mesh_shape(s, h=0.0045)
ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0015)
print("meshed", round(time.time() - t, 1))
bu.reset()
ob = bu.mesh_object("b", v, f); ob.data.materials.append(bu.material("clay", (0.5, 0.5, 0.5), rough=0.5))
eo = bu.mesh_object("e", ev, ef); eo.data.materials.append(bu.material("eye", (0.8, 0.8, 0.8), rough=0.3))
bu.setup_render((600, 1000), 12)
bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.02, 0.02, 0.02, 1)
H = p["height"]
bu.add_light("AREA", (-2.6, -2.2, H * 2.0), 90 * H, (1, 1, 1), 1.2, (0, 0, H * 0.6))
bu.add_light("AREA", (2.0, -1.5, H * 0.9), 12 * H, (1, 1, 1), 2.0, (0, 0, H * 0.5))
bu.add_light("AREA", (0.3, 3, H * 1.4), 45 * H, (1, 1, 1), 2.0, (0, 0, H * 0.7))
cam = bu.camera((0, -6 * H, H * 0.5), (0, 0, H * 0.5), ortho=H * 1.02)
bu.render("/tmp/claude-0/prev/_cmp_front.png")
cam.location = (6 * H, 0, H * 0.5); bu.look_at(cam, (0, 0, H * 0.5)); bu.render("/tmp/claude-0/prev/_cmp_side.png")
cam.location = (0, 6 * H, H * 0.5); bu.look_at(cam, (0, 0, H * 0.5)); bu.render("/tmp/claude-0/prev/_cmp_back.png")

def fit(im):
    a = np.asarray(im.convert("L")).astype(float)
    ys, xs = np.where(a > 22)
    return im.crop((max(xs.min() - 30, 0), ys.min(), min(xs.max() + 30, im.width), ys.max() + 1))

mine = fit(Image.open("/tmp/claude-0/prev/_cmp_front.png").convert("RGB"))
side = fit(Image.open("/tmp/claude-0/prev/_cmp_side.png").convert("RGB"))
back = fit(Image.open("/tmp/claude-0/prev/_cmp_back.png").convert("RGB"))
ref = Image.open(ref_path).convert("RGB").crop(crop)
ref = fit(ref)
Ht = 900
def sc(im): return im.resize((int(im.width * Ht / im.height), Ht))
mine, ref, side, back = sc(mine), sc(ref), sc(side), sc(back)
# outline overlay: reference edges (red) over my render, centred
W = max(mine.width, ref.width)
canvas = Image.new("RGB", (W, Ht)); canvas.paste(mine, ((W - mine.width) // 2, 0))
rm = Image.new("L", (W, Ht)); rl = ref.convert("L").point(lambda v: 255 if v > 30 else 0); rm.paste(rl, ((W - ref.width) // 2, 0))
edge = rm.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.MaxFilter(3))
red = Image.new("RGB", (W, Ht), (255, 40, 40))
overlay = Image.composite(red, canvas, edge)
sheet = Image.new("RGB", (ref.width + mine.width + W + side.width + back.width, Ht))
x = 0
for im in (ref, mine, overlay, side, back):
    sheet.paste(im, (x, 0)); x += im.width
sheet.save(out)
# upper-body close-up next to the reference (same scale)
def upper(im): return im.crop((0, 0, im.width, int(im.height * 0.5)))
ru, mu = upper(ref), upper(mine)
up_sheet = Image.new("RGB", (ru.width + mu.width + upper(back).width + upper(side).width, ru.height))
x = 0
for im in (ru, mu, upper(side), upper(back)):
    up_sheet.paste(im, (x, 0)); x += im.width
up_sheet.save(out.replace(".png", "_upper.png"))
def lower(im): return im.crop((0, int(im.height * 0.42), im.width, int(im.height * 0.85)))
lo_sheet = Image.new("RGB", (lower(ref).width + lower(mine).width + lower(side).width + lower(back).width, lower(ref).height))
x = 0
for im in (lower(ref), lower(mine), lower(side), lower(back)):
    lo_sheet.paste(im, (x, 0)); x += im.width
lo_sheet.save(out.replace(".png", "_lower.png"))
print("saved", out)
