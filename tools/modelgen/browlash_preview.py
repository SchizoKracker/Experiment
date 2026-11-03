"""Fast eye-region preview: cached head mesh + freshly grown brows and lashes only (no scalp hair).
usage: python3 browlash_preview.py <out.png> [preset=human_f] [samples=24]
Needs the mesh cache written by headshot.py (/tmp/claude-0/prev/hs_cache_<preset>.npz)."""
import sys, time
import numpy as np
from PIL import Image
import races, humanoid as hu, body, anatomy, grooms
import hair as hr
from sdf import F
import bpy
import blendutil as bu
from hair_preview import hair_material, add_strands

out = sys.argv[1]
preset = sys.argv[2] if len(sys.argv) > 2 else "human_f"
samples = int(sys.argv[3]) if len(sys.argv) > 3 else 24
t0 = time.time()
hu.ARM_DROP_DEG = 72
p = races.preset(preset)
J = hu.build_skeleton(p)
s = body.build_body(p, J)                      # also fills face.LID_ARCS
hf = hr.HeadFrame(J)
hh = hf.hh
C = np.load(f"/tmp/claude-0/prev/hs_cache_{preset}.npz")
v, f, ev, ef = C["v"], C["f"], C["ev"], C["ef"]
# keep only the upper face for speed
u, fw, zn = hf.local(v)
keepv = (zn > 0.3) & (zn < 0.75) & (fw > 0.0)
fk = f[np.all(keepv[f], axis=1)]
idx = np.unique(fk)
remap = -np.ones(len(v), int); remap[idx] = np.arange(len(idx))
v, f = v[idx], remap[fk]

blo = np.array([-0.08, hf.yc - 0.16, hf.top - 0.62 * hh], F)
bhi = np.array([0.08, hf.yc - 0.02, hf.top - 0.36 * hh], F)
bfield = hr.Field(s, blo, bhi, h=0.0012)
br = grooms.brows(s, J, bfield, density=0.9)
la = grooms.lashes(J, p, shape=s)
print("grown", round(time.time() - t0, 1), br["strands"].shape, la["strands"].shape)

bu.reset()
ob = bu.mesh_object("head", v, f)
ob.data.materials.append(bu.material("skin", (0.62, 0.42, 0.33), rough=0.45, sss=0.2))
for poly in ob.data.polygons:
    poly.use_smooth = True
eo = bu.mesh_object("eyes", ev, ef)
eo.data.materials.append(bu.material("eye", (0.8, 0.8, 0.78), rough=0.05))
add_strands("brows", br["strands"], br["r_root"], br["r_tip"], hair_material("brow", melanin=0.82, red=0.65, rough=0.35))
add_strands("lashes", la["strands"], la["r_root"], la["r_tip"], hair_material("lash", melanin=0.98, red=0.2, rough=0.3))

W, H = 900, 520
bu.setup_render((W, H), samples)
sc = bpy.context.scene
sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.3, 0.3, 0.3, 1)
sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
ec = np.array([0, hf.yc - 0.3 * hh, hf.top - 0.49 * hh])
bu.add_light("AREA", tuple(ec + np.array([-0.4, -0.6, 0.4])), 25, (1, 0.95, 0.9), 0.6, tuple(ec))
bu.add_light("AREA", tuple(ec + np.array([0.5, -0.4, 0.1])), 6, (0.9, 0.95, 1), 0.8, tuple(ec))
cam = bu.camera(tuple(ec), tuple(ec), lens=85)
ims = []
VIEWS = (("front", (0.0, -0.42, 0.0)), ("34", (0.26, -0.34, 0.02)), ("side", (0.42, -0.06, 0.0)))
if "--eye" in sys.argv:
    # close-ups of the left eye (viewer's right): front, 3/4, low-side
    ec = anatomy.eye_centers(p, J)[1]
    VIEWS = (("efront", (0.0, -0.16, 0.0)), ("e34", (0.11, -0.12, 0.01)), ("eside", (0.15, -0.035, 0.0)))
for name, d in VIEWS:
    cam.location = tuple(ec + np.array(d)); bu.look_at(cam, tuple(ec))
    pth = f"/tmp/claude-0/prev/_bl_{name}.png"; bu.render(pth); ims.append(Image.open(pth).convert("RGB"))
sheet = Image.new("RGB", (W, H * len(ims)))
for i, im in enumerate(ims):
    sheet.paste(im, (0, H * i))
sheet.save(out)
print("saved", out, round(time.time() - t0, 1))
