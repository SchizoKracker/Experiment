"""Head close-ups with strand grooms (Cycles hair curves)."""
import sys, time
import numpy as np
from PIL import Image
import races, humanoid as hu, body, mesher, anatomy, grooms, hair as hr
from sdf import Shape, F
import bpy
import blendutil as bu

SKIN = {"m": (0.62, 0.42, 0.33), "f": (0.7, 0.5, 0.4)}
HAIR = {"m": dict(melanin=0.85, red=0.35, rough=0.32), "f": dict(melanin=0.55, red=0.9, rough=0.3)}


def hair_material(name, melanin, red, rough):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    out = nt.nodes["Material Output"]
    for n in list(nt.nodes):
        if n != out:
            nt.nodes.remove(n)
    hb = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    hb.parametrization = "MELANIN"
    hb.inputs["Melanin"].default_value = melanin
    hb.inputs["Melanin Redness"].default_value = red
    hb.inputs["Roughness"].default_value = rough
    hb.inputs["Radial Roughness"].default_value = 0.35
    hb.inputs["Random Color"].default_value = 0.12
    nt.links.new(hb.outputs[0], out.inputs["Surface"])
    return m


def add_strands(name, S, r_root, r_tip, mat):
    N, M, _ = S.shape
    c = bpy.data.hair_curves.new(name)
    c.add_curves([M] * N)
    c.attributes["position"].data.foreach_set("vector", S.reshape(-1).astype(np.float32))
    rad = c.attributes.new("radius", "FLOAT", "POINT")
    r = np.tile(np.linspace(r_root, r_tip, M), N).astype(np.float32)
    rad.data.foreach_set("value", r)
    ob = bpy.data.objects.new(name, c)
    bpy.context.scene.collection.objects.link(ob)
    c.materials.append(mat)
    return ob


def build_grooms(sex, s, J, p):
    hf = hr.HeadFrame(J)
    base = Shape(prims=s.prims)
    lo = np.array([-0.26, hf.yc - 0.22, hf.top - (0.9 if sex == "f" else 0.42)], F)
    hi = np.array([0.26, hf.yc + 0.32, hf.top + 0.04], F)
    field = hr.Field(base, lo, hi, h=0.0035)
    gs = []
    if sex == "m":
        gs.append(grooms.male_swept_back(base, J, field))
        gs.append(grooms.male_beard(base, J, field))
    else:
        gs.extend(grooms.female_braid(base, J, field))
    gs.append(grooms.brows(base, J, field, density=1.0 if sex == "m" else 0.8))
    gs.append(grooms.lashes(J, p, anatomy.eye_centers(p, J), length=0.0075 if sex == "m" else 0.0095))
    return gs


def main():
    out = sys.argv[1]
    sex = sys.argv[2]
    t = time.time()
    hu.ARM_DROP_DEG = 72
    p = races.preset("human_" + sex)
    J = hu.build_skeleton(p)
    s = body.build_body(p, J)
    hf = hr.HeadFrame(J)
    clip_lo = np.array([-0.3, hf.yc - 0.3, hf.top - (1.0 if sex == "f" else 0.5)], F)
    clip_hi = np.array([0.3, hf.yc + 0.35, hf.top + 0.05], F)
    v, f = mesher.mesh_shape(s, h=0.0025, clip=(clip_lo, clip_hi), verbose=False)
    ev, ef = mesher.mesh_shape(anatomy.eye_shape(p, J), h=0.0008, verbose=False)
    print("meshed", round(time.time() - t, 1))
    gs = build_grooms(sex, s, J, p)
    print("groomed", round(time.time() - t, 1), {g["name"]: g["strands"].shape for g in gs})
    bu.reset()
    skin = bu.material("skin", SKIN[sex], rough=0.45, sss=0.25)
    ob = bu.mesh_object("head", v, f); ob.data.materials.append(skin)
    import character
    eo = bu.mesh_object("eyes", ev, ef)
    cols = character.eye_colors(p, J, ev, iris=(0.18, 0.32, 0.22) if sex == "f" else (0.25, 0.18, 0.1))
    bu.set_vertex_colors(eo, cols)
    eo.data.materials.append(bu.vc_material("eye", rough=0.05))
    hm = hair_material("hair", **HAIR[sex])
    for g in gs:
        add_strands(g["name"], g["strands"], g["r_root"], g["r_tip"], hm)
    bu.setup_render((900, 900), 48)
    sc = bpy.context.scene
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.05, 0.06, 1)
    hc = np.array([0, hf.yc, hf.top - 0.5 * hf.hh])
    bu.add_light("AREA", tuple(hc + np.array([-0.8, -1.0, 0.9])), 60, (1.0, 0.92, 0.82), 0.8, tuple(hc))
    bu.add_light("AREA", tuple(hc + np.array([0.9, 0.6, 0.5])), 45, (0.7, 0.8, 1.0), 0.6, tuple(hc))
    bu.add_light("AREA", tuple(hc + np.array([0.0, 1.0, 0.8])), 40, (1, 1, 1), 0.8, tuple(hc))
    cam = bu.camera(tuple(hc + np.array([0.35, -0.75, 0.05])), tuple(hc), lens=70)
    paths = []
    views = [(0.35, -0.75, 0.05), (-0.75, -0.2, 0.05), (0.2, 0.8, 0.12)] if sex == "m" else \
            [(0.35, -0.75, 0.05), (-0.75, -0.2, 0.05), (0.25, 1.05, -0.15)]
    for i, vv in enumerate(views):
        tgt = hc if i < 2 or sex == "m" else hc + np.array([0, 0.03, -0.14])
        cam.location = tuple(tgt + np.array(vv)); bu.look_at(cam, tuple(tgt))
        pth = f"/tmp/claude-0/prev/_hair_{sex}_{i}.png"; bu.render(pth); paths.append(pth)
    ims = [Image.open(x).convert("RGB") for x in paths]
    sheet = Image.new("RGB", (900 * len(ims), 900))
    for i, im in enumerate(ims):
        sheet.paste(im, (900 * i, 0))
    sheet.save(out)
    print("saved", out, round(time.time() - t, 1))


if __name__ == "__main__":
    main()
