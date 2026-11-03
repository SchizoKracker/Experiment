"""Close-up joint deformation checks on a rigged .blend (from rig.py).
usage: python3 posecheck.py <rig.blend> <out.png>"""
import sys
import bpy
from PIL import Image
import blendutil as bu
from rig import POSES, apply_pose

blend, out = sys.argv[1], sys.argv[2]
bpy.ops.wm.open_mainfile(filepath=blend)
rig = bpy.data.objects["rig"]
cam = bpy.context.scene.camera
cam.data.lens = 85
bpy.context.scene.render.resolution_x = 480
bpy.context.scene.render.resolution_y = 480

def bone_world(name, tail=False):
    pb = rig.pose.bones[name]
    return rig.matrix_world @ (pb.tail if tail else pb.head)

shots = [
    ("elbows_knees", "forearm.L", (0.0, -1.3, 0.25)),
    ("elbows_knees", "shin.L", (0.4, -1.3, 0.2)),
    ("arms_up", "upper_arm.L", (0.5, -1.2, 0.2)),
    ("squat", "thigh.L", (0.9, -1.1, 0.3)),
    ("squat", "hips", (0.0, 1.4, 0.2)),
    ("stride", "hand.L", (0.2, -0.9, 0.1)),
]
paths = []
for i, (pose, bone, off) in enumerate(shots):
    apply_pose(rig, POSES[pose])
    target = bone_world(bone)
    cam.location = target + __import__("mathutils").Vector(off)
    bu.look_at(cam, tuple(target))
    pth = f"/tmp/claude-0/prev/_pc_{i}.png"
    bu.render(pth)
    paths.append(pth)
ims = [Image.open(p).convert("RGB") for p in paths]
sheet = Image.new("RGB", (480 * 3, 960))
for i, im in enumerate(ims):
    sheet.paste(im, ((i % 3) * 480, (i // 3) * 480))
sheet.save(out)
print("saved", out)
