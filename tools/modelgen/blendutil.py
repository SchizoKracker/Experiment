"""Blender helpers shared by the preview and export scripts."""
from __future__ import annotations

import math

import bpy
import numpy as np


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def mesh_object(name, verts, tris, smooth=True, collection=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.astype(float).tolist(), [], tris.astype(int).tolist())
    me.validate()
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def material(name, color, rough=0.5, metal=0.0, sss=0.0, emission=None, emission_strength=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if sss > 0:
        b.inputs["Subsurface Weight"].default_value = sss
        b.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.2)
        b.inputs["Subsurface Scale"].default_value = 0.01
    if emission is not None:
        b.inputs["Emission Color"].default_value = (*emission, 1.0)
        b.inputs["Emission Strength"].default_value = emission_strength
    return m


def look_at(ob, target):
    from mathutils import Vector
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def setup_render(res=(640, 800), samples=24):
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"
    sc.cycles.device = "CPU"
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    world = bpy.data.worlds.new("w")
    sc.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.05, 0.045, 0.04, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 1.0


def add_light(kind, loc, energy, color=(1, 1, 1), size=1.0, target=(0, 0, 1)):
    ld = bpy.data.lights.new(kind + "L", kind)
    ld.energy = energy
    ld.color = color
    if hasattr(ld, "size"):
        ld.size = size
    ob = bpy.data.objects.new(kind, ld)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    return ob


def camera(loc, target, lens=50, ortho=None):
    cd = bpy.data.cameras.new("cam")
    cd.lens = lens
    if ortho:
        cd.type = "ORTHO"
        cd.ortho_scale = ortho
    ob = bpy.data.objects.new("cam", cd)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    bpy.context.scene.camera = ob
    return ob


def render(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def set_vertex_colors(ob, cols):
    me = ob.data
    attr = me.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    rgba = np.concatenate([cols, np.ones((len(cols), 1))], axis=1).astype(np.float32)
    attr.data.foreach_set("color", rgba.ravel())
    return attr


def vc_material(name, rough=0.5, sss=0.0, mult=None):
    """Principled BSDF fed by the 'Col' color attribute."""
    m = material(name, (1, 1, 1), rough=rough, sss=sss)
    nt = m.node_tree
    a = nt.nodes.new("ShaderNodeVertexColor")
    a.layer_name = "Col"
    b = nt.nodes.get("Principled BSDF")
    nt.links.new(a.outputs["Color"], b.inputs["Base Color"])
    return m
