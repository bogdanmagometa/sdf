import trimesh
import numpy as np
import torch
import pymeshlab as ml

from ddd import read_obj
from utils import temp_stdout_removal

def is_inside(point, obj):
    n_intersections = 0
    face_idx = float('-inf')
    while face_idx != -1:
        next_face_idx = face_idx
        n_tries = 0
        while next_face_idx == face_idx:
            f = obj.ray_cast(point + mathutils.Vector([0.0, 1e-10 * 2**n_tries, 0.0]), mathutils.Vector([0.0, 1.0, 0.0]))
            result, loc, normal, next_face_idx = f
            n_tries += 1
        face_idx = next_face_idx
        point = loc
        
        if face_idx != -1:
            n_intersections += 1
    return n_intersections % 2 == 1

# def check_inside(obj_path, points):
#     with temp_stdout_removal():
#         imported_object = bpy.ops.wm.obj_import(filepath=obj_path, forward_axis='Y', up_axis='Z')
#     obj_object = bpy.context.selected_objects[0]
#     inside = []
#     for point in points:
#         inside.append(is_inside(mathutils.Vector(point), obj_object))
#     return np.array(inside)

def check_inside(obj_path, points):
    mesh = trimesh.load_mesh(obj_path)
    return mesh.contains(points)

def sample_points_bb(obj_path, num_points=10_000):
    vertices, faces = read_obj(obj_path)
    min_point = np.amin(vertices, axis=0)
    max_point = np.amax(vertices, axis=0)
    sampled_points = np.random.rand(num_points, 3) * (max_point - min_point) + min_point

    return sampled_points, check_inside(obj_path, sampled_points)

def sample_points_ns(obj_path, num_points=10_000):
    ms = ml.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.generate_sampling_poisson_disk(samplenum=num_points)

    STD = 0.01    
    sampled_points = ms.current_mesh().vertex_matrix()
    sampled_points += np.random.randn(len(sampled_points), 3) * STD

    return sampled_points, check_inside(obj_path, sampled_points)

def calc_symmetric_f1(pred, gt):
    return (calc_f1(pred, gt) + calc_f1(~pred, ~gt)) / 2

def calc_f1(pred, gt):
    tp = torch.sum((pred == gt) & (gt == True))
    tn = torch.sum((pred == gt) & (gt == False))
    fp = torch.sum((pred != gt) & (gt == True))
    fn = torch.sum((pred != gt) & (gt == False))
    precision = (tp) / (tp + fp)
    recall = (tp) / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def calc_bb_f1(sdf, obj_path, num_points=10_000):
    """Calculate bounding box f1 score"""
    points, is_inside = sample_points_bb(obj_path, num_points)
    f1 = calc_f1(sdf, points, is_inside)
    return f1

def calc_ns_f1(sdf, obj_path, num_points=10_000):
    """Calculate near surface f1 score"""
    points, is_inside = sample_points_ns(obj_path, num_points)
    f1 = calc_f1(sdf, points, is_inside)
    return f1
