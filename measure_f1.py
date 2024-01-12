import argparse
import os

import torch
import numpy as np
import pandas as pd

def calc_symmetric_f1(pred, gt):
    return (calc_f1(pred, gt) + calc_f1(~pred, ~gt)) / 2

def calc_f1(pred, gt):
    eps = 1e-8
    tp = torch.sum((pred == gt) & (gt == True))
    tn = torch.sum((pred == gt) & (gt == False))
    fp = torch.sum((pred != gt) & (gt == True))
    fn = torch.sum((pred != gt) & (gt == False))
    precision = (tp) / (tp + fp + eps)
    recall = (tp) / (tp + fn + eps)
    return 2 * precision * recall / (precision + recall + eps)

def load_nglod(weights_path, obj_path):
    from train_nglod import build_parser
    from lib.models import OctreeSDF
    parser = build_parser()
    args = parser.parse_args([
        '--net', 'OctreeSDF', 
        '--num-lods', '3',
        '--feature-dim', '24',
        # '--num-lods', '4',
        # '--feature-dim', '32',
        ])
    model = OctreeSDF(args)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.cuda()
    model = {'model': model, 'obj_path': obj_path}
    return model

def load_siren(weights_path):
    from siren import Siren
    model = Siren(3, 256, 3, 1)
    weights = torch.load(weights_path, map_location='cpu')
    centroid, coord_min, coord_max = weights['centroid'], weights['coord_min'], weights['coord_max']
    centroid, coord_min, coord_max = np.array(centroid), np.array(coord_min), np.array(coord_max)
    weights = weights['weights']
    model.load_state_dict(weights)
    model.cuda()
    model = {'centroid': centroid, 'coord_min': coord_min, 'coord_max': coord_max, 'model': model}
    return model


def siren_preprocess(points: np.ndarray, centroid, coord_min, coord_max):
    points = points - centroid
    points = (points - coord_min) / (coord_max - coord_min)
    points -= 0.5
    points *= 2.0
    points = torch.tensor(points, device='cuda', dtype=torch.float32)
    return points

def siren_inference(model, points: torch.Tensor):
    points = siren_preprocess(points, model['centroid'], model['coord_min'], model['coord_max'])
    pred_dist = model['model']({'coords': points})['model_out']
    return pred_dist

def nglod_preprocess(points: np.ndarray, obj_path) -> torch.Tensor:
    from lib.torchgp import load_obj
    points = torch.tensor(points, dtype=torch.float32)

    V, _ = load_obj(obj_path)
    
    # Normalize mesh
    V_max, _ = torch.max(V, dim=0)
    V_min, _ = torch.min(V, dim=0)
    V_center = (V_max + V_min) / 2.
    V = V - V_center
    points = points - V_center

    # Find the max distance to origin
    max_dist = torch.sqrt(torch.max(torch.sum(V**2, dim=-1)))
    V_scale = 1. / max_dist
    points *= V_scale
    return points.cuda()

def nglod_inference(model, points: torch.Tensor) -> torch.Tensor:
    points = nglod_preprocess(points, model['obj_path'])
    return model['model'].sdf(points)

def load_points(points_path):
    points = []
    inside = []
    with open(points_path) as points_f:
        for line in points_f.readlines():
            if not line.startswith('f'):
                continue
            *coords, w = [float(c) for c in line.split()[1:]]
            points.append(coords)
            inside.append(bool(w))
    return np.array(points), np.array(inside)


def old_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='siren')
    # parser.add_argument('--weights', type=str, default='siren_weights/0.pt')
    parser.add_argument('--weights', type=str, default='./_results_old/models/0.pth')
    parser.add_argument('--obj_path', type=str, default='test_task_meshes/0.obj')
    parser.add_argument('--bounding', type=str, default=None)
    parser.add_argument('--surface', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.model == "siren":
        model = load_siren(args.weights)
    elif args.model == "nglod":
        model = load_nglod(args.weights, args.obj_path)
    else:
        raise ValueError("Invalid model type")

    if args.bounding is not None:
        points_b, are_inside_b = load_points(args.bounding)
    else:
        from metrics import sample_points_bb
        points_b, are_inside_b = sample_points_bb(args.obj_path, 3_000)
 
    if args.surface is not None:
        points_s, are_inside_s = load_points(args.surface)
    else:
        from metrics import sample_points_ns
        points_s, are_inside_s = sample_points_ns(args.obj_path, 3_000)

    metrics = {}
    for metric_name, points, are_inside in [
        ('bounding F1', points_b, are_inside_b),
        ('surface F1', points_s, are_inside_s)
        ]:

        if args.model == 'siren':
            pred_dist = siren_inference(model, points)
        else:
            pred_dist = nglod_inference(model, points)

        f1 = calc_symmetric_f1(pred_dist[:, 0] < 0, torch.tensor(are_inside, dtype=torch.bool, device='cuda'))
        metrics[metric_name] = f1
        print(f"{metric_name} = {f1:.2f}")

    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='siren', choices=['siren', 'nglod'])
    parser.add_argument('--weights_dir', type=str, default='siren_weights')
    # parser.add_argument('--weights_dir', type=str, default='./_results_nn/models/')
    parser.add_argument('--obj_dir', type=str, default='test_task_meshes/')
    parser.add_argument('--bounding_dir', type=str, default='points/bounding')
    parser.add_argument('--surface_dir', type=str, default='points/surface')
    parser.add_argument('--results_path', type=str, required=True)
    # parser.add_argument('--use-mesh2sdf', action='store_true')
    
    args = parser.parse_args()
    
    f1_scores = {
        'mesh': [],
        'bounding': [],
        'surface': []
    }
    
    for mesh in range(50):
        f1_scores['mesh'].append(mesh)
        weights_path = os.path.join(args.weights_dir, f"{mesh}.pth")
        if not os.path.exists(weights_path):
            weights_path = os.path.join(args.weights_dir, f"{mesh}.pt")
        obj_path = os.path.join(args.obj_dir, f"{mesh}.obj")
        bounding_path = os.path.join(args.bounding_dir, f"{mesh}.obj")
        surface_path = os.path.join(args.surface_dir, f"{mesh}.obj")

        if args.model == "siren":
            model = load_siren(weights_path)
        elif args.model == "nglod":
            model = load_nglod(weights_path, obj_path)

        points_b, are_inside_b = load_points(bounding_path)
        points_s, are_inside_s = load_points(surface_path)
        # from metrics import sample_points_bb, sample_points_ns
        # points_b, are_inside_b = sample_points_bb(obj_path, 3000)
        # points_s, are_inside_s = sample_points_ns(obj_path, 3000)

        if args.model == "siren":
            pred_dist_b = siren_inference(model, points_b)
            pred_dist_s = siren_inference(model, points_s)
        elif args.model == "nglod":
            pred_dist_b = nglod_inference(model, points_b)
            pred_dist_s = nglod_inference(model, points_s)

        f1_b = calc_symmetric_f1(pred_dist_b[:, 0] < 0, torch.tensor(are_inside_b, dtype=torch.bool, device='cuda'))
        f1_s = calc_symmetric_f1(pred_dist_s[:, 0] < 0, torch.tensor(are_inside_s, dtype=torch.bool, device='cuda'))

        f1_scores['bounding'].append(f1_b.item())
        f1_scores['surface'].append(f1_s.item())

    f1_scores = pd.DataFrame(f1_scores)
    f1_scores.round(2).to_csv(args.results_path, index=False)
    agg_f1 = f1_scores[['bounding', 'surface']].mean()

    print(f1_scores.round(3))
    print(agg_f1.round(3))
