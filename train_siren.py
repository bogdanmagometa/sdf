import argparse
from collections import OrderedDict

import numpy as np
from torch.utils.data import DataLoader
from torch import nn
import torch
import wandb

from metrics import calc_symmetric_f1, sample_points_bb, sample_points_ns
from siren import sdf, PointCloud, Siren
from ddd import process_obj, read_obj


def eval_step(model, train_dataloader, obj_path):
    model.eval()
    metrics = {}
    with torch.no_grad():
        coord_max = train_dataloader.dataset.coord_max
        coord_min = train_dataloader.dataset.coord_min
        centroid = train_dataloader.dataset.centroid

        for metric_name, sample_fn in [('bounding F1', sample_points_bb), ('surface F1', sample_points_ns)]:
            points, are_inside = sample_fn(obj_path, 10_000)
            points -= centroid
            points = (points - coord_min) / (coord_max - coord_min)
            points -= 0.5
            points *= 2.0
            points = torch.tensor(points, device='cuda', dtype=torch.float32)
            pred_dist = model({'coords': points})['model_out']
            
            f1 = calc_symmetric_f1(pred_dist[:, 0] < 0, torch.tensor(are_inside, dtype=torch.bool, device='cuda'))
            metrics[metric_name] = f1
    return metrics


def train_siren(model, train_dataloader, epochs, lr, loss_fn, obj_path, weights_path, clip_grad=False):
    min_train_loss = float('inf')

    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=4)
    
    wandb.init(project='siren-sdf', entity='bogdanmagometa')

    total_steps = 0
    for epoch in range(epochs):
        model.train()

        train_losses = []
        for step, (model_input, gt) in enumerate(train_dataloader):
            model_input = {key: value.cuda() for key, value in model_input.items()}
            gt = {key: value.cuda() for key, value in gt.items()}

            model_output = model(model_input)
            losses = loss_fn(model_output, gt)
            reduced_losses = {loss_name: torch.mean(loss) for loss_name, loss in losses.items()}
            train_loss = sum(torch.mean(loss) for loss in losses.values())

            # wandb.log(reduced_losses, commit=False)

            optim.zero_grad()
            train_loss.backward()
            train_losses.append(train_loss.item())

            if clip_grad:
                if isinstance(clip_grad, bool):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optim.step()

            total_steps += 1
        total_train_loss = np.mean(train_losses)

        if (epoch % 10) == 0:
            print(f"Train loss = {total_train_loss:.2f}")
            metrics = eval_step(model, train_dataloader, obj_path)
            for name, val in metrics.items():
                print(f"{name} = {val:.2f}")

            wandb.log({'total_train_loss': total_train_loss, 'global_step': total_steps}, commit=False)
            wandb.log({**metrics, 'epoch': epoch})

            if total_train_loss < min_train_loss:
                min_train_loss = total_train_loss
                torch.save(
                    {
                        'centroid': dataloader.dataset.centroid,
                        'coord_min': dataloader.dataset.coord_min,
                        'coord_max': dataloader.dataset.coord_max,
                        'weights': model.state_dict()
                    },
                    weights_path)
            scheduler.step(total_train_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--obj_path', type=str, required=True)
    parser.add_argument('--xyz_path', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    
    args = parser.parse_args()
    
    process_obj(args.obj_path, args.xyz_path, num_samples=100_000)

    sdf_dataset = PointCloud(args.xyz_path, on_surface_points=5000)
    dataloader = DataLoader(sdf_dataset, batch_size=None, shuffle=True, pin_memory=True, num_workers=0)

    model = Siren(3, 256, 3, 1)
    model.cuda()

    train_siren(model, dataloader, epochs=args.epochs, lr=1e-4, loss_fn=sdf, obj_path=args.obj_path, weights_path=args.weights_path, clip_grad=False)
    wandb.finish()
