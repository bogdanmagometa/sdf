import sys
sys.path.append('./nglod/sdf-net')
# import wandb

import torch

from lib.trainer import Trainer
from lib.models import SOL_NGLOD
from metrics import sample_points_bb, sample_points_ns, calc_symmetric_f1
from nglod import build_parser, args_to_str

from lib.torchgp import load_obj

import trimesh

class CustomizedTrainer(Trainer):
    def __init__(self, args, args_str):
        super().__init__(args, args_str)
        self.set_scheduler()
        self.timer.check('set_scheduler')

    def set_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, threshold=0.02, threshold_mode='rel')

    def train(self):
        # wandb.init()
        super().train()
        # wandb.finish()

    def wb_log(self, metrics, epoch):
        wandb.log({**metrics, 'epoch': epoch}, commit=False)

    def post_epoch(self, epoch):
        WB_LOG_INTERVAL = 10
        F1_LOG_INTERVAL = 30
        lr = self.optimizer.param_groups[0]['lr']
        # if epoch % WB_LOG_INTERVAL == 0:
        #     if epoch % F1_LOG_INTERVAL == 0:
        #         metrics = self.f1_step()
        #         self.wb_log(metrics, epoch)
        #     wandb.log({'lr': lr, 'total_loss': self.log_dict['total_loss'], 'epoch': epoch})
        super().post_epoch(epoch)
        print(f"Total loss: {self.log_dict['total_loss']}")
        self.scheduler.step(self.log_dict['total_loss'])
        if lr < 1e-6:
            exit()

    def f1_step(self):
        obj_path = self.args.dataset_path
        
        self.net.eval()
        metrics = {}
        with torch.no_grad():
            for metric_name, sample_fn in [('bounding F1', sample_points_bb), ('surface F1', sample_points_ns)]: 
                points, are_inside = sample_fn(obj_path, 3_000)
                unnorm_points = points
                
                points = torch.tensor(points, dtype=torch.float32)
                points = self.normalize(points).cuda()
                pred_dist = self.net.sdf(points)
                
                print((pred_dist.detach().cpu().numpy() < 0).sum())
                cloud = trimesh.PointCloud(unnorm_points[(pred_dist[:, 0].detach().cpu().numpy() < 0) & are_inside])
                cloud.export('tp.obj')
                cloud = trimesh.PointCloud(unnorm_points[(pred_dist[:, 0].detach().cpu().numpy() > 0) & are_inside])
                cloud.export('fp.obj')
                cloud = trimesh.PointCloud(unnorm_points[(pred_dist[:, 0].detach().cpu().numpy() < 0) & ~are_inside])
                cloud.export('fn.obj')
                cloud = trimesh.PointCloud(unnorm_points[(pred_dist[:, 0].detach().cpu().numpy() > 0) & ~are_inside])
                cloud.export('tn.obj')

                f1 = calc_symmetric_f1(pred_dist[:, 0] < 0, torch.tensor(are_inside, dtype=torch.bool, device='cuda'))
                metrics[metric_name] = f1
                break
        return metrics

    def normalize(self, points: torch.Tensor):
        V, _ = load_obj(self.args.dataset_path)
        
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
        return points



if __name__ == "__main__":
    obj_num = sys.argv[1]

    parser = build_parser()
    args = parser.parse_args([
        '--net', 'OctreeSDF', 
        '--num-lods', '3',
        '--dataset-path', f'./test_task_meshes/{obj_num}.obj', #'nglod/sdf-net/data/armadillo.obj'
        '--epoch', '600',
        '--exp-name', obj_num,
        '--matcap-path', 'nglod/sdf-net/data/matcap/green.png',
        # '--feature-dim', '24',
        '--feature-dim', '24',
        '--num-samples', '100000',
        '--pretrained', f'_results_old/models/{obj_num}.pth',
        '--batch-size', '512',
        '--lr', '0.001',
        '--model-path', '_results_nn/models',
        '--logs', '_results_nn/logs/runs/',
        ])
    args_str = args_to_str(args, parser)

    model = CustomizedTrainer(args, args_str)
    # pprint.pprint(model.f1_step())
    model.train()
    
    # net = SOL_NGLOD(model.net)
    # torch.save(net, 'sol_nglod.pth')
