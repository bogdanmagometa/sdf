from collections import OrderedDict

import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ddd import read_xyz

def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
    '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    grad = gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(grad, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(grad[..., :1]))
    grad_constraint = torch.abs(grad.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = nn.Sequential()
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for _ in range(hidden_layers):
            self.net.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0)
                )

        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                            np.sqrt(6 / hidden_features) / hidden_omega_0)
        self.net.append(final_linear)

    def forward(self, model_input):
        coords = model_input['coords'].clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        with torch.set_grad_enabled(True):
            output = self.net(coords)
        return {'model_in': coords, 'model_out': output}

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class PointCloud(Dataset):
    def __init__(self, pointcloud_path, on_surface_points, keep_aspect_ratio=True):
        super().__init__()

        # point_cloud = np.genfromtxt(pointcloud_path)
        coords, self.normals = read_xyz(pointcloud_path)
        # coords = point_cloud[:, :3]
        # self.normals = point_cloud[:, 3:]

        # Reshape point cloud such that it lies in bounding box of (-1, 1) (distorts geometry, but makes for high
        # sample efficiency)
        self.centroid = np.mean(coords, axis=0, keepdims=True)
        coords -= self.centroid
        if keep_aspect_ratio:
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)
        else:
            coord_max = np.amax(coords, axis=0, keepdims=True)
            coord_min = np.amin(coords, axis=0, keepdims=True)
        self.coord_max = coord_max
        self.coord_min = coord_min

        self.coords = (coords - coord_min) / (coord_max - coord_min)
        self.coords -= 0.5
        self.coords *= 2.

        self.on_surface_points = on_surface_points

    def __len__(self):
        return self.coords.shape[0] // self.on_surface_points

    def __getitem__(self, idx):
        point_cloud_size = self.coords.shape[0]

        off_surface_samples = self.on_surface_points  # **2
        total_samples = self.on_surface_points + off_surface_samples

        # Random coords
        rand_idcs = np.random.choice(point_cloud_size, size=self.on_surface_points)

        on_surface_coords = self.coords[rand_idcs, :]
        on_surface_normals = self.normals[rand_idcs, :]

        off_surface_coords = np.random.uniform(-1, 1, size=(off_surface_samples, 3))
        off_surface_normals = np.ones((off_surface_samples, 3)) * -1

        sdf = np.zeros((total_samples, 1))  # on-surface = 0
        sdf[self.on_surface_points:, :] = -1  # off-surface = -1

        coords = np.concatenate((on_surface_coords, off_surface_coords), axis=0)
        normals = np.concatenate((on_surface_normals, off_surface_normals), axis=0)

        return {'coords': torch.from_numpy(coords).float()}, {'sdf': torch.from_numpy(sdf).float(),
                                                              'normals': torch.from_numpy(normals).float()}
