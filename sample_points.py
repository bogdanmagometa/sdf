import argparse
import os
import sys

import numpy as np

def sample_mesh(mesh_path, out_path, type, num_points, store_inside):
    from metrics import sample_points_bb, sample_points_ns

    sample_fn = sample_points_bb if type == 'bounding' else sample_points_ns
    points, inside = sample_fn(mesh_path, num_points)
    if store_inside:
        lines = ["# f {x} {y} {z} {w} where {x}, {y}, {z} are coordinates and "
                 "{w} specifies weather the point is inside or outside"]
    else:
        lines = ["# f {x} {y} {z} where {x}, {y}, {z} are coordinates"]    

    if store_inside:
        points = np.concatenate([points, inside[:, None]], axis=1)

    for p in points:
        line = f"f {' '.join([str(c) for c in p])}"
        lines.append(line)

    with open(out_path, 'w') as out_f:
        out_f.write("\n".join(lines))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample points from bounding volume "
                                     "or near surface of the mesh. "
                                     "Optionally, store the inside/outside indicators")
    parser.add_argument('--type', type=str, choices=['bounding', 'surface'], required=True)
    parser.add_argument('--mesh', type=str, required=True,
                        help="Path to .obj mesh or dir of .obj meshes")
    parser.add_argument('--out', type=str, required=True,
                        help="Path to output .obj file of points or dir where to put the files")
    parser.add_argument('--num-points', type=int, default=3000)
    parser.add_argument('--store-inside', action='store_true',
                        help='Store inside/outside location as 1/0 in w coordinate (see wiki for details on format)')
    
    args = parser.parse_args()
    
    if os.path.isdir(args.mesh):
        assert os.path.isdir(args.out), "No such directory"
        for entry in os.listdir(args.mesh):
            if entry.endswith('.obj'):
                sample_mesh(
                    os.path.join(args.mesh, entry), 
                    os.path.join(args.out, entry), 
                    args.type, args.num_points, args.store_inside
                    )
            else:
                print(f"Warning: skipped {entry} because has no .obj extension", file=sys.stderr)
    elif os.path.exists(args.mesh):
        with open(args.out, 'a'):
            pass
        sample_mesh(
            args.mesh,
            args.out,
            args.type, args.num_points, args.store_inside
            )
    else:
        print(f"Error: no such file of dir {args.mesh}", file=sys.stderr)
        exit(1)
