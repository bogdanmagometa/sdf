"""ddd.py

Module with 3D stuff
"""

import pymeshlab as ml
import numpy as np

def read_obj(file_path):
    """Read vertices and faces from .obj file"""
    vertices = []
    faces = []

    with open(file_path, 'r') as file:
        for line in file:
            tokens = line.strip().split()
            if not tokens:
                continue

            if tokens[0] == 'v':
                vertices.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            elif tokens[0] == 'f':
                face = [int(vertex.split('/')[0]) - 1 for vertex in tokens[1:]]
                faces.append(face)

    return np.array(vertices), np.array(faces)


def process_obj(input_obj, output_xyz, num_samples=100_000):
    # Create a MeshSet object
    ms = ml.MeshSet()

    # Load the input OBJ file
    ms.load_new_mesh(input_obj)

    # Sample points from the surface using Poisson disk sampling
    ms.generate_sampling_poisson_disk(samplenum=num_samples)

    # Get sampled points and normals
    points = ms.current_mesh().vertex_matrix()
    normals = ms.current_mesh().vertex_normal_matrix()

    # Save sampled points and normals to XYZ file
    save_to_xyz(output_xyz, points, normals)

def save_to_xyz(output_xyz, points, normals=None):
    with open(output_xyz, 'w') as f:
        for i in range(points.shape[0]):
            point = points[i]
            line = f"{point[0]} {point[1]} {point[2]}"
            if normals is not None:
                normal = normals[i]
                line += f" {normal[0]} {normal[1]} {normal[2]}"
            line += "\n"
            f.write(line)

def read_xyz(input_xyz: str):
    vertices = []
    normals = []

    with open(input_xyz, 'r') as f:
        for line in f:
            values = line.split()
            if len(values) < 3:
                # Skip lines with insufficient data
                continue

            # Extract vertex coordinates
            vertex = [float(values[0]), float(values[1]), float(values[2])]
            vertices.append(vertex)

            # Extract normal coordinates if available
            if len(values) >= 6:
                normal = [float(values[3]), float(values[4]), float(values[5])]
                normals.append(normal)

    vertices_matrix = np.array(vertices)
    normals_matrix = np.array(normals) if len(normals) > 0 else None

    return vertices_matrix, normals_matrix


