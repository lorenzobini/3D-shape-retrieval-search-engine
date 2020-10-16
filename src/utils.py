import numpy as np
import matplotlib.pyplot as plt
import os
import re
import copy
import trimesh as trm
from tkinter import *
from tkinter import ttk
from tkinter.filedialog import askopenfilename

# from src.shape import Shape
from shape import Shape

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep
# Parse a .off file
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, other = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    return verts, faces, n_verts, n_faces


# Parse a .ply file
def parse_ply(file):
    if 'ply' != file.readline().strip():
        raise ('Not a valid PLY header')
    while True:
        line = str(file.readline().strip())
        if line.startswith('element vertex'):
            n_verts = int(line.split()[-1])  # element vertex 290 --> 290
        if line.startswith('element face'):
            n_faces = int(line.split()[-1])  # element face 190 --> 190
        if line.startswith('property'):
            # performing check for valid XYZ structure
            if (line.split()[-1] == 'x' and
                  str(file.readline().strip()).split()[-1] == 'y' and
                  str(file.readline().strip()).split()[-1] == 'z' and
                  not str(file.readline().strip()).startswith('property')):
                continue
            elif line == 'property list uchar int vertex_indices':
                continue
            else:
                raise ('Not a valid PLY header. Extra properties can not be evaluated.')
        if line == 'end_header':
            break

    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    return verts, faces, n_verts, n_faces


# Read the classes from the .cla file
def read_classes(file, class_list):
    if 'PSB' not in file.readline().strip():
        raise ('Not a valid PSB classification header')
    _, num_models = file.readline().strip().split()
    modelcount = 0
    class_dict = {}
    while modelcount < int(num_models):
        line = file.readline().strip().split()
        if len(line) == 0:
            pass  
        elif len(line) > 2  and line[2] == '0': # empty class label
            pass
        elif len(line) > 2:
            class_name = str(line[0])
            # if the class not in the class_list add it
            if class_name  not in class_list:
                class_list.append(class_name)
        else: # add the class to the number of the model
            class_id = class_list.index(class_name) # give class id based on class_list index
            class_dict[line[0]] = (class_id, class_name)
            modelcount += 1

    return class_dict, class_list


# Read the .txt field and retrieve information
def read_info(file, shape):
    for line in file:
        if line.startswith('mid'):
            shape.set_id(int(line.split()[-1])) 

        if line.startswith('avg_depth'):
            shape.set_avg_depth(float(line.split()[-1]))
        if line.startswith('center'):
            pattern = 'center: \((?P<x>.*),(?P<y>.*),(?P<z>.*)\)'
            matches = re.match(pattern, line)
            shape.set_center((float(matches.group('x')),
                              float(matches.group('y')),
                              float(matches.group('z'))))
        if line.startswith('scale'):
            shape.set_scale(float(line.split()[-1]))

    return shape


def calculate_box(vertices):
    
    x_coords = [x[0] for x in vertices]
    y_coords = [x[1] for x in vertices]
    z_coords = [x[2] for x in vertices]

    return [min(x_coords),min(y_coords), min(z_coords), max(x_coords), max(y_coords), max(z_coords)]


# Function that removes TriangleMesh objects for saving
def remove_meshes(shapes):
    new_shapes = []
    for shape in shapes:
        new_shape = copy.deepcopy(shape)
        new_shape.delete_mesh()
        new_shapes.append(new_shape)

    return new_shapes


# Function to write off file on disk
def write_off(path, shape):
    verts = shape.get_vertices()
    faces = shape.get_faces()

    f = open(path + 'n' + str(shape.get_id()) + ".off", "w+")
    f.write("OFF\n")
    f.write(str(len(verts)) + " " + str(len(faces)) + " " + "0\n")
    for i in range(0, len(verts)):
        vert = [str(x) for x in verts[i]]
        vert = ' '.join(vert)
        f.write(vert + '\n')
    for i in range(0, len(faces)):
        face = [str(x) for x in faces[i]]
        face = ' '.join(face)
        f.write( face + '\n')
    f.close()


# Calculates eigenvectors based on the vertices
def calc_eigenvectors(verts):
    A = np.zeros((3, len(verts)))

    A[0] = np.array([x[0] for x in verts]) # First row is all the X_coords
    A[1] = np.array([x[1] for x in verts]) # second row is all the Y_coords
    A[2] = np.array([x[2] for x in verts]) # third row is all the z-coords
    
    A_cov = np.cov(A) # This is returns a 3x3
    eigenvalues, eigenvectors = np.linalg.eigh(A_cov)
    return eigenvalues, eigenvectors


# Compute the angle between 3 points
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.abs(cosine_angle))
    return angle


# Normalizing a histogram H = {hi} is simple: Replace each value hi by hi/ Si hi. This way, each bar becomes a percentage
# in [0,1], regardless of the total number of samples Si hi.
def normalize_hist(hist):
    hsum = np.sum(hist)
    newhist = []
    for hi in hist:
        newhist.append(hi/hsum)
    return newhist

# Standardize the non-histogram features 
def standardize(features):
    V, A, C, BB, D, E = [],[],[],[],[],[]
    for id, featuresList in features.items():
        V.append(featuresList["volume"])
        A.append(featuresList['area'])
        C.append(featuresList['compactness'])
        BB.append(featuresList["bbox_volume"])
        D.append(featuresList["diameter"])
        E.append(featuresList["eccentricity"])
    
    sdVals = save_standardization_vals(V, A, C, BB, D, E)

    for id, featuresList in features.items():
        features[id]['volume'] = (featuresList["volume"]-sdVal["V_mean"])/sdVals["V_std"]
        features[id]['area'] = (featuresList['area']-sdVal["A_mean"])/sdVals["A_std"]
        features[id]['compactness'] = (featuresList['compactness']-sdVal["C_mean"])/sdVals["C_std"]
        features[id]['bbox_volume'] = (featuresList["bbox_volume"]-sdVal["BB_mean"])/sdVals["BB_std"]
        features[id]['diameter'] = (featuresList["diameter"] - sdVal["D_mean"])/sdVals["D_std"]
        features[id]['eccentricity'] = (featuresList["eccentricity"] - sdVal["E_mean"])/sdVals["E_std"]
    
    np.save(SAVED_DATA + "features.npy", features)
    np.save(SAVED_DATA + "standardization_values.npy", sdVals)
    return features

def save_standardization_vals(V, A, C, BB, D, E):
    standardVals = {}
    standardVals["V_mean"] = np.mean(V)
    standardVals["V_std"] = np.std(V)

    standardVals["A_mean"] = np.mean(A)
    standardVals["A_std"] = np.std(A)

    standardVals["C_mean"] = np.mean(C)
    standardVals["C_std"] = np.std(C)

    standardVals["BB_mean"] = np.mean(BB)
    standardVals["BB_std"] = np.std(BB)

    standardVals["D_mean"] = np.mean(D)
    standardVals["D_std"] = np.std(D)

    standardVals["E_mean"] = np.mean(E)
    standardVals["E_std"] = np.std(E)

    return standardVals