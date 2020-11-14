import os
import numpy as np
import matplotlib.pyplot as plt

from shape import Shape
from visualize import visualize
from normalize import normalize_data, normalize_shape
from dataLoader import import_dataset, import_normalised_data
from featureExtraction import *
from featureMatching import *
from utils import pick_file

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep 
SAVED_DATA = DATA_PATH + 'cache' + os.sep
NORMALIZED_DATA = SAVED_DATA + 'processed_data' + os.sep

try:
    features = np.load(SAVED_DATA + "features.npy", allow_pickle=True)
    features = features.item()
except:
    pass

#TODO: Include a file picker to show that aspect we also implemented, and the speed of the feature calculation
# Retrieving shape
print("\n----------------------------------------------------")
print("3D Shapes Search Engine")
print("----------------------------------------------------")

while True:
    print("Select a shape to search for similar ones. Only OFF and PLY formats are supported. \n")
    try:
        shape = pick_file()
        break
    except FileNotFoundError:
        print("File not found. Try again.\n")
        continue
    except FileExistsError:
        print("Format not supported. Please select an OFF or PLY file.\n")
        continue



print('Normalising query shape . . . ')
shape, new_n_verts, new_n_faces = normalize_shape(shape)

# Calculating features for the shape
print('Calculating features for query shape and standardize them. . .')
shape_features = calculate_single_shape_metrics(shape)
shape_features = standardize_single_shape(shape_features)

# Calculate nearest neighbors via ANN and R-Nearest Neighbors
neighbors = r_neighbors(shape_features, features)
n_shapes_id, n_distances = neighbors[0][1:], neighbors[1][1:]

# Retrieving shapes from database
n_shapes = []
for id in n_shapes_id:
    filename =NORMALIZED_DATA + "n" + str(id) + ".off"
    file = open(filename, 'r')
    verts, faces, n_verts, n_faces = read_off(file)
    mesh = trm.load_mesh(filename)
    
    shape = Shape(verts, faces, mesh)
    shape.set_id(id)
    n_shapes.append(shape)

visualize(n_shapes)


