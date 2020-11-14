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

# Retrieving shape
print("\n----------------------------------------------------")
print("3D Shapes Search Engine")
print("----------------------------------------------------")

# TODO: Select goodworking one other than the sword or a human
id = 1150 # Demo ID


print("Selected query shape is "+ str(id))
query_shape_features = features[id]
file = open(NORMALIZED_DATA + '\\' + "n"+str(id) + ".off", "r")
verts, faces, n_verts, n_faces = read_off(file)
mesh = trm.load_mesh(NORMALIZED_DATA + "n"+str(id) + ".off" + os.sep)
query_shape = Shape(verts, faces, mesh) 

print("Calcualting similarities...")
similarities = calc_distance(features, query_shape_features, 204)

print("Retrieving and showing similar shapes...")
shapes = load_similar(similarities, query_shape)

visualize(shapes[1:])
