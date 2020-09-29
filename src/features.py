import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import defaultdict

# Other file imports
# from shape import Shape
from src.shape import Shape

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep

def calculate_metrics(shapes):
    print("Calculating all the object features...")

    features = defaultdict(lambda: defaultdict(None))

     # calculate the metrics for each shape
    for shape in shapes:

        id = shape.get_id()

        # calculate the default metrics
        features[id]["surface_area"] = surface_area(shape)
        features[id]["compactness"] = compactness(shape)
        features[id]["bbox_volume"] = bbox_volume(shape)
        features[id]["diameter"] = diameter(shape)
        features[id]["eccentricity"] = eccentricity(shape)

        # Distributions
        features[id]["A3"] = calc_A3(shape)
        features[id]["D1"] = calc_D1(shape)
        features[id]["D2"] = calc_D2(shape)
        features[id]["D3"] = calc_D3(shape)
        features[id]["D4"] = calc_D4(shape)

    print("Done calculating the features.")

    # Saving features to disk
    np.save(SAVED_DATA + "features.npy", features)

    return shapes, features

def surface_area(shape):
    volume_arr = [["id", "volume"]]
    
    # The mesh must be closed (watertight) to compute the volume
    if shape.is_watertight():
        #center = shape.get_center()
        verts = shape.get_vertices()
        faces = shape.get_faces()
        volume = float(0)
        # Loop through all the triangles in the shape
        for face in faces:
            # The three vertex coordinates of the triangle
            p1 = verts[face[0]]
            p2 = verts[face[1]]
            p3 = verts[face[2]]
            # Calculate the volume of the tetrahetron (triangle + origin)
            volume_temp = signed_volume_of_triangle(p1, p2, p3)
            volume = volume + volume_temp
        volume_arr = np.append(volume_arr, [[shape.get_id(), round(abs(volume),2)]], axis=0)

    return volume_arr

def signed_volume_of_triangle(p1, p2, p3):
    
    # p[0] = x coordinate, p[1] = y coordinate, p[2] = z coordinate
    v321 = p3[0]*p2[1]*p1[2]
    v231 = p2[0]*p3[1]*p1[2]
    v312 = p3[0]*p1[1]*p2[2]
    v132 = p1[0]*p3[1]*p2[2]
    v213 = p2[0]*p1[1]*p3[2]
    v123 = p1[0]*p2[1]*p3[2]

    return 1/6*(-v321 + v231 + v312 - v132 - v213 + v123)

def compactness(shape):
    pass

def bbox_volume(shape):
    pass

def diameter(shape):
    pass

def eccentricity(shape):
    pass

# Angle between 3 random vertices
def calc_A3(shape):
    pass

# Distance between barycenter and random vertex, returns histogram vector and bin_edges
def calc_D1(shape):
    xc, yc, zc = shape.get_center()
    D = []
    for x, y, z in shape.vertices:
        D.append(np.sqrt(((x-xc)**2)+((y-yc)**2)+((z-zc)**2)))
    # crate histogram with 10 bins from 0 - 1.0
    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0,1, 0.1))
    return (hist, bin_edges)

# Distance between 2 random vertices
def calc_D2(shape):
    pass

# Square root of area of triangle given by 3 random vertices
def calc_D3(shape):
    pass

# Cube root of volume of tetrahedron formed by 4 random vertices
def calc_D4(shape):
    pass




