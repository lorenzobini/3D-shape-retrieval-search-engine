import open3d as o3d
import numpy as np
import trimesh as trm
import math
import os
import matplotlib.pyplot as plt
import random


# Other file imports
# from shape import Shape
# from boundingbox import BoundingBox
# from utils import calc_eigenvectors, euclidean
from src.utils import *
from src.boundingbox import BoundingBox
from src.shape import Shape
from src.utils import calc_eigenvectors
DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep


def calculate_metrics(shapes):
    print("Calculating all the object features...")

    features = {}

     # calculate the metrics for each shape
    for shape in shapes:

        id = shape.get_id()
        features[id] = {}

        # calculate the default metrics
        features[id]["volume"] = volume(shape)
        features[id]["area"] = area(shape)
        features[id]["compactness"] = compactness(shape)
        features[id]["bbox_volume"] = bbox_volume(shape)
        features[id]["diameter"] = diameter(shape)
        features[id]["eccentricity"] = eccentricity(shape)

        # Distributions
        distributions = calc_distributions(shape)
        features[id]["A3"] = distributions["A3"]
        features[id]["D1"] = distributions["D1"]
        features[id]["D2"] = distributions["D2"]
        features[id]["D3"] = distributions["D3"]
        features[id]["D4"] = distributions["D4"]

    print("Done calculating the features.")
    # Saving features to disk
    np.save(SAVED_DATA + "features.npy", features)

    return shapes, features


def volume(shape):  

    # TODO: Fix that holes get filled
    volume = shape.get_mesh_trm().volume
    #print('Volume : ', volume)

    # The mesh must be closed (watertight) to compute the volume
    if shape.is_watertight():
        
        #print('is watertight')

        trm.repair.fix_winding(shape.get_mesh_trm())
        # Check that all triangles are consistently oriented over the surface
        if shape.get_mesh_trm().is_winding_consistent:
            #print('is winding consistent')
            
            # TODO: Do pre-processing steps before feature extraction
            trm.repair.fix_normals(shape.get_mesh_trm())
            trm.repair.fix_inversion(shape.get_mesh_trm())
            trm.repair.fill_holes(shape.get_mesh_trm())
            shape.get_mesh_trm().remove_degenerate_faces()

            # Check if the mesh has all the properties required to represent a valid volume
            if shape.get_mesh_trm().is_volume:
                volume = shape.get_mesh_trm().volume
                print('Volume : ', volume)

            else:
                print('Not a valid mesh to calculate the volume')
                pass

    return volume


def area(shape):
    mesh_trm = shape.get_mesh_trm()
    area = mesh_trm.area
    print('Area : ', area)

    return area


def compactness(shape):

    # TODO: Not calculate it again, but get it from the saved features
    mesh_trm = shape.get_mesh_trm()
    V = mesh_trm.volume
    S = mesh_trm.area

    compactness = pow(S, 3) / (36*math.pi*(pow(V, 2)))
    sphericity = 1/ compactness
    print('Compactness : ', compactness)
    print('Sphericity : ', sphericity)
    
    return compactness


# Volume of the shape's axis-aligned bounding box
def bbox_volume(shape):
    bbox = shape.get_buinding_box()
    x = bbox.get_xmax() - bbox.get_xmin()
    y = bbox.get_ymax() - bbox.get_ymin()
    z = bbox.get_zmax() - bbox.get_zmin()

    volume = x * y * z

    return volume


# Largest distance between two points on the surface of the shape
def diameter(shape):
    max_distance = 0
    for x1, y1, z1 in shape.get_vertices():
        for x2, y2, z2 in shape.get_vertices():
            distance = np.sqrt(((x1-x2)**2)+((y1-y2)**2)+((z1-z2)**2))
            if distance > max_distance:
                max_distance = distance

    return max_distance


# Calculate the ratio between the minor eigenvalue en major eigenvalue
def eccentricity(shape):
    eigenvalues, _ = calc_eigenvectors(shape.get_vertices())
    min_eigen = np.min(eigenvalues)
    max_eigen = np.max(eigenvalues)
    return np.abs(max_eigen)/np.abs(min_eigen)


def calc_distributions(shape):
    descriptors = {}

    verticeList = random.sample(list(shape.get_vertices()), k=1000)  # select 1000 random vertices
    center  = shape.get_center()
    # Computing D1 -----------------------
    descriptors["D1"] = calc_D1(center, verticeList)

    # Computing D2 -----------------------
    verticesOne = verticeList[:500]
    verticesTwo = verticeList[500:]

    descriptors["D2"] = calc_D2(verticesOne, verticesTwo)

    # Computing A3, D3, D4 ---------------
    verticesOne = verticeList[:333]
    verticesTwo = verticeList[333:667]
    verticesThree = verticeList[667:]

    A3 = []
    D3 = []
    D4 = []
    for a in verticesOne:
        for b in verticesTwo:
            for c in verticesThree:
                # Computing A3 for the current combination
                A3.append(calc_A3(a, b, c))
                # Computing D3 for the current combination
                D3.append(calc_D3(a, b, c))
                # Computing D4 for the current combination
                D4.append(calc_D4(center, a, b, c))

    # Computing Histogram A3
    hist, bin_edges = np.histogram(np.array(A3), bins=np.arange(0.0, 180.0, 18))
    hist = normalize_hist(hist)

    descriptors["A3"] = (hist, bin_edges)

    # Computing Histogram D3
    hist, bin_edges = np.histogram(np.array(D3), bins=np.arange(0.0, 0.75, 0.075))
    hist = normalize_hist(hist)

    descriptors["D3"] = (hist, bin_edges)

    # Computing Histogram D4
    hist, bin_edges = np.histogram(np.array(D4), bins=np.arange(0.0, 0.75, 0.075))
    hist = normalize_hist(hist)

    descriptors["D4"] = (hist, bin_edges)

    return descriptors


# Angle between 3 random vertices
def calc_A3(a, b, c):
    angle = compute_angle(a, b, c)

    return angle


# Distance between barycenter and random vertex, returns histogram vector and bin_edges
def calc_D1(center, vertices):
    (xc, yc, zc) = center
    D = []

    for x, y, z in vertices:
        D.append(euclidean(x, y, z, xc, yc, zc))
    # crate histogram with 10 bins from 0 - 1.0
    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0,1, 0.1))
    hist = normalize_hist(hist)

    return (hist, bin_edges)


# Distance between 2 random vertices
def calc_D2(verticesOne, verticesTwo):

    D = []

    # Loop over both sets to create the vertice combinations, 250000 total
    for x1, y1, z1 in verticesOne:
        for x2, y2, z2 in verticesTwo:
            D.append(euclidean(x1, y1, z1, x2, y2, z2))
    # Create histogram with 10 bins from 0 to 1.25 (as there where some values above 1)

    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0, 1.25, 0.125))
    hist = normalize_hist(hist)

    return(hist, bin_edges)


# Square root of area of triangle given by 3 random vertices
def calc_D3(a, b, c):
    (x1, y1, z1) = a
    (x2, y2, z2) = b
    (x3, y3, z3) = c

    # Calculating the area of the triangle using
    # side lengths
    p1_p2 = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2))
    p1_p3 = np.sqrt(((x1 - x3) ** 2) + ((y1 - y3) ** 2) + ((z1 - z3) ** 2))
    p2_p3 = np.sqrt(((x2 - x3) ** 2) + ((y2 - y3) ** 2) + ((z2 - z3) ** 2))

    sp = 0.5 * (p1_p2 + p1_p3 + p2_p3)
    area = np.sqrt(sp * (sp-p1_p2) * (sp-p1_p3) * (sp-p2_p3))

    return np.sqrt(area)


# Cube root of volume of tetrahedron formed by 4 random vertices
def calc_D4(center, p1, p2, p3):

    p1_c = np.subtract(p1, center)
    p2_c = np.subtract(p2, center)
    p3_c = np.subtract(p3, center)

    volume = np.abs(np.dot(p1_c, np.cross(p2_c, p3_c))) / 6

    return np.cbrt(volume)









