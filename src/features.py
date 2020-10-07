import open3d as o3d
import numpy as np
import trimesh as trm
import math
import os
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Other file imports
from shape import Shape
from boundingbox import BoundingBox
from utils import calc_eigenvectors, euclidean
# from src.boundingbox import BoundingBox
# from src.shape import Shape
# from src.utils import calc_eigenvectors
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
        features[id]["A3"] = calc_A3(shape)
        features[id]["D1"] = calc_D1(shape)
        features[id]["D2"] = calc_D2(shape)
        features[id]["D3"] = calc_D3(shape)
        features[id]["D4"] = calc_D4(shape)

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

# Compute the angle between 3 points
def compute_angle(a, b, c):
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.abs(cosine_angle))
    return np.degrees(angle)

# Angle between 3 random vertices
def calc_A3(shape):
    verticeList = random.sample(list(shape.get_vertices()), k=300)  # select 300 random vertices
    # Divide into two lists, do it this way to prevent selecting the same vertice
    verticesOne = verticeList[:100]
    verticesTwo = verticeList[100:200]
    verticesThree = verticeList[200:]
    D = []
    for a in verticesOne:
        for b in verticesTwo:
            for c in verticesThree:
                # Calculating the area of the triangle using
                # side lengths

                D.append(compute_angle(a, b, c))

    hist, bin_edges = np.histogram(np.array(D), bins=np.arange(0.0, 180.0, 18)) # TODO: bins may need adjustment
    hist = normalize_hist(hist)
    return (hist, bin_edges)


# Distance between barycenter and random vertex, returns histogram vector and bin_edges
def calc_D1(shape):
    xc, yc, zc = shape.get_center()
    D = []
    # Select from
    vertices = random.choices(shape.get_vertices(), k = 5000)
    for x, y, z in vertices:
        D.append(euclidean(x,y,z,xc,yc,zc))
    # crate histogram with 10 bins from 0 - 1.0
    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0,1, 0.1))
    hist = normalize_hist(hist)
    return (hist, bin_edges)


# Distance between 2 random vertices
def calc_D2(shape):
    verticeList = random.sample(list(shape.get_vertices()), k = 1000) # select 1000 random vertices
    # Divide into two lists, do it this way to prevent selecting the same vertice
    verticesOne = verticeList[:500]
    verticesTwo = verticeList[500:]
    D = []
    # Loop over both sets to create the vertice combinations, 250000 total
    for x1, y1, z1 in verticesOne:
        for x2, y2, z2 in verticesTwo:
            D.append(euclidean(x1,y1,z1,x2,y2,z2))
    # Create histogram with 10 bins from 0 to 1.25 (as there where some values above 1)
    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0, 1.25, 0.125))
    hist = normalize_hist(hist)
    return(hist, bin_edges)


# Square root of area of triangle given by 3 random vertices
def calc_D3(shape):
    verticeList = random.sample(list(shape.get_vertices()), k=300)  # select 300 random vertices
    # Divide into two lists, do it this way to prevent selecting the same vertice
    verticesOne = verticeList[:100]
    verticesTwo = verticeList[100:200]
    verticesThree = verticeList[200:]
    D = []
    for x1, y1, z1 in verticesOne:
        for x2, y2, z2 in verticesTwo:
            for x3, y3, z3 in verticesThree:
                # Calculating the area of the triangle using
                # side lengths
                p1_p2 = np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2) + ((z1 - z2) ** 2))
                p1_p3 = np.sqrt(((x1 - x3) ** 2) + ((y1 - y3) ** 2) + ((z1 - z3) ** 2))
                p2_p3 = np.sqrt(((x2 - x3) ** 2) + ((y2 - y3) ** 2) + ((z2 - z3) ** 2))

                sp = 0.5 * (p1_p2 + p1_p3 + p2_p3)
                area = np.sqrt(sp * (sp-p1_p2) * (sp-p1_p3) * (sp-p2_p3))

                D.append(np.sqrt(area))

    hist, bin_edges = np.histogram(np.array(D), bins=np.arange(0.0, 0.75, 0.075)) # TODO: bins may need adjustment
    hist = normalize_hist(hist)
    return (hist, bin_edges)


# Cube root of volume of tetrahedron formed by 4 random vertices
def calc_D4(shape):
    verticeList = random.sample(list(shape.get_vertices()), k=300)  # select 999 random vertices
    # Divide into three lists, do it this way to prevent selecting the same vertice
    verticesOne = verticeList[:100]
    verticesTwo = verticeList[100:200]
    verticesThree = verticeList[200:]
    c = shape.get_center()
    D = []
    for p1 in verticesOne:
        for p2 in verticesTwo:
            for p3 in verticesThree:
                p1_c = np.subtract(p1, c)
                p2_c = np.subtract(p2, c)
                p3_c = np.subtract(p3, c)

                volume = np.abs(np.dot(p1_c, np.cross(p2_c, p3_c))) / 6

                D.append(np.cbrt(volume))

    hist, bin_edges = np.histogram(np.array(D),bins=np.arange(0.0, 0.75, 0.075)) 
    hist = normalize_hist(hist) 
    return (hist, bin_edges)

# Normalizing a histogram H = {hi} is simple: Replace each value hi by hi/ Si hi. This way, each bar becomes a percentage
# in [0,1], regardless of the total number of samples Si hi.
def normalize_hist(hist):
    hsum = np.sum(hist)
    newhist = []
    for hi in hist:
        newhist.append(hi/hsum)
    return newhist



