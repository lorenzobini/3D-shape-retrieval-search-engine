import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# Other file imports
# from shape import Shape
# from boundingbox import BoundingBox
# from utils import calc_eigenvectors
from src.boundingbox import BoundingBox
from src.shape import Shape
from src.utils import calc_eigenvectors
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

    return abs(volume)


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
    eigenvalues, _ = calc_eigenvectors(shape.get_vertices(), shape.get_id())
    min_eigen = np.min(eigenvalues)
    max_eigen = np.max(eigenvalues)
    return np.abs(max_eigen)/np.abs(min_eigen)


# Angle between 3 random vertices
def calc_A3(shape):
    pass


# Distance between barycenter and random vertex, returns histogram vector and bin_edges
def calc_D1(shape):
    xc, yc, zc = shape.get_center()
    D = []
    # Select from
    vertices = random.choices(shape.get_vertices(), k = 5000)
    for x, y, z in vertices:
        D.append(np.sqrt(((x-xc)**2)+((y-yc)**2)+((z-zc)**2)))
    # crate histogram with 10 bins from 0 - 1.0
    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0,1, 0.1))
    return (hist, bin_edges)


# Distance between 2 random vertices
def calc_D2(shape):
    verticeList = random.choices(shape.get_vertices(), k = 1000) # select 1000 random vertices
    # Divide into two lists, do it this way to prevent selecting the same vertice
    verticesOne = verticeList[:500]
    verticesTwo = verticeList[500:]
    D = []
    # Loop over both sets to create the vertice combinations, 250000 total
    for x1, y1, z1 in verticesOne:
        for x2, y2, z2 in verticesTwo:
            D.append(np.sqrt(((x1 - x2) ** 2)+((y1 - y2) ** 2) + ((z1 - z2) ** 2)))
    # Create histogram with 10 bins from 0 to 1.25 (as there where some values above 1)
    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0, 1.25, 0.125))
    return(hist, bin_edges)


# Square root of area of triangle given by 3 random vertices
def calc_D3(shape):
    verticeList = random.choices(shape.get_vertices(), k=999)  # select 999 random vertices
    # Divide into three lists, do it this way to prevent selecting the same vertice
    verticesOne = verticeList[:333]
    verticesTwo = verticeList[333:666]
    verticesThree = verticeList[666:]
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

    hist, bin_edges = np.histogram(np.array(D), bins=np.arange(0, 1.25, 0.125)) # TODO: bins may need adjustment

    return (hist, bin_edges)


# Cube root of volume of tetrahedron formed by 4 random vertices
def calc_D4(shape):
    verticeList = random.choices(shape.get_vertices(), k=999)  # select 999 random vertices
    # Divide into three lists, do it this way to prevent selecting the same vertice
    verticesOne = verticeList[:333]
    verticesTwo = verticeList[333:666]
    verticesThree = verticeList[666:]
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

    hist, bin_edges = np.histogram(np.array(D), bins=np.arange(0, 1.25, 0.125))  # TODO: bins may need adjustment

    return (hist, bin_edges)




