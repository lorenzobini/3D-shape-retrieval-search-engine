import math
import random
from scipy import spatial


# Other file imports
# from shape import Shape
# from boundingbox import BoundingBox
# from utils import *
from src.utils import *
from src.boundingbox import BoundingBox
from src.shape import Shape


DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep


def calculate_metrics(shapes, last_batch = False):
    print("Calculating all the object features . . .")

    # If some features are present, load them
    if os.path.isfile(SAVED_DATA + "features.npy"):
        features = np.load(SAVED_DATA + "features.npy", allow_pickle=True)
        try:
            features = features.item()
        except:
            pass
    else:
        features = {}

     # calculate the metrics for each shape
    for shape in shapes:
        id = shape.get_id()
        features[id] = calculate_single_shape_metrics(shape)

    print("Done calculating the features.")

    print("Saving the features.")

    # Saving features to disk
    np.save(SAVED_DATA + "features.npy", features)

    return shapes, features


# Calculate features for a single shape
def calculate_single_shape_metrics(shape):
    features = {}

    # calculate the default metrics
    features["volume"] = volume(shape)
    features["area"] = area(shape)
    features["compactness"] = compactness(features["area"], features["volume"])
    features["bbox_volume"] = bbox_volume(shape)
    features["diameter"] = diameter(shape)
    features["eccentricity"] = eccentricity(shape)

    # Distributions
    distributions = calc_distributions(shape)
    features["A3"] = distributions["A3"]
    features["D1"] = distributions["D1"]
    features["D2"] = distributions["D2"]
    features["D3"] = distributions["D3"]
    features["D4"] = distributions["D4"]

    return features


def volume(shape):  

    mesh = shape.get_mesh()
    # Take the convex hull: the smallest enclosing convex polyhedron
    ch_verts = mesh.convex_hull.vertices
    ch_faces = mesh.convex_hull.faces
    center = shape.get_center()
    volume = float(0)

    if mesh.convex_hull.is_volume:
        for face in ch_faces:
            p1 = np.subtract(ch_verts[face[0]], center)
            p2 = np.subtract(ch_verts[face[1]], center)
            p3 = np.subtract(ch_verts[face[2]], center)

            volume = volume + (np.abs(np.dot(p1, np.cross(p2, p3))) / 6)
    return volume


def area(shape):
    mesh = shape.get_mesh()
    return mesh.area

def compactness(V, S):
    return pow(S, 3) / (36*math.pi*(pow(V, 2))) # Volume of the shape's axis-aligned bounding box

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

    verticeList = random.choices(list(shape.get_vertices()), k=5000)  # select 1000 random vertices
    center  = shape.get_center()

    # Computing D1 -----------------------
    print("Computing D1 distribution . . .")
    descriptors["D1"] = calc_D1(center, verticeList)

    verticeList = random.choices(list(shape.get_vertices()), k=1000)

    # Computing D2 -----------------------
    print("Computing D2 distribution . . .")
    verticesOne = verticeList[:]
    verticesTwo = verticeList[500:]

    descriptors["D2"] = calc_D2(verticesOne, verticesTwo)
    verticeList = random.sample(list(shape.get_vertices()), k=150)
    # Computing A3, D3, D4 ---------------
    verticesOne = verticeList[:50]
    verticesTwo = verticeList[50:100]
    verticesThree = verticeList[100:]

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
                while True:
                    d = random.sample(list(shape.get_vertices()), k =1)[0]
                    if d is not a and d is not b and d is not c:
                        D4.append(calc_D4(a, b, c, d))
                        break

    # Computing Histogram A3
    hist, bin_edges = np.histogram(np.array(A3), bins=np.arange(0.0, 2.75, 0.25))
    hist = normalize_hist(hist)

    descriptors["A3"] = (hist, bin_edges)

    # Computing Histogram D3
    hist, bin_edges = np.histogram(np.array(D3), bins=np.arange(0.0, 0.825, 0.075))
    hist = normalize_hist(hist)

    descriptors["D3"] = (hist, bin_edges)

    # Computing Histogram D4
    hist, bin_edges = np.histogram(np.array(D4), bins=np.arange(0.0, 0.55, 0.05))
    hist = normalize_hist(hist)

    descriptors["D4"] = (hist, bin_edges)

    return descriptors


# Angle between 3 random vertices
def calc_A3(a, b, c):
    angle = compute_angle(a, b, c)

    return angle


# Distance between barycenter and random vertex, returns histogram vector and bin_edges
def calc_D1(center, vertices):
    D = []
    (xc, yc, zc) = center

    for x, y, z in vertices:
        D.append(np.linalg.norm([[x, y, z], [ xc, yc, zc]]))
        # D.append(spatial.distance.euclidean([x,y,z], center))
    # crate histogram with 10 bins from 0 - 1.0
    hist, bin_edges = np.histogram(np.array(D), bins = np.arange(0, 1.65, 0.15))
    hist = normalize_hist(hist)

    return (hist, bin_edges)


# Distance between 2 random vertices
def calc_D2(verticesOne, verticesTwo):

    D = []

    # Loop over both sets to create the vertice combinations, 250000 total
    for x1, y1, z1 in verticesOne:
        for x2, y2, z2 in verticesTwo:
            D.append(spatial.distance.cityblock([x1, y1, z1] , [ x2, y2, z2]))
    # Create histogram with 10 bins from 0 to 1.25 (as there where some values above 1)

    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0, 2.2, 0.2))
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
def calc_D4(p1, p2, p3, p4):
    p1_c = np.subtract(p1, p4)
    p2_c = np.subtract(p2, p4)
    p3_c = np.subtract(p3, p4)

    volume = np.abs(np.dot(p1_c, np.cross(p2_c, p3_c))) / 6
    return np.cbrt(volume)



# Standardize the non-histogram features
def standardize(features):
    V, A, C, BB, D, E = [], [], [], [], [], []
    for id, featuresList in features.items():
        V.append(featuresList["volume"])
        A.append(featuresList["area"])
        C.append(featuresList["compactness"])
        BB.append(featuresList["bbox_volume"])
        D.append(featuresList["diameter"])
        E.append(featuresList["eccentricity"])

    sdVals = save_standardization_vals(V, A, C, BB, D, E)

    for id, featuresList in features.items():
        features[id]["volume"] = (featuresList["volume"] - sdVals["V_mean"]) / sdVals["V_std"]
        features[id]["area"] = (featuresList["area"] - sdVals["A_mean"]) / sdVals["A_std"]
        features[id]["compactness"] = (featuresList["compactness"] - sdVals["C_mean"]) / sdVals["C_std"]
        features[id]["bbox_volume"] = (featuresList["bbox_volume"] - sdVals["BB_mean"]) / sdVals["BB_std"]
        features[id]["diameter"] = (featuresList["diameter"] - sdVals["D_mean"]) / sdVals["D_std"]
        features[id]["eccentricity"] = (featuresList["eccentricity"] - sdVals["E_mean"]) / sdVals["E_std"]

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
