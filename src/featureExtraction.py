import math
import random
import warnings
import os
from scipy import spatial
# from utils import *
# from settings import Settings
from src.utils import *
from src.settings import Settings

s = Settings()


def calculate_metrics(shapes: [Shape]) -> ([Shape], {}):
    """
    For each shape of the batch, it calculates the shape metrics and adds them
    the the 2D features dictionary.
    ----------------------------
    Args:
        shapes (obj: 'list' of  obj: 'Shape'): The list of shapes

    Returns:
        shapes (obj: 'list' of obj: 'Shape'): The list of shapes
        features (obj: dict): The dictionary containing the feature metrics of each shape
    """

    print("Calculating all the object features . . .")

    # If some features are present, load them
    if os.path.isfile(s.SAVED_DATA + "features.npy"):
        features = np.load(s.SAVED_DATA + "features.npy", allow_pickle=True)
        try:
            features = features.item()
        except:
            warnings.warn_explicit("Error reading feature dictionary.", ImportWarning)
    else:
        features = {}

    # Calculate the metrics for each shape
    for shape in shapes:
        shape_id = shape.get_id()
        features[shape_id] = calculate_single_shape_metrics(shape)

    print("Done calculating the features.")

    print("Saving the features.")

    # Saving features to disk
    np.save(s.SAVED_DATA + "features.npy", features)

    return shapes, features


def calculate_single_shape_metrics(shape: Shape) -> {}:
    """
    It calculates the shape metrics and adds them
    the the 1D features dictionary of the shape.
    ----------------------------
    Args:
        shape (obj: 'Shape'): The shape

    Returns:
        features (obj: 'dict'): The dictionary containing the feature metrics of the shape
    """

    features = {}

    # calculate the default metrics
    features["volume"] = volume(shape)
    features["area"] = area(shape)
    features["bbox_volume"] = bbox_volume(shape)
    features["compactness"] = compactness(features["area"], features["bbox_volume"])
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


def volume(shape: Shape) -> float:
    """
    It calculates the volume of the shape
    ----------------------------
    Args:
        shape (obj: 'Shape'): The shape

    Returns:
        volume (float): The volume of the shape
    """
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


def area(shape: Shape) -> float:
    """
    It returns the area of the shape
    ----------------------------
    Args:
        shape (obj: 'Shape'): The shape

    Returns:
        area (float): The area of the shape
    """
    mesh = shape.get_mesh()
    return mesh.area


def compactness(area: float, bbox_volume: float) -> float:
    """
    It calculates the compactness of the shape using the shape's surface area
    and the axis-aligned bounding box volume
    ----------------------------
    Args:
        area (float): The area of the shape
        bbox_volume (float): The volume of the shape axis aligned bounding box

    Returns:
        compactness (float): The compactness of the shape
    """

    compactness = pow(area, 3) / (36 * math.pi * (pow(bbox_volume, 2)))
    return compactness


def bbox_volume(shape: Shape) -> float:
    """
    It calculates the volume of the axis-aligned bounding box surrounding the shape
    ----------------------------
    Args:
        shape (obj: 'Shape'): The shape

    Returns:
        bbox_volume (float): The volume of the axis-aligned bounding box surrounding the shape
    """
    bbox = shape.get_buinding_box()
    x = bbox.get_xmax() - bbox.get_xmin()
    y = bbox.get_ymax() - bbox.get_ymin()
    z = bbox.get_zmax() - bbox.get_zmin()

    volume = x * y * z

    return volume


def diameter(shape: Shape) -> float:
    """
    It calculates the diameter of the shape defined as the largest distance between
    two points on the surface of the shape
    ----------------------------
    Args:
        shape (obj: 'Shape'): The shape

    Returns:
        diameter (float): The diameter of the shape
    """
    max_distance = 0
    for x1, y1, z1 in shape.get_vertices():
        for x2, y2, z2 in shape.get_vertices():
            distance = np.sqrt(((x1-x2)**2)+((y1-y2)**2)+((z1-z2)**2))
            if distance > max_distance:
                max_distance = distance

    return max_distance


def eccentricity(shape: Shape) -> float:
    """
    It calculates the eccentricity of the shape defined as the ratio between the
    minor eigenvalue and the major eigenvalue
    ----------------------------
    Args:
        shape (obj: 'Shape'): The shape

    Returns:
        eccentricity (float): The eccentricity of the shape
    """
    eigenvalues, _ = calc_eigenvectors(shape.get_vertices())
    min_eigen = np.min(eigenvalues)
    max_eigen = np.max(eigenvalues)

    eccentricity = np.abs(max_eigen)/np.abs(min_eigen)

    return eccentricity


def calc_distributions(shape: Shape) -> {}:
    """
    It calculates the distribution A3,D1,D2,D3 of the shape by sampling several sets of
    random vertices, and computes the respective histograms of the distributions
    ----------------------------
    Args:
        shape (obj: 'Shape'): The shape

    Returns:
        descriptors (obj: 'dict' of obj: 'tuple' of (obj: 'list' of int, obj: 'list' of float)):
                            The dictionary containing the five distributions in the form of
                            histograms (number of occurrencies per bin, bins)
    """

    descriptors = {}
    center = shape.get_center()

    # Randomly sampling 5000 vertices
    vertices = random.choices(list(shape.get_vertices()), k=5000)

    # Computing D1 -----------------------
    print("Computing D1 distribution . . .")
    descriptors["D1"] = calc_D1(center, vertices)

    # Randomly sampling 1000 vertices
    vertices = random.choices(list(shape.get_vertices()), k=1000)

    # Computing D2 -----------------------
    print("Computing D2 distribution . . .")
    verticesOne = vertices[:]
    verticesTwo = vertices[500:]
    descriptors["D2"] = calc_D2(verticesOne, verticesTwo)

    # Randomly sampling 150 vertices
    vertices = random.sample(list(shape.get_vertices()), k=150)

    # Computing A3, D3, D4 ---------------
    verticesOne = vertices[:50]
    verticesTwo = vertices[50:100]
    verticesThree = vertices[100:]

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
                        # Computing D4 for the current combination
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


def calc_A3(a: [float], b: [float], c: [float]) -> float:
    """
    It calculates the angle between 3 random vertices
    ----------------------------
    Args:
        a (obj: 'list' of float): First vertex coordinates
        b (obj: 'list' of float): Second vertex coordinates
        c (obj: 'list' of float): Third vertex coordinates

    Returns:
        angle (float): The amplitude of the angle between the tree vertices
    """
    angle = compute_angle(a, b, c)

    return angle


def calc_D1(center: [float], vertices: [[float]]) -> ([int], [float]):
    """
    It calculates the distribution of distances between barycenter and each vertex in the list
    of sampled vertices
    ----------------------------
    Args:
        center (obj: 'list' of float): The center of the shape
        vertices (obj: 'list' of obj: 'list' of float): The list of samples vertices coordinates

    Returns:
        (hist, bin_edges) (obj: 'tuple' of (obj: 'list' of int, obj: 'list' of float)):
                            The histogram of the distribution of distances between barycenter and vertices
    """

    D = []
    (xc, yc, zc) = center

    for x, y, z in vertices:
        D.append(np.linalg.norm([[x, y, z], [ xc, yc, zc]]))

    hist, bin_edges = np.histogram(np.array(D), bins=np.arange(0, 1.65, 0.15))
    hist = normalize_hist(hist)

    return (hist, bin_edges)


def calc_D2(verticesOne: [[float]], verticesTwo: [[float]]) -> ([int], [float]):
    """
    It calculates the distribution of distances between each combinatorial pair of vertices in the lists
    of sampled vertices
    ----------------------------
    Args:
        verticesOne (obj: 'list' of obj: 'list' of float): The center of the shape
        verticesTwo (obj: 'list' of obj: 'list' of float): The list of samples vertices coordinates

    Returns:
        (hist, bin_edges) (obj: 'tuple' of (obj: 'list' of int, obj: 'list' of float)):
                            The histogram of the distribution of distances between the combinatorial pair of vertices
    """

    D = []

    # Loop over both sets to create the vertices combinations
    for x1, y1, z1 in verticesOne:
        for x2, y2, z2 in verticesTwo:
            D.append(spatial.distance.cityblock([x1, y1, z1] , [ x2, y2, z2]))

    hist, bin_edges = np.histogram(np.array(D), bins= np.arange(0, 2.2, 0.2))
    hist = normalize_hist(hist)

    return hist, bin_edges


def calc_D3(a: [float], b: [float], c: [float]) -> float:
    """
    It calculates the square root of the area of a triangle given 3 random vertices
    ----------------------------
    Args:
        a (obj: 'list' of float): First vertex coordinates
        b (obj: 'list' of float): Second vertex coordinates
        c (obj: 'list' of float): Third vertex coordinates

    Returns:
        sqrt_area (float): The square root of area of triangle
    """
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
    sqrt_area = np.sqrt(area)

    return sqrt_area


def calc_D4(a: [float], b: [float], c: [float], d: [float]) -> float:
    """
    It calculates the cube root of the volume of a tetrahedron formed by 4 random vertices
    ----------------------------
    Args:
        a (obj: 'list' of float): First vertex coordinates
        b (obj: 'list' of float): Second vertex coordinates
        c (obj: 'list' of float): Third vertex coordinates
        d (obj: 'list' of float): Fourth vertex coordinates

    Returns:
        cbrt_volume (float): The cube root of the volume of the tetrahedron
    """
    p1_c = np.subtract(a, d)
    p2_c = np.subtract(b, d)
    p3_c = np.subtract(c, d)

    volume = np.abs(np.dot(p1_c, np.cross(p2_c, p3_c))) / 6
    cbrt_volume = np.cbrt(volume)

    return cbrt_volume


def standardize(features: {}) -> {}:
    """
    It standardizes the features of all non-histogram features
    ----------------------------
    Args:
        features (obj: 'dict'): The dictionary of features

    Returns:
        features (obj: 'dict'): The dictionary of all features after standardisation
    """

    # Extracting single set of features to compute standardisation values
    V, A, C, BB, D, E = [], [], [], [], [], []
    for id, featuresList in features.items():
        V.append(featuresList["volume"])
        A.append(featuresList["area"])
        C.append(featuresList["compactness"])
        BB.append(featuresList["bbox_volume"])
        D.append(featuresList["diameter"])
        E.append(featuresList["eccentricity"])

    sdVals = save_standardization_vals(V, A, C, BB, D, E)

    # Standardizing non-histogram features
    for id, featuresList in features.items():
        features[id]["volume"] = (featuresList["volume"] - sdVals["V_mean"]) / sdVals["V_std"]
        features[id]["area"] = (featuresList["area"] - sdVals["A_mean"]) / sdVals["A_std"]
        features[id]["compactness"] = (featuresList["compactness"] - sdVals["C_mean"]) / sdVals["C_std"]
        features[id]["bbox_volume"] = (featuresList["bbox_volume"] - sdVals["BB_mean"]) / sdVals["BB_std"]
        features[id]["diameter"] = (featuresList["diameter"] - sdVals["D_mean"]) / sdVals["D_std"]
        features[id]["eccentricity"] = (featuresList["eccentricity"] - sdVals["E_mean"]) / sdVals["E_std"]

    # Saving standardized features and standardization values in cache
    np.save(s.SAVED_DATA + "features.npy", features)
    np.save(s.SAVED_DATA + "standardization_values.npy", sdVals)

    return features


def save_standardization_vals(V: [float], A: [float], C: [float], BB: [float], D: [float], E: [float]):
    """
     Computes standardisation values (mean and standard deviations) for each set of features
     ----------------------------
     Args:
         V (obj: 'list' of float): Set of volume features
         A (obj: 'list' of float): Set of area features
         C (obj: 'list' of float): Set of compactness features
         BB (obj: 'list' of float): Set of bounding-box volume features
         D (obj: 'list' of float): Set of diameter features
         E (obj: 'list' of float): Set of eccentricity features

     Returns:
         standardVals (obj: 'dict'): The dictionary of all standardization values
     """
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
