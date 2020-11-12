import numpy as np
import os
from annoy import AnnoyIndex
import trimesh as trm
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

# from shape import Shape
# from utils import flatten_features_array, read_off
# from settings import Settings
from src.utils import flatten_features_array, read_off
from src.shape import Shape
from src.settings import Settings

s = Settings()


def calculate_weights(features: {}):
    """
    It determines the weights of the single features for distance computation
    The features are compared in pairs to determine the euclidean distance, for simple features,
    or the Wassertein distance, for distributions. The weights are computed as 1 over the standard
    deviation of the respective set of distances.
    The weights are then saved to cache.
    ----------------------------
    Args:
        features (obj: 'dict'): The dictionary containing the feature metrics of each shape
    """
    d_v, d_a, d_c, d_bb, d_d, d_e, d_a3, d_d1, d_d2, d_d3, d_d4 = [],[],[],[],[],[],[],[],[],[],[]
    for i in range(0, len(features.keys())):
        featureList1 = list(features.values())[i]
        for j in range(i+1, len(features.keys())):
            featureList2 = list(features.values())[j]

            d_v.append(distance.euclidean(featureList1['volume'], featureList2['volume']))
            d_a.append(distance.euclidean(featureList1['area'], featureList2['area']))
            d_c.append(distance.euclidean(featureList1['compactness'], featureList2['compactness']))
            d_bb.append(distance.euclidean(featureList1['bbox_volume'], featureList2['bbox_volume']))
            d_d.append(distance.euclidean(featureList1['diameter'], featureList2['diameter']))
            d_e.append(distance.euclidean(featureList1['eccentricity'], featureList2['eccentricity']))

            d_a3.append(wasserstein_distance(featureList1['A3'][0], featureList2['A3'][0]))
            d_d1.append(wasserstein_distance(featureList1['D1'][0], featureList2['D1'][0]))
            d_d2.append(wasserstein_distance(featureList1['D2'][0], featureList2['D2'][0]))
            d_d3.append(wasserstein_distance(featureList1['D3'][0], featureList2['D3'][0]))
            d_d4.append(wasserstein_distance(featureList1['D4'][0], featureList2['D4'][0]))
        
    
    weights = {}
    weights["w_v"] = 1/np.std(d_v)
    weights["w_a"] = 1/np.std(d_a)
    weights["w_c"] = 1/np.std(d_c)
    weights["w_bb"] = 1/np.std(d_bb)
    weights["w_d"] = 1/np.std(d_d)
    weights["w_e"] = 1/np.std(d_e)
    
    weights["w_A3"] = 1/np.std(d_a3)
    weights["w_D1"] = 1/np.std(d_d1)
    weights["w_D2"] = 1/np.std(d_d2)
    weights["w_D3"] = 1/np.std(d_d3)
    weights["w_D4"] = 1/np.std(d_d4)

    np.save(s.SAVED_DATA + "distance_weights.npy", weights)


def standardize_single_shape(shape_features: {}) -> {}:
    """
    It performs standardisation over all simple features of a shape.
    ----------------------------
    Args:
        shape_features (obj: 'dict'): The dictionary containing the feature metrics of the shape

    Returns:
        shape_features (obj: 'dict'): The dictionary containing the feature metrics of the shape
                                      after standardisation
    """

    sd_vals = np.load(s.SAVED_DATA + "standardization_values.npy", allow_pickle=True).item()

    shape_features["volume"] = (shape_features["volume"]-sd_vals["V_mean"])/sd_vals["V_std"]
    shape_features["area"] = (shape_features["area"]-sd_vals["A_mean"])/sd_vals["A_std"]
    shape_features["compactness"] = (shape_features["compactness"]-sd_vals["C_mean"])/sd_vals["C_std"]
    shape_features["bbox_volume"] = (shape_features["bbox_volume"]-sd_vals["BB_mean"])/sd_vals["BB_std"]
    shape_features["diameter"] = (shape_features["diameter"] - sd_vals["D_mean"])/sd_vals["D_std"]
    shape_features["eccentricity"] = (shape_features["eccentricity"] - sd_vals["E_mean"])/sd_vals["E_std"]

    return shape_features


def calc_distance(features: {}, shape_features: {}) -> {}:
    """
    It determines the closest shape to the query shape by computing a custom distance function
    between the features of dataset's shape and the features of the query shape.
    ----------------------------
    Args:
        features (obj: 'dict): The dictionary containing the feature metrics of the shapes
        shape_features (obj: 'dict'): The dictionary containing the feature metrics of the shape

    Returns:
        similarities (obj: 'dict'): The dictionary containing the closest shapes (key) and the respective
                                    distance to the query shape (value)
    """
    similarities = {}
    weights = np.load(s.SAVED_DATA + "distance_weights.npy", allow_pickle=True).item()

    for id, featuresList in features.items():
        # Distance is the square root of the sum of squared differences
        dist_v = distance.euclidean(featuresList['volume'], shape_features.get('volume'))
        dist_a = distance.euclidean(featuresList['area'], shape_features.get('area'))
        dist_c = distance.euclidean(featuresList['compactness'], shape_features.get('compactness'))
        dist_bb = distance.euclidean(featuresList['bbox_volume'], shape_features.get('bbox_volume'))
        dist_d = distance.euclidean(featuresList['diameter'], shape_features.get('diameter'))
        dist_e = distance.euclidean(featuresList['eccentricity'], shape_features.get('eccentricity'))

        dist_A3 = wasserstein_distance(featuresList['A3'][0], shape_features.get('A3')[0])
        dist_D1 = wasserstein_distance(featuresList['D1'][0], shape_features.get('D1')[0])
        dist_D2 = wasserstein_distance(featuresList['D2'][0], shape_features.get('D2')[0])
        dist_D3 = wasserstein_distance(featuresList['D3'][0], shape_features.get('D3')[0])
        dist_D4 = wasserstein_distance(featuresList['D4'][0], shape_features.get('D4')[0])

        similarity = weights["w_v"]*dist_v + \
            weights["w_a"]*dist_a + \
            weights["w_c"]*dist_c + \
            weights["w_bb"]*dist_bb + \
            weights["w_d"]*dist_d + \
            weights["w_e"]*dist_e + \
            weights["w_A3"]*dist_A3 + \
            weights["w_D1"]*dist_D1 + \
            weights["w_D2"]*dist_D2 + \
            weights["w_D3"]*dist_D3 + \
            weights["w_D4"]*dist_D4
         
        similarities[id] = similarity

    return similarities


def load_similar(similarities: {}, shape: Shape) -> [Shape]:
    """
    It loads the most similar shapes to the query shape from memory
    ----------------------------
    Args:
        similarities (obj: 'dict'): The dictionary containing the closest shapes (key) and the respective
                                    distance to the query shape (value)
        shape (obj: 'Shape'): The query shape

    Returns:
        similarities (obj: 'list' of obj: 'Shape'): The list of retrieved shapes
    """
    
    ord_similarities = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1])}
    shapes = [shape]
    
    for i, id in enumerate(ord_similarities.keys()):
        if i == 10:
            return shapes
        
        name = "n" + str(id) + ".off"
        filename = find(name, s.DATA_PATH)
        if str(filename).endswith(".off"):
            file = open(filename, 'r')
            verts, faces, _, _ = read_off(file)
            mesh = trm.load_mesh(filename)

            shape = Shape(verts, faces, mesh)
            shape.set_id(id)

        shapes.append(shape)

    return shapes


def find(name: str, path: str) -> str:
    """
    Determine if a file is stored in any subfolder within a specified path
    ----------------------------
    Args:
        name (str): The name of the fle
        path (str): The global path
    Returns:
        (str): The complete global path of the specified file
               if the file exists in any subfolder within the specified path
        None: otherwise
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)
        else:
            return None


def k_neighbors(shape_features: {}, db_features: {}, k=s.KNN_SIZE) -> []:
    """
    It determines the closest shape to the query shape by computing K-Nearest Neighbors on a
    N-dimensional Approximate Nearest Neighbors feature mapping.
    ----------------------------
    Args:
        shape_features (obj: 'dict'): The dictionary containing the feature metrics of the shape
        db_features (obj: 'dict): The dictionary containing the feature metrics of the shapes
        k (int): The number of neighbors to return, the default value specified in Settings

    Returns:
        neighbors (obj: 'dict'): The dictionary containing the closest shapes (key) and the respective
                                    distance to the query shape (value)
    """
    ann = AnnoyIndex(56, 'euclidean')  # 56 features
    for id, featureList in db_features.items():
        features_flatten = flatten_features_array(featureList)
        ann.add_item(id, features_flatten)

    shape_features_flat = flatten_features_array(shape_features)

    # To get the neighbors, it is necessary to add the new item to the mapping first
    shape_id = ann.get_n_items()
    ann.add_item(shape_id, shape_features_flat)

    ann.build(s.CATEGORIES)

    neighbors = ann.get_nns_by_item(shape_id, k, include_distances=True)

    return neighbors


def r_neighbors(shape_features: {}, db_features: {}, r=s.RNN_RANGE) -> []:
    """
    It determines the closest shape to the query shape by computing R-Nearest Neighbors on a
    N-dimensional Approximate Nearest Neighbors feature mapping.
    ----------------------------
    Args:
        shape_features (obj: 'dict'): The dictionary containing the feature metrics of the shape
        db_features (obj: 'dict): The dictionary containing the feature metrics of the shapes
        r (int): The distance range, the default value specified in Settings

    Returns:
        neighbors (obj: 'dict'): The dictionary containing the closest shapes (key) and the respective
                                    distance to the query shape (value)
    """
    ann = AnnoyIndex(56, 'euclidean')  # 56 features
    for id, featureList in db_features.items():
        features_flatten = flatten_features_array(featureList)
        ann.add_item(id, features_flatten)

    shape_features_flat = flatten_features_array(shape_features)

    # To get the neighbors, it is necessary to add the new item to the mapping first
    shape_id = ann.get_n_items()
    ann.add_item(shape_id, shape_features_flat)

    ann.build(s.CATEGORIES)

    neighbors = ann.get_nns_by_item(shape_id, 200, include_distances=True)


    range_neighbors = ([], [])
    for i, distance in enumerate(neighbors[1]):
        if distance < r:
            range_neighbors[0].append(neighbors[0][i])
            range_neighbors[1].append(distance)

    return range_neighbors

