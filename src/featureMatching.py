import math
import numpy as np
import os
from annoy import AnnoyIndex

# from utils import flatten
from src.utils import flatten_features_array

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep


def k_neighbors(shape_features, db_features, k=11):
    ann = AnnoyIndex(56, 'euclidean')  # 16 features
    for id, featureList in db_features.items():
        features_flatten = flatten_features_array(featureList)
        ann.add_item(id, features_flatten)

    shape_features_flat = flatten_features_array(shape_features)

    # To get the neighbors, it is necessary to add the new item to the mapping first
    shape_id = ann.get_n_items()
    ann.add_item(shape_id, shape_features_flat)

    ann.build(42)  # 42 categories TODO: if we don't use coarse1, replace with new number of categories

    neighbors = ann.get_nns_by_item(shape_id, k, include_distances=True)

    return neighbors


def r_neighbors(shape_features, db_features, r=0.1):  # TODO: find a suitable range
    ann = AnnoyIndex(56, 'euclidean')  # 16 features
    for id, featureList in db_features.items():
        features_flatten = flatten_features_array(featureList)
        ann.add_item(id, features_flatten)

    shape_features_flat = flatten_features_array(shape_features)

    # To get the neighbors, it is necessary to add the new item to the mapping first
    shape_id = ann.get_n_items()
    ann.add_item(shape_id, shape_features_flat)

    ann.build(42)  # 42 categories TODO: if we don't use coarse1, replace with new number of categories

    neighbors = ann.get_nns_by_item(shape_id, 100, include_distances=True)

    range_neighbors = []
    for neighbor, distance in neighbors:
        if distance < r:
            range_neighbors.append((neighbor, distance))

    return range_neighbors


def calc_distance(features, shape_features, shape_id):

    similarities = {} # key: shape ID, value: distance
    similarity = float(0)

    for id, featuresList in features.items():
        # Distance is the sqaure root of the sum of squared differences
        dist_v = pow(featuresList["volume"] - shape_features.get("volume"), 2)
        dist_a = pow(featuresList["area"] - shape_features.get("area"), 2)
        dist_c = pow(featuresList["compactness"] - shape_features.get("compactness"), 2)
        dist_bb = pow(featuresList["bbox_volume"] - shape_features.get("bbox_volume"), 2)
        dist_d = pow(featuresList["diameter"] - shape_features.get("diameter"), 2)
        dist_e = pow(featuresList["eccentricity"] - shape_features.get("eccentricity"), 2)

        dist_A3 = np.sum(pow(np.subtract(featuresList["A3"][0], shape_features.get("A3")[0]), 2))
        dist_D1 = np.sum(pow(np.subtract(featuresList["D1"][0], shape_features.get("D1")[0]), 2))
        dist_D2 = np.sum(pow(np.subtract(featuresList["D2"][0], shape_features.get("D2")[0]), 2))
        dist_D3 = np.sum(pow(np.subtract(featuresList["D3"][0], shape_features.get("D3")[0]), 2))
        dist_D4 = np.sum(pow(np.subtract(featuresList["D4"][0], shape_features.get("D4")[0]), 2))

        similarity = math.sqrt \
            (dist_v + dist_a + dist_c + dist_bb + dist_d + dist_e + dist_A3 + dist_D1 + dist_D2 + dist_D3 + dist_D4)
        similarities[id] = similarity

    print(similarities)

    return similarities
