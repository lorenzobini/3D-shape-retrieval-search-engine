import math
import numpy as np
import os
from annoy import AnnoyIndex

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep

# TODO: how to do the same with r-means?

def k_means(shape_features, db_features):
    if not os.path.isfile(SAVED_DATA + 'clusters.ann'):
        ann = clustering(db_features)
    else:
        ann = AnnoyIndex(db_features, 'euclidean')
        ann.load(SAVED_DATA + 'clusters.ann')

    neighbors = ann.get_nns_by_item(shape_features, 10, include_distances=True)

    return neighbors


def clustering(features):
    ann = AnnoyIndex(features, 'euclidean')

    ann.build(10)  # 10 trees
    ann.save(SAVED_DATA + 'clusters.ann')

    return ann


def calc_distance(features, shape_features, shape_id):

    similarities = {} # key: shape ID, value: distance
    similarity = float(0)

    for featuresList in features:
        id = featuresList["id"]
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
