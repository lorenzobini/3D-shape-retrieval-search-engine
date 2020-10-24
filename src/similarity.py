import os
import numpy as np
import trimesh as trm
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

from shape import Shape
from utils import read_off


DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep


def calculate_weights(features):
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
    weights["w_c"] = 1/np.std(d_a)
    weights["w_bb"] = 1/np.std(d_bb)
    weights["w_d"] = 1/np.std(d_d)
    weights["w_e"] = 1/np.std(d_e)
    
    weights["w_A3"] = 1/np.std(d_a3)
    weights["w_D1"] = 1/np.std(d_d1)
    weights["w_D2"] = 1/np.std(d_d2)
    weights["w_D3"] = 1/np.std(d_d3)
    weights["w_D4"] = 1/np.std(d_d4)

    print(weights)

    np.save(SAVED_DATA + "distance_weights.npy", weights)  
    return    


        
def standardize_single_shape(shape_features):
    sdVals = np.load(SAVED_DATA + "standardization_values.npy", allow_pickle=True).item()
    shape_features["volume"] = (shape_features["volume"]-sdVals["V_mean"])/sdVals["V_std"]
    shape_features["area"] = (shape_features["area"]-sdVals["A_mean"])/sdVals["A_std"]
    shape_features["compactness"] = (shape_features["compactness"]-sdVals["C_mean"])/sdVals["C_std"]
    shape_features["bbox_volume"] = (shape_features["bbox_volume"]-sdVals["BB_mean"])/sdVals["BB_std"]
    shape_features["diameter"] = (shape_features["diameter"] - sdVals["D_mean"])/sdVals["D_std"]
    shape_features["eccentricity"] = (shape_features["eccentricity"] - sdVals["E_mean"])/sdVals["E_std"]
    return shape_features


def calc_distance(features, shape_features, shape_id):
    similarities = {} # key: shape ID, value: distance
    similarity = float(0)
    weights = np.load(SAVED_DATA + "distance_weights.npy", allow_pickle=True).item()
    for id, featuresList in features.items():
        # Distance is the sqaure root of the sum of squared differences
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

        similarity = 0.1*dist_v + 0.1*dist_a + 0.1*dist_c + weights["w_bb"]*dist_bb + weights["w_d"]*dist_d + \
            weights["w_e"]*dist_e + weights["w_A3"]*dist_A3 + weights["w_D1"]*dist_D1 + weights["w_D2"]*dist_D2 + weights["w_D3"]*dist_D3 + weights["w_D4"]*dist_D4
        similarities[id] = similarity
    

    return similarities

def load_similar(similarities, shape):
    DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep + 'cache' + os.sep + 'processed_data' + os.sep
    
    ordSimilarities = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1])}
    shapes = [shape]
    
    for i, id in enumerate(ordSimilarities.keys()):
        if i == 10:
            return shapes
        
        name = "n" + str(id) + ".off"
        filename = find(name, DATA_PATH)
        if str(filename).endswith(".off"):
            file = open(filename, 'r')
            verts, faces, _, _ = read_off(file)
            mesh = trm.load_mesh(filename)

            shape = Shape(verts, faces, mesh)
            shape.set_id(id)

        
        shapes.append(shape)


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

