import os
import numpy as np
import matplotlib.pyplot as plt

from shape import Shape
from visualize import visualize
from normalize import normalize_data, normalize_shape
from dataLoader import import_dataset, import_normalised_data
from featureExtraction import *
from featureMatching import *
from utils import pick_file
from settings import Settings

# from src.shape import Shape
# from src.visualize import visualize
# from src.normalize import normalize_data, normalize_shape
# from src.dataLoader import import_dataset, import_normalised_data
# from src.featureExtraction import *
# from src.featureMatching import *
# from src.utils import pick_file
# from src.settings import Settings

s = Settings()

try:
    features = np.load(s.SAVED_DATA + "features.npy", allow_pickle=True).item()
    labels = np.load(s.SAVED_DATA + "labels.npy", allow_pickle=True).item()
except:
    pass

# Calculate number of elements per class label
label_counts = {}
feature_ids = [str(x) for x in features.keys()]
for id, label in labels.items():
    if id in feature_ids:
        try:
            label_counts[label] += 1
        except:
            label_counts[label] = 1


# Code for calculating number of elements in class histogram.
counts = [int(x) for x in label_counts.values()]
plt.hist(counts, bins='auto')
plt.title("Histogram showing the model counts")
plt.xlabel("Number of models")
plt.ylabel("Times a class has x models")
plt.show()

# Initialize parameters
database_size = len(features.keys())
class_metrics = {}
precision, recall, accuracy = 0,0, 0

print("Calculating accuracy, precision and recall for the entire database . . .")
# Loop over all the features in the database 
for id, featureList in features.items():
    TP, FP, FN, TN = 0,0,0,0
    correct_label = labels[str(id)]
    if correct_label[0] == 28:
        continue
    
    # Check if label already in class metrics dict
    if correct_label not in class_metrics.keys():
        class_metrics[correct_label] = {}

    if s.USE_RNN:
        # Calculate nearest neighbors via ANN and R-Nearest Neighbors
        neighbors = r_neighbors(featureList, features, r = 100.0)
        similarModels, n_distances = neighbors[0][1:], neighbors[1][1:]
    
    if s.USE_CUSTOM_DISTANCE: 
        # Calculate the similar models using our distance function
        similarities = calc_distance(features, featureList)
        similarModels = {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1])}
        similarModels = similarModels.keys()
    
    # Calcualte TP, FP for shape 
    for i, id_sim in enumerate(similarModels):
        if i == label_counts[correct_label]: # Query shape is equal to the number of elements in the correct class
            break
        if id != id_sim:
            label = labels[str(id_sim)]
            if label[1] == correct_label[1]:
                TP += 1
            elif label[1] != correct_label[1]:
                FP += 1
    
    # Number of elements in the class minus the query
    d = label_counts[correct_label]-1
   
    # Compute the false and true negatives    
    FN = d - TP
    TN = (database_size- d) - FP

    
    # Calculte precision, recall and accuracy for current shape
    a = (TP+TN)/database_size
    p = TP/(TP+FP)
    r = (TP/(TP+FN))

    # store in dict for metric per class computation
    try:
        class_metrics[correct_label]["accuracy"] += a
        class_metrics[correct_label]["precision"] += p
        class_metrics[correct_label]["recall"] += r
    except:
        class_metrics[correct_label]["accuracy"] = a
        class_metrics[correct_label]["precision"] = p
        class_metrics[correct_label]["recall"] = r

    # Overall measures
    accuracy += a
    precision += p
    recall += r
    
print("Overall database performance measures: ")
print("Accuracy: ", accuracy/database_size)
print("Precision: ", precision/database_size)
print("Recall: ", recall/database_size)

print("Performance measures per class are: ")
print("Class label:            Accuracy:                 Precision:               Recall:                 # in database:")
avg_num_classe = 0
for classlabel, measures in class_metrics.items():
    c_acc = round(measures["accuracy"]/label_counts[classlabel], 3)
    c_precision = round(measures["precision"]/label_counts[classlabel], 3)
    c_recall = round(measures["recall"]/label_counts[classlabel], 3)
    print(classlabel[1],"  ", c_acc,"   ",  c_precision, "  ", c_recall, "  ", label_counts[classlabel])
    avg_num_classe += label_counts[classlabel]
