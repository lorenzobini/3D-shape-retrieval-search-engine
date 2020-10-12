from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt

# from shape import Shape
# from visualize import visualize
# from normalize import normalize_data
# from dataLoader import import_dataset, import_normalised_data
# from features   import calculate_metrics


from src.shape import Shape
from src.visualize import visualize
from src.normalize import normalize_data
from src.dataLoader import import_dataset, import_normalised_data
from src.features import *

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep

# TODO: to import the entire dataset remove the '0' and the redundant os.sep, REMOVE FOR FINAL PROGRAM
DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep # + '0' + os.sep + 'm1' + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep
NORMALIZED_DATA = SAVED_DATA + 'processed_data' + os.sep


def main():
    # --------------------------------------------------------
    # OFFLINE WORKFLOW - TO BE EXECUTED ONLY ONCE
    # --------------------------------------------------------

    # Step 1: Importing data ---------------------------------
    FORCE_IMPORT = True
    shapes, labels, features = None, None, None

    if FORCE_IMPORT or len(os.listdir(NORMALIZED_DATA)) == 0:
        # Normalised shapes not present, importing and normalising dataset
        # Dividing import in batches
        batches = [f.path for f in os.scandir(DATA_SHAPES_PRICETON) if f.is_dir()]
        # batches = [batches[0]] # TODO: replace 0 with the batch to import for partial import, remove for final progr.
        for i, batch in enumerate(batches):

            shapes, labels = import_dataset(batch)

            # Visualising shapes
            # visualize(shapes, labels)

            # Step 2: Normalising and remeshing shapes --------------
            shapes, tot_verts, tot_faces = normalize_data(shapes)

            # Visualising normalised shapes
            # visualize(shapes, labels)

            # Step 3: Feature extraction -------------------------------
            shapes, features = calculate_metrics(shapes)

            print("Progress:" + str((i/len(batches))*100) + "%")


        print("Import and normalisation processes completed.")
        print("Features extracted.")

    # Computing statistics
    feature_statistics(features, labels)

    np.delete(shapes)
    np.delete(labels)
    np.delete(tot_faces)
    np.delete(tot_verts)
    np.delete(features)


    # When saved numpy.load returns a numpy.ndarry so this handles that
    try:
        features = features.item()
        labels = labels.item()
    except:
        pass



    # ---------------------------------------------------------
    # ONLINE WORKFLOW - QUERYING A SHAPE AND DISPLAYING RESULTS
    # ---------------------------------------------------------
    features = np.load(SAVED_DATA + "features.npy")

    pass


main()
