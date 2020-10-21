import os
import numpy as np

# from shape import Shape
# from visualize import visualize
# from normalize import normalize_data, normalize_shape
# from dataLoader import import_dataset, import_normalised_data
# from features import *
# from utils import standardize, calc_distance, pick_file

from src.shape import Shape
from src.visualize import visualize
from src.normalize import normalize_data, normalize_shape
from src.dataLoader import import_dataset, import_normalised_data
from src.featureExtraction import *
from src.utils import pick_file


DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep

# TODO: to import the entire dataset remove the '0' and the redundant os.sep, REMOVE FOR FINAL PROGRAM
DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep # + 'test' + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep
NORMALIZED_DATA = SAVED_DATA + 'processed_data' + os.sep


def main():
    # --------------------------------------------------------
    # OFFLINE WORKFLOW - TO BE EXECUTED ONLY ONCE
    # --------------------------------------------------------

    # Step 1: Importing data ---------------------------------
    FORCE_IMPORT = False
    shapes, labels, features = None, None, None

    if FORCE_IMPORT or len(os.listdir(NORMALIZED_DATA)) == 0:
        # Normalised shapes not present, importing and normalising dataset
        # Dividing import in batches
        batches = [f.path for f in os.scandir(DATA_SHAPES_PRICETON) if f.is_dir()]
        # batches = [0] # TODO: replace 0 with the batch to import for partial import, remove for final progr.
        for i, batch in enumerate(batches):

            print("##### Importing Batch " + str(i+1) + " of " + str(len(batches)))

            shapes, labels = import_dataset(batch)

            # Visualising shapes
            # visualize(shapes, labels)

            # Step 2: Normalising and remeshing shapes --------------
            shapes, tot_verts, tot_faces = normalize_data(shapes)

            # Visualising normalised shapes
            # visualize(shapes, labels)

            # Step 3: Feature extraction -------------------------------
            shapes, features = calculate_metrics(shapes, False)

            print("Progress:" + str(int((i+1/len(batches))*100)) + "%")
        


        features = np.load(SAVED_DATA + "features.npy", allow_pickle=True)
        
        # Standarize numeric features
        features = standardize(features.item())

        print("Import and normalisation processes completed.")
        print("Features extracted.")

        del shapes
        del labels
        del tot_faces
        del tot_verts
        del features

    # --------------------------------------------------------
    # ONLINE WORKFLOW - QUERYING AND DISPLAYING RESULTS
    # --------------------------------------------------------

    try:
        features = np.load(SAVED_DATA + "features.npy", allow_pickle=True)
        features = features.item()
    except:
        pass

    # Retrieving shape

    print("\n----------------------------------------------------")
    print("3D Shapes Search Engine")
    print("\n----------------------------------------------------")

    while True:
        print("Select a shape to search for similar ones. Only OFF and PLY formats are supported. \n")
        try:
            shape = pick_file()
            break
        except FileNotFoundError:
            print("File not found. Try again.\n")
            continue
        except FileExistsError:
            print("Format not supported. Please select an OFF or PLY file.\n")
            continue

    # Normalising shape

    print('Normalising query shape . . . ')
    shape, new_n_verts, new_n_faces = normalize_shape(shape)

    # Calculating features for the shape
    print('Calculating features for query shape . . .')
    shape_features = calculate_single_shape_metrics(shape)

    # Calculate similarities
    print('Calculate similarities . . .')
    similarities = calc_distance(features, shape_features, shape.get_id())

    print(sorted(similarities.values()))
    
    #print({k: v for k, v in sorted(similarities.items(), key=lambda item: item[1])})



main()
