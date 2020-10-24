import os
import numpy as np
import matplotlib.pyplot as plt

# from shape import Shape
# from visualize import visualize
# from normalize import normalize_data, normalize_shape
# from dataLoader import import_dataset, import_normalised_data
# from featureExtraction import *
# from featureMatching import *
# from utils import pick_file
# from similarity import *

from src.shape import Shape
from src.visualize import visualize
from src.normalize import normalize_data, normalize_shape
from src.dataLoader import import_dataset, import_normalised_data
from src.featureExtraction import *
from src.featureMatching import *
from src.utils import pick_file
from src.similarity import *


DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep 
SAVED_DATA = DATA_PATH + 'cache' + os.sep
NORMALIZED_DATA = SAVED_DATA + 'processed_data' + os.sep


if __name__ == "__main__":
    # --------------------------------------------------------
    # OFFLINE WORKFLOW - TO BE EXECUTED ONLY ONCE
    # --------------------------------------------------------

    FORCE_IMPORT = False
    shapes, labels, features = None, None, None

    if FORCE_IMPORT or len(os.listdir(NORMALIZED_DATA)) == 0:
        # Step 1: Importing data -----------------------------------------
        # Normalised shapes not present, importing and normalising dataset
        # Dividing import in batches
        batches = [f.path for f in os.scandir(DATA_SHAPES_PRICETON) if f.is_dir()]
        # batches = [0] # TODO: replace 0 with the batch to import for partial import, remove for final progr.
        for i, batch in enumerate(batches):

            print("##### Importing Batch " + str(i+1) + " of " + str(len(batches)))

            shapes, labels = import_dataset(batch)

            # Visualising shapes
            # visualize(shapes, labels)

            # Step 2: Normalising and remeshing shapes --------------------
            shapes, tot_verts, tot_faces = normalize_data(shapes)

            # Visualising normalised shapes
            # visualize(shapes, labels)

            # Step 3: Feature extraction ----------------------------------
            shapes, features = calculate_metrics(shapes, False)

            print("Progress:" + str(int((i+1/len(batches))*100)) + "%")
        


        features = np.load(SAVED_DATA + "features.npy", allow_pickle=True)
        
        # Standarize numeric features
        features = standardize(features.item())

        # calculate the weights for the similarity measure
        calculate_weights(features)
        

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

    # Step 4: Querying a shape ------------------------------------

    
    # Calculate weights if not done in the feature loading
    # calculate_weights(features)

    # Retrieving shape
    print("\n----------------------------------------------------")
    print("3D Shapes Search Engine")
    print("----------------------------------------------------")

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
    shape_features = standardize_single_shape(shape_features)

    # Calculate similarities
    print('Calculate similarities . . .')
    similarities = calc_distance(features, shape_features, shape.get_id())

    print(sorted(similarities.values()))
    print("Retrieving and showing similar shapes")
    shapes = load_similar(similarities, shape)
    
    visualize(shapes)
    

    # Step 5: Scalable querying -----------------------------------------------

    # Calculate nearest neighbors via ANN
    neighbors = k_means(shape_features, features)
    n_shapes_features, n_distances = ([n[0] for n in neighbors], [n[1] for n in neighbors])
    n_shapes_id = [n["id"] for n in n_shapes_features]

    # Retrieving shapes from database
    n_shapes = []
    for id in n_shapes_id:
        filename =NORMALIZED_DATA + "n" + str(id) + ".off"
        file = open(filename, 'r')
        verts, faces, n_verts, n_faces = read_off(file)
        mesh = trm.load_mesh(filename)

        n_shapes.append(Shape(verts, faces, mesh))

    visualize(n_shapes, labels=[])





