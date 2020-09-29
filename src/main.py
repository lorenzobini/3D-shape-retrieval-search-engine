from src.dataLoader import import_dataset, import_normalised_data
from collections import defaultdict

# from shape import Shape
# from visualize import visualize
# from normalize import normalize_data

from src.shape import Shape
from src.visualize import visualize
from src.normalize import normalize_data

import os

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep
NORMALIZED_DATA = SAVED_DATA + 'processed_data' + os.sep


def main():
    # --------------------------------------------------------
    # OFFLINE WORKFLOW - TO BE EXECUTED ONLY ONCE
    # --------------------------------------------------------

    # Step 1: Importing data ---------------------------------
    FORCE_IMPORT = True
    if FORCE_IMPORT or len(os.listdir(NORMALIZED_DATA)) == 0:
        # Normalised shapes not present, importing and normalising dataset
        shapes, labels = import_dataset()

        # Visualising shapes
        visualize(shapes, labels)

        # Step 2: Normalising and remeshing shapes --------------
        shapes, tot_verts, tot_faces = normalize_data(shapes)


        # Visualising normalised shapes
        visualize(shapes, labels)

    else:
        # Normalised shapes in cache, loading them directly
        shapes, labels = import_normalised_data()

        # Visualising normalised shapes
        visualize(shapes, labels)

    # Step 3: Feature extraction -------------------------------
    surface_area(shapes)

    # ---------------------------------------------------------
    # ONLINE WORKFLOW - QUERYING A SHAPE AND DISPLAYING RESULTS
    # ---------------------------------------------------------
    pass


main()
