from src.dataLoader import import_dataset, import_normalised_data
from collections import defaultdict
from src.visualize import visualize
from src.normalize import normalize_data
from src.shape import Shape
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
        print('Normalising shapes . . .')
        shapes, tot_verts, tot_faces = normalize_data(shapes)
        print("Shapes normalised succesfully.")

        # Visualising normalised shapes
        visualize(shapes, labels)

    else:
        # Normalised shapes in cache, loading them directly
        shapes, labels = import_normalised_data()

        # Visualising normalised shapes
        visualize(shapes, labels)

    # Step 3: Feature extraction -------------------------------
    pass

    # --------------------------------------------------------
    # ONLINE WORKFLOW - QUERYING SHAPES AND DISPLAYING RESULTS
    # --------------------------------------------------------
    pass


main()
