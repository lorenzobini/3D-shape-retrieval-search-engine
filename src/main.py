from visualize import visualize
from normalize import normalize_data, normalize_shape
from dataLoader import import_dataset, import_normalised_data
from featureExtraction import *
from featureMatching import *
from utils import pick_file, tsne_plot, convert_dict_to_arr

# from src.visualize import visualize
# from src.normalize import normalize_data, normalize_shape
# from src.dataLoader import import_dataset
# from src.featureExtraction import *
# from src.featureMatching import *
# from src.utils import pick_file
# from src.settings import Settings


s = Settings()


if __name__ == "__main__":
    # --------------------------------------------------------
    # OFFLINE WORKFLOW - TO BE EXECUTED ONLY ONCE
    # --------------------------------------------------------

    if s.FORCE_IMPORT or len(os.listdir(s.NORMALIZED_DATA)) == 0:
        # Step 1: Importing data -----------------------------------------
        # Normalised shapes not present, importing and normalising dataset
        # Dividing import in batches
        batches = [f.path for f in os.scandir(s.DATA_SHAPES_PRICETON) if f.is_dir()]
        for i, batch in enumerate(batches):

            print("##### Importing Batch " + str(i+1) + " of " + str(len(batches)))

            shapes, labels = import_dataset(batch)

            # Visualising shapes
            if s.DISPLAY_BATCH_BN:
                visualize(shapes)

            # Step 2: Normalising and remeshing shapes --------------------
            shapes, tot_verts, tot_faces = normalize_data(shapes)

            # Visualising normalised shapes
            if s.DISPLAY_BATCH_AN:
                visualize(shapes)

            # Step 3: Feature extraction ----------------------------------
            shapes, features = calculate_metrics(shapes, False)

            if s.DISPLAY_PROGRESS:
                print("Progress:" + str(int((i+1/len(batches))*100)) + "%")
        

        features = np.load(s.SAVED_DATA + "features.npy", allow_pickle=True)
        
        # Standardize numeric features
        features = standardize(features)

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
        features = np.load(s.SAVED_DATA + "features.npy", allow_pickle=True)
        labels = np.load(s.SAVED_DATA + "labels.npy", allow_pickle=True)
        features = features.item()
        labels = labels.item()
    except:
        pass


    # Step 4: Querying a shape ------------------------------------

    # Retrieving shape
    print("\n----------------------------------------------------")
    print("3D Shapes Search Engine")
    print("----------------------------------------------------")

    while True: # Query loop, see settings.INFINITE_QUERY_LOOP for info

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

        if s.USE_CUSTOM_DISTANCE:
            # Calculate similarities
            print('Calculate similarities using distance function . . .')
            similarities = calc_distance(features, shape_features)

            print("Retrieving and showing similar shapes")
            shapes = load_similar(similarities, shape)

            visualize(shapes)


        # Step 5: Scalable querying -----------------------------------------------

        if s.USE_KNN:
            print("Calculating similar using KNN . . .")
            # Calculate nearest neighbors via ANN and K-Nearest Neighbors
            neighbors = k_neighbors(shape_features, features)
            n_shapes_id, n_distances = neighbors[0][1:], neighbors[1][1:]

            for dist in n_distances:
                print(dist)

            # Retrieving shapes from database
            n_shapes = [shape]
            for shape_id in n_shapes_id:
                filename = s.NORMALIZED_DATA + "n" + str(shape_id) + ".off"
                file = open(filename, 'r')
                verts, faces, n_verts, n_faces = read_off(file)
                mesh = trm.load_mesh(filename)

                shape.set_id(shape_id)

                n_shapes.append(Shape(verts, faces, mesh))

            visualize(n_shapes)

        if s.USE_RNN:
            print("Calculating similar using RNN . . .")
            # Calculate nearest neighbors via ANN and R-Nearest Neighbors
            neighbors = r_neighbors(shape_features, features)
            n_shapes_id, n_distances = neighbors[0][1:], neighbors[1][1:]

            # Retrieving shapes from database
            n_shapes = [shape]
            for shape_id in n_shapes_id:
                filename = s.NORMALIZED_DATA + "n" + str(shape_id) + ".off"
                file = open(filename, 'r')
                verts, faces, n_verts, n_faces = read_off(file)
                mesh = trm.load_mesh(filename)

                n_shapes.append(Shape(verts, faces, mesh))

            visualize(n_shapes)


        if s.INFINITE_QUERY_LOOP:
            continue
        else:
            break

    # Show Dimensionality reduction graphs
    if s.DISPLAY_TSNE:
        print("Dimensionality reduction using TSNE")
        features_arr, labels_arr = convert_dict_to_arr(features, labels)
        tsne_plot(features_arr, labels_arr)
