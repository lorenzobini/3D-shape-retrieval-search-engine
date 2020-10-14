from collections import defaultdict
import os
import numpy as np
import matplotlib.pyplot as plt

# from shape import Shape
# from visualize import visualize
# from normalize import normalize_data, normalize_shape
# from dataLoader import import_dataset, import_normalised_data
# from features import *


from src.shape import Shape
from src.visualize import visualize
from src.normalize import normalize_data, normalize_shape
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
    FORCE_IMPORT = False
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


    # ---------------------------------------------------------
    # ONLINE WORKFLOW - QUERYING A SHAPE AND DISPLAYING RESULTS
    # ---------------------------------------------------------

    # Uploading the previously computed features
    '''
    try:
        features = np.load(SAVED_DATA + "features.npy")
        features = features.item()
    except:
        pass
    '''

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
    shape, new_n_verts, new_n_faces = normalize_shape(shape)

    # Calculating features for the shape
    shape_features = calculate_single_shape_metrics(shape)
    

    # v_h, a_h, d_h, c_h, bb_h, e_h = [],[],[],[],[],[]
    # v_w, a_w, d_w, c_w, bb_w, e_w = [],[],[],[],[],[]
    # v_f, a_f, d_f, c_f, bb_f, e_f = [],[],[],[],[],[]
    # for id, featurelist in features.items():
    #     label = labels[str(id)][1]
    #     if label == 'human':
    #         v_h.append(featurelist['volume'])
    #         a_h.append(featurelist["area"])
    #         d_h.append(featurelist['diameter'])
    #         c_h.append(featurelist['compactness'])
    #         bb_h.append(featurelist['bbox_volume'])
    #         e_h.append(featurelist['eccentricity'])
    #     if label == 'wheel':
    #         v_w.append(featurelist['volume'])
    #         a_w.append(featurelist["area"])
    #         d_w.append(featurelist['diameter'])
    #         c_w.append(featurelist['compactness'])
    #         bb_w.append(featurelist['bbox_volume'])
    #         e_w.append(featurelist['eccentricity'])
    #     if label == 'underwater_creature':
    #         v_f.append(featurelist['volume'])
    #         a_f.append(featurelist["area"])
    #         d_f.append(featurelist['diameter'])
    #         c_f.append(featurelist['compactness'])
    #         bb_f.append(featurelist['bbox_volume'])
    #         e_f.append(featurelist['eccentricity'])

    # print("Values for the human models")
    # print("Area mean:", np.mean(a_h), " std: ", np.std(a_h))
    # print("Volume mean:", np.mean(v_h), " std: ", np.std(v_h))
    # print("Compactness mean:", np.mean(c_h), " std: ", np.std(c_h))
    # print("bbox_volume mean:", np.mean(bb_h), " std: ", np.std(bb_h))
    # print("Diameter mean:", np.mean(d_h), " std: ", np.std(d_h))
    # print("Eccentricity mean:", np.mean(e_h), " std: ", np.std(e_h))

    # print("\n")
    # print("Values for the wheel models")
    # print("Area mean:", np.mean(a_w), " std: ", np.std(a_w))
    # print("Volume mean:", np.mean(v_w), " std: ", np.std(v_w))
    # print("Compactness mean:", np.mean(c_w), " std: ", np.std(c_w))
    # print("bbox_volume mean:", np.mean(bb_w), " std: ", np.std(bb_w))
    # print("Diameter mean:", np.mean(d_w), " std: ", np.std(d_w))
    # print("Eccentricity mean:", np.mean(e_w), " std: ", np.std(e_w))

    # print("\n")
    # print("Values for the wheel models")
    # print("Area mean:", np.mean(a_f), " std: ", np.std(a_f))
    # print("Volume mean:", np.mean(v_f), " std: ", np.std(v_f))
    # print("Compactness mean:", np.mean(c_f), " std: ", np.std(c_f))
    # print("bbox_volume mean:", np.mean(bb_f), " std: ", np.std(bb_f))
    # print("Diameter mean:", np.mean(d_f), " std: ", np.std(d_f))
    # print("Eccentricity mean:", np.mean(e_f), " std: ", np.std(e_f))
    # 
    hists, hists2, hists3, hists4, hists5, hists6 = [], [], [], [], [], []
    for id, featurelist in features.items():
        label = labels[str(id)][1]
        if label == 'human':
            hists.append(featurelist['D3'][0])
            bin_edges1 = featurelist['D3'][1]
            hists4.append(featurelist["D1"][0])
            bin_edges4 = featurelist['D1'][1]
        if label == 'wheel':
            hists2.append(featurelist['D3'][0])
            bin_edges2 = featurelist['D3'][1]
            hists5.append(featurelist['D1'][0])
            bin_edges5 = featurelist['D1'][1]
        if label == 'underwater_creature':
            hists3.append(featurelist['D3'][0])
            bin_edges3 = featurelist['D3'][1]
            hists6.append(featurelist['D1'][0])
            bin_edges6 = featurelist['D1'][1]

    plt.figure(figsize=[5,3])
    for i, hist in enumerate(hists):
        plt.plot(bin_edges1[:-1], hist, color = 'green')
        plt.plot(bin_edges2[:-1], hists2[i], color = 'red')
        plt.plot(bin_edges3[:-1], hists3[i], color = 'blue')
        plt.ylim(ymax = 1.0, ymin=0.0)
    plt.title('D3 distribution', fontsize=15)
    plt.legend(["Human", "Wheel", "Fish"], loc='upper left')
    plt.show()

    plt.figure(figsize=[5,3])
    for i, hist in enumerate(hists4):
        plt.plot(bin_edges4[:-1], hist, color = 'green')
        plt.plot(bin_edges5[:-1], hists5[i], color = 'red')
        plt.plot(bin_edges6[:-1], hists6[i], color = 'blue')
        plt.ylim(ymax = 1.0, ymin=0.0)
    
    plt.title('D1 distribution', fontsize=15)
    plt.legend(["Human", "Wheel", "Fish"], loc='upper left')
    plt.show()
        
        




main()
