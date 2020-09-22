from collections import defaultdict

# Imports from other files,
# from src.shape import Shape
# from src.normalize import normalizeData
# from src.utils import *

from shape import Shape
from normalize import normalizeData
from utils import *


def import_data() -> ([Shape], defaultdict):
    DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep

    # TODO: to import the entire dataset remove the '0' and the redundant os.sep, REMOVE FOR FINAL PROGRAM
    DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep  
    DATA_CLASSIFICATION_PRINCETON = DATA_PATH + 'benchmark' + os.sep + 'classification' + os.sep + 'v1' + os.sep + 'coarse1' + os.sep
    

    SAVED_DATA = DATA_PATH + 'cache' + os.sep
    NORMALIZED_DATA = SAVED_DATA + 'processed_data' + os.sep

    if not os.listdir(NORMALIZED_DATA):
        print("First startup, normalizing the data.")
        normalizeData(DATA_SHAPES_PRICETON, NORMALIZED_DATA)
    
    if(
       not(
          os.path.isfile(SAVED_DATA + 'shapes.NPY') and
          os.path.isfile(SAVED_DATA + 'labels.NPY') and
          os.path.isfile(SAVED_DATA + 'n_verts.NPY') and
          os.path.isfile(SAVED_DATA + 'n_faces.NPY')
       )):
        print('Importing shapes and labels . . .')
                  
        shapes = []
        labels = defaultdict(lambda def_value: None)
        tot_verts = []
        tot_faces = []

        # Importing labels #
        temp1 = {}
        class_list = []
        for dirName, subdirList, objList in os.walk(DATA_CLASSIFICATION_PRINCETON):
            for obj in objList:
                if obj.endswith('.cla'):
                    file = open(dirName + '\\' + obj, "r")
                    class_dict, class_list = read_classes(file, class_list)
                    temp1.update(class_dict)

        # Add the dictionary of label, format of 'mesh number': (classname, classnumber)
        labels = temp1

        # navigating through the dataset to find .off and .ply files
        for dirName, subdirList, objList in os.walk(NORMALIZED_DATA):
            # Importing the shape first
            shape = None
            for obj in objList:
                if obj.endswith('.off'):
                    file = open(dirName + '\\' + obj, "r")
                    verts, faces, n_verts, n_faces = read_off(file)
                    tot_verts.append(n_verts)
                    tot_faces.append(n_faces)

                    shape = Shape(verts, faces)

                elif (obj.endswith('.ply')):
                    file = open(dirName + '\\' + obj, "r")
                    verts, faces, n_verts, n_faces = parse_ply(file)
                    tot_verts.append(n_verts)
                    tot_faces.append(n_faces)

                    shape = Shape(verts, faces)
                
                # Find the text data of the object
                name = obj.split('.')[0]
                path = find(name + '_info.txt', DATA_SHAPES_PRICETON)
                file = open(path, "r")
                shape = read_info(file, shape)

                if shape is not None:
                    # Assigning the class if present
                    shape = calculate_box(shape)
                    shape_id = str(shape.get_id())
                    shape.set_class(labels[shape_id][0], labels[shape_id][1])  # class id, class name
                    # Appending to list
                    shapes.append(shape)

        np.save(SAVED_DATA + 'labels.npy', labels)
        np.save(SAVED_DATA + 'shapes.npy', shapes)
        np.save(SAVED_DATA + 'n_verts.npy', tot_verts)
        np.save(SAVED_DATA + 'n_faces.npy', tot_faces)

        print('Image train and val sets successfully imported.')
    else:
        print('Loading shapes and labels from cache . . .')

        labels = np.load(SAVED_DATA + 'labels.npy', allow_pickle=True)
        shapes = np.load(SAVED_DATA + 'shapes.npy', allow_pickle=True)
        tot_verts = np.load(SAVED_DATA + 'n_verts.npy', allow_pickle=True)
        tot_faces = np.load(SAVED_DATA + 'n_faces.npy', allow_pickle=True)

        print('Existing image train and val sets successfully loaded.')

    # Computing average number of vertices and standard deviation
    avg_verts = np.mean(tot_verts)
    sd_verts = np.std(tot_verts)

    # Showing the normal distribution of vertices on screen
    SHOW_GRAPH = True
    if SHOW_GRAPH:
        show_graph(tot_verts, avg_verts, sd_verts)

    return shapes, labels


