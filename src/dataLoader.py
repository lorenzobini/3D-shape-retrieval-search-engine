from collections import defaultdict
import open3d as o3d

# Imports from other files,
# from src.shape import Shape
# from src.normalize import normalize_data
# from src.utils import *

from shape import Shape
from normalize import normalize_data
from utils import *


DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep

# TODO: to import the entire dataset remove the '0' and the redundant os.sep, REMOVE FOR FINAL PROGRAM
DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep + '0' + os.sep #+ 'm57' + os.sep
DATA_CLASSIFICATION_PRINCETON = DATA_PATH + 'benchmark' + os.sep + 'classification' + os.sep + 'v1' + os.sep + 'coarse1' + os.sep

SAVED_DATA = DATA_PATH + 'cache' + os.sep
NORMALIZED_DATA = SAVED_DATA + 'processed_data' + os.sep


def import_dataset() -> ([Shape], defaultdict):

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
    for dirName, subdirList, objList in os.walk(DATA_SHAPES_PRICETON):
        # Importing the shape first
        shape = None
        for obj in objList:
            if obj.endswith('.off'):
                file = open(dirName + '\\' + obj, "r")
                verts, faces, n_verts, n_faces = read_off(file)
                mesh = o3d.io.read_triangle_mesh(dirName + '\\' + obj)
                tot_verts.append(n_verts)
                tot_faces.append(n_faces)

                shape = Shape(verts, faces, mesh)

            elif (obj.endswith('.ply')):
                file = open(dirName + '\\' + obj, "r")
                verts, faces, n_verts, n_faces = parse_ply(file)
                mesh = o3d.io.read_triangle_mesh(dirName + '\\' + obj)
                tot_verts.append(n_verts)
                tot_faces.append(n_faces)

                shape = Shape(verts, faces, mesh)

        # Importing extra information
        for obj in objList:
            if obj.endswith('.txt'):
                file = open(dirName + '\\' + obj, "r")
                shape = read_info(file, shape)
        if shape is not None:
            # Assigning the class if present
            shape_class = labels[str(shape.get_id())]
            shape.set_class(shape_class[0], shape_class[1])  # class id, class name
            # Appending to list
            shapes.append(shape)

    np.save(SAVED_DATA + 'shapes.npy', remove_meshes(shapes))
    np.save(SAVED_DATA + 'labels.npy', labels)
    np.save(SAVED_DATA + 'n_verts.npy', tot_verts)
    np.save(SAVED_DATA + 'n_faces.npy', tot_faces)

    print('Image train and val sets successfully imported.')


    # Computing average number of vertices and standard deviation
    avg_verts = np.mean(tot_verts)
    sd_verts = np.std(tot_verts)

    # Showing the normal distribution of vertices on screen
    return shapes, labels


def import_normalised_data():

    print('Loading normalised shapes and labels from cache . . .')

    labels = np.load(SAVED_DATA + 'labels.npy', allow_pickle=True)
    tot_verts = np.load(SAVED_DATA + 'n_verts.npy', allow_pickle=True)
    tot_faces = np.load(SAVED_DATA + 'n_faces.npy', allow_pickle=True)

    shapes = []
    for filename in os.listdir(NORMALIZED_DATA):
        if filename.endswith('.npy'):
            shape = np.load(NORMALIZED_DATA + filename, allow_pickle=True)[0]

            off_file_name = 'n' + str(shape.get_id()) + '.off'
            mesh = o3d.io.read_triangle_mesh(NORMALIZED_DATA + off_file_name)
            shape.set_mesh(mesh)
            shapes.append(shape)

    print("Existing normalised image set successfully loaded.")

    return shapes, labels