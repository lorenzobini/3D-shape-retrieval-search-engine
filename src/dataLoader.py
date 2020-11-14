from collections import defaultdict
import os
from utils import *
from settings import Settings

# from src.utils import *
# from src.settings import Settings


s = Settings()


def import_dataset(path) -> ([Shape], defaultdict):
    """
    It import the shapes stored within the path specified.
    ----------------------------
    Args:
        path (str): The global path of the shape batch

    Returns:
        (obj: 'tuple' of (obj: 'list' of obj: 'Shape', obj: 'defaultdict')):
                    The loaded shapes, the dictionary containing the labels of all shapes loaded thus far
    """

    print('Importing shapes and labels . . .')

    shapes = []
    labels = defaultdict(lambda def_value: None)
    tot_verts = []
    tot_faces = []

    if os.path.isfile(s.SAVED_DATA + "labels.npy"):
        # Retrieving labels
        labels = np.load(s.SAVED_DATA + "labels.npy", allow_pickle=True)
        # When saved numpy.load returns a numpy.ndarry so this handles that
        try:
            labels = labels.item()
        except:
            pass
    else:
        # Importing labels
        temp1 = {}
        class_list = []
        for dirName, subdirList, objList in os.walk(s.DATA_CLASSIFICATION_PRINCETON):
            for obj in objList:
                if obj.endswith('.cla'):
                    file = open(dirName + '\\' + obj, "r")
                    class_dict, class_list = read_classes(file, class_list)
                    temp1.update(class_dict)

        # Add the dictionary of label, format of 'mesh number': (classname, classnumber)
        labels = temp1
    # Opens the file with the excluded models
    file = open(s.SAVED_DATA + "exclude.txt")
    exclude = list(file.readline().split(','))
    exclude = [x.strip() for x in exclude]

    # navigating through the dataset to find .off and .ply files
    for dirName, subdirList, objList in os.walk(path):
        # Importing the shape first
        shape = None
        for obj in objList:
            if obj.split(".")[0].split("_")[0].split("m")[1] in exclude: # Exclude if the shape is in the exlude list
                next
            elif obj.endswith('.off'):
                file = open(dirName + '\\' + obj, "r")
                verts, faces, n_verts, n_faces = read_off(file)
                mesh = trm.load_mesh(dirName + '\\' + obj)
                tot_verts.append(n_verts)
                tot_faces.append(n_faces)

                shape = Shape(verts, faces, mesh)

            elif (obj.endswith('.ply')):
                file = open(dirName + '\\' + obj, "r")
                verts, faces, n_verts, n_faces = parse_ply(file)
                mesh = trm.load_mesh(dirName + '\\' + obj)
                tot_verts.append(n_verts)
                tot_faces.append(n_faces)

                shape = Shape(verts, faces, mesh)

        # Importing extra information
        for obj in objList:
            if obj.split(".")[0].split("_")[0].split("m")[1] in exclude: 
                next
            elif obj.endswith('.txt'):
                file = open(dirName + '\\' + obj, "r")
                shape = read_info(file, shape)
        if shape is not None:
            # Assigning the class if present
            shape_class = labels[str(shape.get_id())]
            shape.set_class(shape_class[0], shape_class[1])  # class id, class name
            # Appending to list
            shapes.append(shape)

    np.save(s.SAVED_DATA + 'labels.npy', labels)

    return shapes, labels


def import_normalised_data():
    """
    It import the normalised shapes stored in cache.
    ----------------------------
    Returns:
        (obj: 'tuple' of (obj: 'list' of obj: 'Shape', obj: 'dict', obj: 'dict')):
                    The loaded shapes, the dictionary of labels, the dictionary of features
    """

    print('Loading normalised shapes and labels from cache . . .')

    labels = np.load(s.SAVED_DATA + 'labels.npy', allow_pickle=True)
    features = np.load(s.SAVED_DATA + 'features.npy', allow_pickle=True)

    shapes = []
    for filename in os.listdir(s.NORMALIZED_DATA):
        if filename.endswith('.npy'):
            off_file_name = 'n' + str(shape.get_id()) + '.off'

            shape = np.load(s.NORMALIZED_DATA + filename, allow_pickle=True)[0]
            mesh = trm.load_mesh(s.NORMALIZED_DATA + off_file_name)
            shape.set_mesh(mesh)

            shapes.append(shape)

    print("Existing normalised image set successfully loaded.")

    return shapes, labels, features
