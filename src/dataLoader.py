import numpy as np
import os
import OpenGL
from collections import OrderedDict


def import_data():
    DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep

    # TODO: to import the entire dataset remove the '0' and the redundant os.sep, REMOVE FOR FINAL PROGRAM
    DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep + '0' + os.sep
    DATA_CLASSIFICATION_PRINCETON = DATA_PATH + 'benchmark' + os.sep + 'classification' + os.sep + 'v1' + os.sep + 'coarse1' + os.sep

    SAVED_DATA = DATA_PATH + 'cache' + os.sep

    if(
       not(
          os.path.isfile(SAVED_DATA + 'shapes.npy') and
          os.path.isfile(SAVED_DATA + 'shape_info.npy')
       )):
        print('Importing shapes and labels . . .')

        shapes = []
        shape_info = {}

        # navigating through the dataset to find .off and .ply files
        for dirName, subdirList, objList in os.walk(DATA_SHAPES_PRICETON):
            for obj in objList:
                if obj.endswith('.off'):
                    file = open(dirName + '\\' + obj, "r")
                    verts, faces, n_verts, n_faces, face_type = read_off(file)
                    shapes.append((verts, faces))
                    shape_info["n_verts"].append(n_verts)
                    shape_info["n_faces"].append(n_faces)
                    shape_info["faces_types"].append(face_type)
                elif (obj.endswith('.ply')):
                    file = open(dirName + '\\' + obj, "r")
                    verts, faces, n_verts, n_faces, face_type = parse_ply(file)
                    shapes.append((verts, faces))
                    shape_info["n_verts"].append(n_verts)
                    shape_info["n_faces"].append(n_faces)
                    shape_info["faces_types"].append(face_type)
                elif (obj.endswith('.txt')):
                    # TODO: implement if meaningful, perhaps for the PSB that has labels
                    continue

        temp1 = {}
        for dirName, subdirList, objList in os.walk(DATA_CLASSIFICATION_PRINCETON):
            for obj in objList:
                if obj.endswith('.cla'):
                    file = open(dirName + '\\' + obj, "r")
                    temp1.update(read_classes(file))

        # Add the dictionary of label, format of 'mesh number': (classname, classnumber)
        shape_info["labels"] = temp1
        np.save(SAVED_DATA + 'shapes.npy', shapes)
        # np.save(SAVED_DATA + 'shape_info.npy', shape_info)
        print('Image train and val sets successfully imported.')

    else:
        print('Loading shapes and labels from cache . . .')

        shapes = np.load(SAVED_DATA + 'shapes.npy', allow_pickle=True)
        shape_info = np.load(SAVED_DATA + 'shape_info.npy', allow_pickle=True)

        print('Existing image train and val sets successfully loaded.')

    return shapes, shape_info


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, other = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    faces_types = list(set([x[0] for x in faces]))[0]
    return verts, faces, n_verts, n_faces, faces_types


def parse_ply(file):
    if 'ply' != file.readline().strip():
        raise ('Not a valid PLY header')
    while True:
        line = str(file.readline().strip())
        if line.startswith('element vertex'):
            n_verts = int(line.split()[-1])  # element vertex 290 --> 290
        if line.startswith('element face'):
            n_faces = int(line.split()[-1])  # element face 190 --> 190
        if line.startswith('property'):
            # performing check for valid XYZ structure
            if (line.split()[-1] == 'x' and
                  str(file.readline().strip()).split()[-1] == 'y' and
                  str(file.readline().strip()).split()[-1] == 'z' and
                  not str(file.readline().strip()).startswith('property')):
                continue
            elif line == 'property list uchar int vertex_indices':
                continue
            else:
                raise ('Not a valid PLY header. Extra properties can not be evaluated.')
        if line == 'end_header':
            break

    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    faces_types = list(set([x[0] for x in faces]))[0]
    return verts, faces,n_verts, n_faces, faces_types


def read_classes(file):
    if 'PSB' not in file.readline().strip():
        raise ('Not a valid PSB classification header')
    num_classes, num_models = file.readline().strip().split()
    modelcount, class_count = 0, 0
    class_dict = {}
    while modelcount < int(num_models):
        line = file.readline().strip().split()
        if len(line) == 0:
            pass  
        elif len(line) > 1:
            class_name = str(line[0])
            class_count += 1
        elif len(line) > 2 and line[1] == '0' and line[2] == '0': # empty class label
            pass
        else: # add the class to the number of the model
            class_dict[line[0]] = (class_name, class_count)
            modelcount += 1
    return class_dict


