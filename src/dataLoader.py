import numpy as np
import matplotlib.pyplot as plt
import os
import re
from src.shape import Shape
from collections import defaultdict


def import_data():
    DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep

    # TODO: to import the entire dataset remove the '0' and the redundant os.sep, REMOVE FOR FINAL PROGRAM
    DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep + '0' + os.sep
    DATA_CLASSIFICATION_PRINCETON = DATA_PATH + 'benchmark' + os.sep + 'classification' + os.sep + 'v1' +\
                                    os.sep + 'coarse1' + os.sep

    SAVED_DATA = DATA_PATH + 'cache' + os.sep

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
        for dirName, subdirList, objList in os.walk(DATA_CLASSIFICATION_PRINCETON):
            for obj in objList:
                if obj.endswith('.cla'):
                    file = open(dirName + '\\' + obj, "r")
                    temp1.update(read_classes(file))

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
                    tot_verts.append(n_verts)
                    tot_faces.append(n_faces)

                    shape = Shape(verts, faces)

                elif (obj.endswith('.ply')):
                    file = open(dirName + '\\' + obj, "r")
                    verts, faces, n_verts, n_faces = parse_ply(file)
                    tot_verts.append(n_verts)
                    tot_faces.append(n_faces)

                    shape = Shape(verts, faces)

            # Importing extra information
            for obj in objList:
                if obj.endswith('.txt'):
                    file = open(dirName + '\\' + obj, "r")
                    shape = read_info(file, shape)


            if shape is not None:
                # Assigning the class if present
                shape_id = str(shape.get_id())
                shape.set_class(labels[shape_id][1], labels[shape_id][0])  # class id, class name
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
    avg_faces = np.mean(tot_faces)
    sd_faces = np.std(tot_faces)

    # Showing the normal distribution of vertices on screen

    SHOW_GRAPH = True
    if SHOW_GRAPH:
        show_graph(tot_faces, avg_faces, sd_faces)


    return shapes, labels


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, other = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    return verts, faces, n_verts, n_faces


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
    return verts, faces, n_verts, n_faces


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


def read_info(file, shape):
    for line in file:
        if line.startswith('mid'):
            shape.set_id(int(line.split()[-1]))  # element vertex 290 --> 290
        if line.startswith('bounding_box'):
            pattern = 'bounding_box: xmin = (?P<xmin>.*), ymin = (?P<ymin>.*), zmin = (?P<zmin>.*),' \
                      ' xmax = (?P<xmax>.*), ymax = (?P<ymax>.*), zmax = (?P<zmax>.*)'
            matches = re.match(pattern, line)
            shape.set_bounding_box(float(matches.group('xmin')),
                                   float(matches.group('ymin')),
                                   float(matches.group('zmin')),
                                   float(matches.group('xmax')),
                                   float(matches.group('ymax')),
                                   float(matches.group('zmax')))

        if line.startswith('avg_depth'):
            shape.set_avg_depth(float(line.split()[-1]))
        if line.startswith('center'):
            pattern = 'center: \((?P<x>.*),(?P<y>.*),(?P<z>.*)\)'
            matches = re.match(pattern, line)
            shape.set_center((float(matches.group('x')),
                              float(matches.group('y')),
                              float(matches.group('z'))))
        if line.startswith('scale'):
            shape.set_scale(float(line.split()[-1]))

    return shape


def show_graph(faces, avg, sd):
    hist, bin_edges = np.histogram(faces, bins=range(0, 40000, 5000))

    plt.figure(figsize=[10, 8])

    plt.bar(bin_edges[:-1], hist, width=4000, color='#0504aa', alpha=0.7)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Number of Vertices', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Normal Distribution Histogram', fontsize=15)

    plt.show()
