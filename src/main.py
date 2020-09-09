import numpy as np
import os
import OpenGL


def main():
    shapes, shape_labels = import_data()
    None

def import_data():
    DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep

    # TODO: to import the entire dataset remove the '0' and the redundant os.sep
    DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep + '0' + os.sep
    DATA_SHAPES_PSB = DATA_PATH + 'LabeledDB_new' + os.sep

    SAVED_DATA = DATA_PATH + 'cache' + os.sep

    if(
       not(
          os.path.isfile(SAVED_DATA + 'shapes.npy') and
          os.path.isfile(SAVED_DATA + 'shape_labels.npy')
       )):
        print('Importing shapes and labels . . .')

        shapes = []
        shape_labels = []

        # navigating through the dataset to find .off and .ply files
        for dirName, subdirList, objList in os.walk(DATA_SHAPES_PRICETON):
            for obj in objList:
                if obj.endswith('.off'):
                    file = open(dirName + '\\' + obj, "r")
                    verts, faces = read_off(file)
                    shapes.append((verts, faces))
                elif (obj.endswith('.ply')):
                    file = open(dirName + '\\' + obj, "r")
                    shape = parse_ply(file)
                    shapes.append(shape)
                elif (obj.endswith('.txt')):
                    # TODO: implement if meaningful, perhaps for the PSB that has labels
                    continue

        np.save(SAVED_DATA + 'shapes.npy', shapes)
        np.save(SAVED_DATA + 'shape_labels.npy', shape_labels)

        print('Image train and val sets successfully imported.')

    else:
        print('Loading shapes and labels from cache . . .')

        shapes = np.load(SAVED_DATA + 'shapes.npy')
        shape_labels = np.load(SAVED_DATA + 'shape_labels.npy')

        print('Existing image train and val sets successfully loaded.')

    return shapes, shape_labels


def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, other = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


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
            elif line == 'property list uchar int vertex_indeces':
                continue
            else:
                raise ('Not a valid PLY header. Extra properties can not be evaluated.')
        if line == 'end_header':
            break

    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
    return verts, faces


main()