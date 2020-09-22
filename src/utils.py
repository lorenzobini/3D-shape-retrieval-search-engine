import numpy as np
import matplotlib.pyplot as plt
import os
import re


# Parse a .off file
def read_off(file):
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header')
    n_verts, n_faces, other = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]
    return verts, faces, n_verts, n_faces


# Parse a .ply file
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


# Read the classes from the .cla file
def read_classes(file, class_list):
    if 'PSB' not in file.readline().strip():
        raise ('Not a valid PSB classification header')
    _, num_models = file.readline().strip().split()
    modelcount = 0
    class_dict = {}
    while modelcount < int(num_models):
        line = file.readline().strip().split()
        if len(line) == 0:
            pass  
        elif len(line) > 2  and line[2] == '0': # empty class label
            pass
        elif len(line) > 2:
            class_name = str(line[0])
            # if the class not in the class_list add it
            if class_name  not in class_list:
                class_list.append(class_name)
        else: # add the class to the number of the model
            class_id = class_list.index(class_name) # give class id based on class_list index
            class_dict[line[0]] = (class_id, class_name)
            modelcount += 1

    return class_dict, class_list


# Read the .txt field and retrieve information
def read_info(file, shape):
    for line in file:
        if line.startswith('mid'):
            shape.set_id(int(line.split()[-1])) 

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


def calculate_box(shape):
    # Setting the correct id for the shape so the 
    vertices = shape.get_vertices()
    x_coords = [x[0] for x in vertices]
    y_coords = [x[1] for x in vertices]
    z_coords = [x[2] for x in vertices]

    shape.set_bounding_box(min(x_coords),min(y_coords), min(z_coords), max(x_coords), max(y_coords), max(z_coords))
    return shape


# Code to show a histogram
def show_graph(faces, avg, sd):
    hist, bin_edges = np.histogram(faces, bins=range(0, 20000, 500))

    plt.figure(figsize=[10, 8])

    plt.bar(bin_edges[:-1], hist, width=500, color='#0504aa', alpha=0.7)
    plt.xlim(min(bin_edges), max(bin_edges))
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Number of Vertices', fontsize=15)
    plt.ylabel('Frequency', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Vertices Distribution Histogram', fontsize=15)

    plt.show()


# Function to find a file given a certain path
def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)