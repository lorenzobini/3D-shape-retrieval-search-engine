import numpy as np
import matplotlib.pyplot as plt
import os
import re
import copy

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


def calculate_box(vertices):
    
    x_coords = [x[0] for x in vertices]
    y_coords = [x[1] for x in vertices]
    z_coords = [x[2] for x in vertices]

    return [min(x_coords),min(y_coords), min(z_coords), max(x_coords), max(y_coords), max(z_coords)]


# Code to show a histogram
def show_graph(faces, avg, sd):
    hist, bin_edges = np.histogram(faces, bins = np.arange(0, 10000, 250))

    plt.figure(figsize=[10, 8])

    plt.bar(bin_edges[:-1], hist, width=250, color='#0504aa', alpha=0.7)
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


# Function that removes TriangleMesh objects for saving
def remove_meshes(shapes):
    new_shapes = []
    for shape in shapes:
        new_shape = copy.deepcopy(shape)
        new_shape.delete_mesh()
        new_shapes.append(new_shape)

    return new_shapes


# Function to write off file on disk
def write_off(path, shape):
    verts = shape.get_vertices()
    faces = shape.get_faces()

    f = open(path + 'n' + str(shape.get_id()) + ".off", "w+")
    f.write("OFF\n")
    f.write(str(len(verts)) + " " + str(len(faces)) + " " + "0\n")
    for i in range(0, len(verts)):
        vert = [str(x) for x in verts[i]]
        vert = ' '.join(vert)
        f.write(vert + '\n')
    for i in range(0, len(faces)):
        face = [str(x) for x in faces[i]]
        face = ' '.join(face)
        f.write('3 '+ face + '\n')
    f.close()

# Calculates eigenvectors based on the vertices
def calc_eigenvectors(verts):
    A = np.zeros((3, len(verts)))

    A[0] = np.array([x[0] for x in verts]) # First row is all the X_coords
    A[1] = np.array([x[1] for x in verts]) # second row is all the Y_coords
    A[2] = np.array([x[2] for x in verts]) # third row is all the z-coords
    
    A_cov = np.cov(A) # This is returns a 3x3
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    return eigenvalues, eigenvectors