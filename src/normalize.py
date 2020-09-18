import numpy as np
import matplotlib.pyplot as plt
import os
from utils import * 

# Normalizes the data based on Module 4 of the INFOMR course
def normalizeData(data_path, new_path):
    write = False
    for dirName, subdirList, objList in os.walk(data_path):
        for obj in objList:
            if obj.endswith('.off'):
                file = open(dirName + '\\' + obj, "r")
                verts, faces, n_verts, n_faces = read_off(file)
                name = obj
            elif (obj.endswith('.ply')):
                file = open(dirName + '\\' + obj, "r")
                verts, faces, n_verts, n_faces = parse_ply(file)
            elif obj.endswith('.txt'):
                file = open(dirName + '\\'+ obj, "r")
                center = read_txt(file)
                write = True
        
        # Write to new file once all the information is gathered
        if write:
            # Move the object to the center
            new_verts = toCenter(verts, center)

            # rotate based on PCA
            # new_verts = rotatePCA(new_verts)

            # Scale to bounding box
            # new_verts = scaleToBox(new_verts)

            # Write to a new .OFF file
            f= open(new_path + name,"w+")
            f.write("OFF\n")
            f.write(str(n_verts)+" "+ str(n_faces) + " " + "0\n")
            for i in range(0, len(new_verts)):
                vert = [str(x) for x in new_verts[i]]
                vert = ' '.join(vert)
                f.write(vert + '\n')
            for i in range(0, len(faces)):
                face = [str(x) for x in faces[i]]
                face = ' '.join(face)
                f.write(face + '\n')
            
            # Reset for next object
            write = False   


# Read the center of mass from the .txt file
def read_txt(file):
    while True: 
        line = file.readline().strip().split()
        if line[0] == 'center:':
            coordinates = line[1].strip('()').split(',')
            center = [float(x) for x in coordinates]
            break
    return center

# Translate each vertice based on the center to the origin. 
def toCenter(verts, center):
     # Normalize to center. 
    new_verts = verts
    for i in range(0, len(verts)):
        new_verts[i] = [vertice - coord for vertice, coord in zip(verts[i], center)]   
    return new_verts

# Tryout PCA function
def rotatePCA(verts):
    A = np.zeros((3, len(verts)))
    A[0] = [x[0] for x in verts]
    A[1] = [x[1] for x in verts]
    A[2] = [x[2] for x in verts]

    A_cov = np.cov(A)
    
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    min_eigen = np.argmin(eigenvalues)
    max_eigen = np.argmax(eigenvalues)
    
    
    
    # new_verts = verts
    # for i in range(0, len(verts)):
    #     print(new_verts[i])
    #     new_verts[i] = verts[i]
    #     print(new_verts[i])
    
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix    



    