import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils import *
import open3d as o3d

# Normalizes the data based on Module 4 of the INFOMR course
def normalizeData(data_path, new_path):
    write = False

    tot_verts = []
    tot_faces = []
    tot_new_verts = []
    tot_new_faces = []

    skip = False
    for dirName, subdirList, objList in os.walk(data_path):
        for obj in objList:
            print(obj)
            if obj.endswith('.off'):
                file = open(dirName + '\\' + obj, "r")
                verts, faces, n_verts, n_faces = read_off(file)
                if n_verts >= 10000:
                    print(obj +" skipped")
                    skip = True
                else:
                    tot_verts.append(n_verts)
                    tot_faces.append(n_faces)
                    name = obj

                # Create Open3D mesh object
                mesh = o3d.io.read_triangle_mesh(dirName + '\\' + obj)
            elif (obj.endswith('.ply')):
                file = open(dirName + '\\' + obj, "r")
                verts, faces, n_verts, n_faces = parse_ply(file)
                if n_verts >= 10000:
                    skip = True
                else:
                    tot_verts.append(n_verts)
                    tot_faces.append(n_faces)

                # Create Open3D mesh object
                mesh = o3d.io.read_triangle_mesh(dirName + '\\' + obj)
            elif obj.endswith('.txt'):
                if skip:
                    write = False
                    skip = False
                else:
                    file = open(dirName + '\\'+ obj, "r")
                    center = read_txt(file)
                    write = True
        
        # Write to new file once all the information is gathered
        if write and not skip:
            
            # TODO change to dynamic values
            avg_verts = 5000
            q1_verts = 2500
            q3_verts = 7500

            # print(f'Before refinement the mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
          
            new_mesh, new_n_verts, new_n_faces = remeshing(mesh, avg_verts, q1_verts, q3_verts)
            center = new_mesh.get_center()
            
            tot_new_verts.append(new_n_verts)
            tot_new_faces.append(new_n_faces)

            new_verts = new_mesh.vertices
            new_faces = new_mesh.triangles


            new_verts = toCenter(new_verts, center)
            # rotate based on PCA
            # new_verts = rotatePCA(new_verts)

            # Scale to bounding box
            # new_verts = scaleToBox(new_verts)

            # Write to a new .OFF file
            f= open(new_path + name,"w+")
            f.write("OFF\n")
            f.write(str(len(new_verts))+" "+ str(len(new_faces)) + " " + "0\n")
            for i in range(0, len(new_verts)):
                vert = [str(x) for x in new_verts[i]]
                vert = ' '.join(vert)
                f.write(vert + '\n')
            for i in range(0, len(new_faces)):
                face = [ str(x) for x in new_faces[i]]
                face = ' '.join(face)
                f.write(face + '\n')
            f.close()
            
            # Reset for next object
            write = False   

    # Computing average number of vertices and standard deviation
    avg_faces = np.mean(tot_faces)
    sd_faces = np.std(tot_faces)
    q1_verts = np.percentile(tot_verts, 25)
    q3_verts = np.percentile(tot_verts, 75)

    # print(f'Before refinement: average number of vertices is: {np.mean(tot_verts)} with sd of: {np.std(tot_verts)} \n and average number of faces is: {np.mean(tot_faces)} with sd of: {np.std(tot_faces)}')
    # print(f'After refinement: average number of vertices is: {np.mean(tot_new_verts)} with sd of: {np.std(tot_new_verts)} \n and average number of faces is: {np.mean(tot_new_faces)} with sd of: {np.std(tot_new_faces)}')
    

# Remeshes shapes that 
def remeshing(mesh, avg_verts, q1_verts, q3_verts):

    while len(mesh.vertices) < q1_verts or len(mesh.vertices) > q3_verts:     
        if len(mesh.vertices) < avg_verts:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        elif len(mesh.vertices) >= avg_verts:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=avg_verts)
            
    # print(f'After simplifying the mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    
    return mesh, n_verts, n_faces


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



    