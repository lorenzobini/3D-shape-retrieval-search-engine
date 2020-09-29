import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils import *
import open3d as o3d

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
DATA_SHAPES_PRICETON = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep + '0' + os.sep
DATA_CLASSIFICATION_PRINCETON = DATA_PATH + 'benchmark' + os.sep + 'classification' + os.sep + 'v1' + os.sep + 'coarse1' + os.sep

SAVED_DATA = DATA_PATH + 'cache' + os.sep
NORMALIZED_DATA = SAVED_DATA + 'processed_data' + os.sep


# Normalizes the data based on Module 4 of the INFOMR course
def normalize_data(shapes):

    write = False

    tot_verts = []
    tot_faces = []
    tot_new_verts = []
    tot_new_faces = []

    print('Normalising shapes . . .')
    for shape in shapes:

        # TODO: determine suitable parameters
        # Recalling from the lecture: Alex said that between 1000 and 3000 is a good value
        avg_verts = 2000
        q1_verts = 1000
        q3_verts = 3000

        # print(f'Before refinement the mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')

        new_mesh, new_n_verts, new_n_faces = remeshing(shape.get_mesh(), avg_verts, q1_verts, q3_verts)
                
        tot_new_verts.append(new_n_verts)
        tot_new_faces.append(new_n_faces)


        new_verts = new_mesh.vertices
        
        # Translate to center
        center = new_mesh.get_center()
        new_mesh = new_mesh.translate(-center)

        # Scale to bounding box
        bbox = [abs(x) for x in calculate_box(new_verts)]
        ratio = (1/max(bbox))
        new_mesh.scale(ratio, center=(0,0,0))
       
        # rotate based on PCA
        new_mesh = rotate_PCA(new_mesh)


        # Updating shape
        shape.set_vertices(np.asarray(new_mesh.vertices))
        shape.set_faces(np.asarray(new_mesh.triangles).tolist())
        shape.set_center(tuple(new_mesh.get_center()))
        # TODO: update avg_depth? bounding_box? scale?

    print("Shapes normalised succesfully.")
    print("Saving normalised shapes in cache.")

    # Saving normalised shapes and respective off files to disk
    for shape in remove_meshes(shapes):
        write_off(NORMALIZED_DATA, shape)
        np.save(NORMALIZED_DATA + 'n' + str(shape.get_id()) + '.npy', [shape])
    # TODO: override tot_verts and tot_faces in file?

    print("Normalised shapes saved.")

    return shapes, tot_new_verts, tot_new_faces


    # Computing average number of vertices and standard deviation
    avg_faces = np.mean(tot_faces)
    sd_faces = np.std(tot_faces)
    #q1_verts = np.percentile(tot_verts, 25)
    #q3_verts = np.percentile(tot_verts, 75)

    # print(f'Before refinement: average number of vertices is: {np.mean(tot_verts)} with sd of: {np.std(tot_verts)} \n and average number of faces is: {np.mean(tot_faces)} with sd of: {np.std(tot_faces)}')
    # print(f'After refinement: average number of vertices is: {np.mean(tot_new_verts)} with sd of: {np.std(tot_new_verts)} \n and average number of faces is: {np.mean(tot_new_faces)} with sd of: {np.std(tot_new_faces)}')
    

# Remeshes shapes that 
def remeshing(mesh, avg_verts, q1_verts, q3_verts):

    voxel_denominator = 32
    
    while len(mesh.vertices) < q1_verts or len(mesh.vertices) > q3_verts:     
        if len(mesh.vertices) < avg_verts:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)

        elif len(mesh.vertices) >= avg_verts:

            # the bigger the voxel size, the more vertices are clustered and thus the more the mesh is simplified
            voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / voxel_denominator
            mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)
            voxel_denominator = voxel_denominator - 2
            
    #print(f'After simplifying the mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')
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


def calc_eigenvectors(verts):
    A = np.zeros((3, len(verts)))
    A[0] = [x[0] for x in verts]
    A[1] = [x[1] for x in verts]
    A[2] = [x[2] for x in verts]

    A_cov = np.cov(A)
    
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    return eigenvalues, eigenvectors


# Tryout PCA function
def rotate_PCA(mesh):
    
    eigenvalues, eigenvectors = calc_eigenvectors(mesh.vertices)
    
    min_eigen = np.argmin(eigenvalues)
    max_eigen = np.argmax(eigenvalues)
    mid_eigen = (set([0,1,2]) - set([max_eigen, min_eigen])).pop()


    verts = mesh.vertices
    new_verts = verts
    for i in range(0, len(verts)):
        v = verts[i]
        p1 = np.dot(v, eigenvectors[max_eigen])
        p2 = np.dot(v, eigenvectors[mid_eigen])
        p3 = np.dot(v, eigenvectors[min_eigen])
        new_verts[i] = [p1, p2, p3]
    mesh.vertices = new_verts
    return mesh
    
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


