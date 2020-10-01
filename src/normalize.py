from src.utils import *
# from utils import *
import open3d as o3d
import os

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

        avg_verts = 2000
        q1_verts = 1000
        q3_verts = 3000

        # print(f'Before refinement the mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')

        new_mesh, new_n_verts, new_n_faces = remeshing(shape.get_mesh(), avg_verts, q1_verts, q3_verts)
                
        tot_new_verts.append(new_n_verts)
        tot_new_faces.append(new_n_faces)
     
        # Translate to center
        center = new_mesh.get_center()
        verts = new_mesh.vertices
        for i,vertix in enumerate(verts):
            verts[i] = vertix-center
        new_mesh.vertices = verts

        # orientates the normals in the same direction
        new_mesh.orient_triangles()
        
        # rotate based on PCA
        new_mesh = rotate_PCA(new_mesh)

        # flipping
        new_mesh = flip_mesh(new_mesh)

        # Scale to bounding box
        x_min, y_min, z_min, x_max, y_max, z_max = calculate_box(new_mesh.vertices)
        scale = max([x_max-x_min, y_max-y_min, z_max-z_min])
        ratio = (1/scale)
        new_mesh.scale(ratio, center=(0,0,0))

        # Updating shape
        shape.set_vertices(np.asarray(new_mesh.vertices))
        shape.set_faces(np.asarray(new_mesh.triangles).tolist())
        shape.set_center(tuple(new_mesh.get_center()))
        shape.set_bounding_box(x_min, y_min, z_min, x_max, y_max, z_max)
        shape.set_scale(scale)
        # TODO: update avg_depth? scale?
    

    print("Shapes normalised succesfully.")
    print("Saving normalised shapes in cache.")

    # Saving normalised shapes and respective off files to disk
    for shape in remove_meshes(shapes):
        write_off(NORMALIZED_DATA, shape)
        np.save(NORMALIZED_DATA + 'n' + str(shape.get_id()) + '.npy', [shape])
    # TODO: override tot_verts and tot_faces in file?

    print("Normalised shapes saved.")

    return shapes, tot_new_verts, tot_new_faces


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

# Applies the PCA to the vertices of the mesh
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
        p3 = np.dot(v, eigenvectors[max_eigen]*eigenvectors[mid_eigen])
        new_verts[i] = [p1, p2, p3]
    mesh.vertices = new_verts
    return mesh

def calculate_f(triangle_coords):
    f_i = 0
    for x in triangle_coords:
        f_i += np.sign(x)*(x**2)
    return f_i

def flip_mesh(mesh):
    triangles = np.zeros((3, len(mesh.triangles)))
    for i, index in enumerate(mesh.triangles[1:]):
        x, y, z = [],[],[]
        for num in index:
            vertice = mesh.vertices[num]
            x.append(vertice[0])
            y.append(vertice[1])
            z.append(vertice[2])
        triangles[0][i] = np.sum(x)/3 
        triangles[1][i] = np.sum(y)/3
        triangles[2][i] = np.sum(z)/3

 
    f_x = calculate_f(triangles[0])
    f_y = calculate_f(triangles[1])
    f_z = calculate_f(triangles[2])

    R = np.array([[np.sign(f_x), 0,0], [0, np.sign(f_y), 0], [0,0, np.sign(f_z)]])
    mesh.rotate(R, center=(0,0,0))
    return mesh



    


