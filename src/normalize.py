import open3d as o3d
import trimesh

# from src.utils import *
# from src.shape import Shape
# from src.settings import Settings
from shape import Shape
from utils import *
from settings import Settings

s = Settings()


def normalize_data(shapes: [Shape]) -> ():
    """
    It normalizes the shapes.
    ----------------------------
    Args:
        shapes (obj: 'list' of obj: 'Shape'): The list of shapes

    Returns:
        (obj: 'tuple' of (obj: 'list' of obj: 'Shape', obj: 'list' of int, obj: 'list' of int)):
                    The normalized shapes, the total new vertices, the total new faces
    """

    normalized_shapes = []
    tot_new_verts = []
    tot_new_faces = [] 

    print('Normalising shapes . . .')
    for shape in shapes:
        normalized_shape, new_n_verts, new_n_faces = normalize_shape(shape)
        normalized_shapes.append(normalized_shape)
        tot_new_verts.append(new_n_verts)
        tot_new_faces.append(new_n_faces)

    print("Shapes normalised succesfully.")
    print("Saving normalised shapes in cache.")

    # Saving normalised shapes and respective off files to disk
    for shape in remove_meshes(normalized_shapes):
        write_off(s.NORMALIZED_DATA, shape)
        np.save(s.NORMALIZED_DATA + 'n' + str(shape.get_id()) + '.npy', [shape])
   
    print("Normalised shapes saved.")

    return shapes, tot_new_verts, tot_new_faces


def normalize_shape(shape: Shape):
    """
    It normalizes a single shape by applying remeshing, translation to center, PCA rotation, flipping
    and scaling to bounding box.
    ----------------------------
    Args:
        shape (obj: Shape): The shape to normalize

    Returns:
        (obj: 'tuple' of (obj: 'Shape', int, int)):
                    The normalized shape, the number of new vertices, the number of new faces
    """

    new_mesh, new_n_verts, new_n_faces = remeshing(shape.get_mesh())
                
    # Translate to center
    new_mesh.vertices -= new_mesh.center_mass
    
    # rotate based on PCA
    new_mesh = rotate_PCA(new_mesh)

    # flipping
    new_mesh = flip_mesh(new_mesh)

    # Scale to bounding box
    x_min, y_min, z_min, x_max, y_max, z_max = calculate_box(new_mesh.vertices)
    scale = max([x_max-x_min, y_max-y_min, z_max-z_min])
    ratio = (1/scale)
    new_mesh.vertices *= ratio
    x_min, y_min, z_min, x_max, y_max, z_max = calculate_box(new_mesh.vertices)

    # Updating shape
    shape.set_vertices(np.asarray(new_mesh.vertices))
    faces = np.insert(np.array(new_mesh.faces), 0, np.full(len(new_mesh.faces), 3), axis=1) # makes sure that the right format is handled
    shape.set_faces(faces.tolist())
    shape.set_center(tuple(new_mesh.center_mass))
    shape.set_bounding_box(x_min, y_min, z_min, x_max, y_max, z_max)
    shape.set_scale(scale)

    return shape, new_n_verts, new_n_faces


def remeshing(mesh, avg_verts=s.AVG_VERTS, q1_verts=s.Q1_VERTS, q3_verts=s.Q3_VERTS):
    """
    It applies remeshing algorithms on a shape mesh.
    ----------------------------
    Args:
        mesh (obj: 'base.Trimesh'): The mesh of the shape in Trimesh format
        avg_verts (int): Average vertices for remeshing, default stored in Settings
        q1_verts (int): Q1 vertices, default stored in Settings
        q3_verts (int): Q3 vertices, default stored in Settings

    Returns:
        (obj: 'tuple' of (obj: 'list' of obj: 'base.Trimesh', int, int)):
                    The remeshed mesh, the number of new vertices, the number of new faces
    """

    v = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    f = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    mesh = o3d.geometry.TriangleMesh(vertices = v, triangles = f)
    voxel_denominator = 32

    while len(mesh.vertices) < q1_verts or len(mesh.vertices) > q3_verts:     

        if len(mesh.vertices) < avg_verts:
            mesh = mesh.subdivide_midpoint(number_of_iterations=1)
        elif len(mesh.vertices) >= avg_verts:
            # the bigger the voxel size, the more vertices are clustered and thus the more the mesh is simplified
            voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / voxel_denominator
            mesh = mesh.simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)
            voxel_denominator = voxel_denominator - 2

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.triangles)
    new_mesh = trimesh.Trimesh(mesh.vertices, mesh.triangles)

    return new_mesh, n_verts, n_faces


def rotate_PCA(mesh):
    """
    It applies PCA rotation on a shape mesh.
    ----------------------------
    Args:
        mesh (obj: 'base.Trimesh'): The mesh of a shape

    Returns:
        mesh (obj: 'base.Trimesh'): The rotated mesh of the shape
    """
    
    eigenvalues, eigenvectors = calc_eigenvectors(mesh.vertices)
    
    min_eigen = np.argmin(eigenvalues)
    max_eigen = np.argmax(eigenvalues)
    mid_eigen = ({0, 1, 2} - {max_eigen, min_eigen}).pop()

    verts = mesh.vertices
    c = mesh.center_mass

    new_verts = []

    for i in range(0, len(verts)):
        v = verts[i]
        p1 = np.dot(v-c, eigenvectors[:, max_eigen])
        p2 = np.dot(v-c, eigenvectors[:, mid_eigen])
        p3 = np.dot(v-c, eigenvectors[:, min_eigen])
        new_verts.append([p1, p2, p3])
    mesh.vertices = new_verts

    return mesh


def calculate_f(triangle_coords: [float]) -> float:
    """
    It computes the F value given the coordinates of a triangle
    ----------------------------
    Args:
        triangle_coords (obj: 'list' of float): The coordinates of a triangle

    Returns:
        f_i (float): The F value given the triangle coordinates
    """
    f_i = 0
    for x in triangle_coords:
        f_i += np.sign(x)*(x**2)
    return f_i


def flip_mesh(mesh):
    """
    It flips the mesh of a shape.
    ----------------------------
    Args:
        mesh (obj: 'base.Trimesh'): The mesh of a shape

    Returns:
        mesh (obj: 'base.Trimesh'): The flipped mesh of the shape
    """

    triangles = np.zeros((3, len(mesh.faces)))
    for i, index in enumerate(mesh.faces[1:]):
        x, y, z = [], [], []
        for num in index:
            vertices = mesh.vertices[num]
            x.append(vertices[0])
            y.append(vertices[1])
            z.append(vertices[2])
        triangles[0][i] = np.sum(x)/3 
        triangles[1][i] = np.sum(y)/3
        triangles[2][i] = np.sum(z)/3

    f_x = calculate_f(triangles[0])
    f_y = calculate_f(triangles[1])
    f_z = calculate_f(triangles[2])

    R = np.array([[np.sign(f_x), 0, 0], [0, np.sign(f_y), 0], [0, 0, np.sign(f_z)]])
    mesh.vertices = np.matmul(mesh.vertices, R)

    return mesh




    


