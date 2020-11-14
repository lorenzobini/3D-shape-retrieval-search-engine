import numpy as np
import copy
import trimesh as trm
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.patheffects as PathEffects
import seaborn as sns

# from src.shape import Shape
# from src.settings import Settings
from shape import Shape
from settings import Settings

s = Settings()

sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})


def read_off(file) -> ([float], [int], int, int):
    """
    It parses an .OFF file.
    ----------------------------
    Args:
        file: The .PLY file

    Returns:
        (obj: 'tuple' of (obj: 'list' of float, obj: 'list' of int, int, int)):
                    The vertices of the shape, the faces of the shape, the number of vertices, the number of faces

    Raises:
        ImportError
    """
    if 'OFF' != file.readline().strip():
        raise('Not a valid OFF header', ImportError)

    n_verts, n_faces, other = tuple([int(s) for s in file.readline().strip().split(' ')])
    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]

    return verts, faces, n_verts, n_faces


def parse_ply(file):
    """
    It parses an .PLY file.
    ----------------------------
    Args:
        file: The .PLY file

    Returns:
        (obj: 'tuple' of (obj: 'list' of float, obj: 'list' of int, int, int)):
                    The vertices of the shape, the faces of the shape, the number of vertices, the number of faces

    Raises:
        ImportError
    """
    if 'ply' != file.readline().strip():
        raise ('Not a valid PLY header', ImportError)

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
                raise ('Not a valid PLY header. Extra properties can not be evaluated.', ImportError)
        if line == 'end_header':
            break

    verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
    faces = [[int(s) for s in file.readline().strip().split(' ')] for i_face in range(n_faces)]

    return verts, faces, n_verts, n_faces


def read_classes(file, class_list):
    """
    It reads the classes from a .CLA file
    ----------------------------
    Args:
        file: The .CLA file
        class_list: The list of classes

    Returns:
        class_dict (obj: 'dict'): The dictionary containing class id and respective class name
        class_list (obj: 'list' of str): The list of all found classes

    Raises:
        ImportError
    """

    if 'PSB' not in file.readline().strip():
        raise ('Not a valid PSB classification header', ImportError)

    _, num_models = file.readline().strip().split()
    modelcount = 0
    class_dict = {}

    while modelcount < int(num_models):
        line = file.readline().strip().split()
        if len(line) == 0:
            pass  
        elif len(line) > 2 and line[2] == '0':  # empty class label
            pass
        elif len(line) > 2:
            class_name = str(line[0])
            # if the class not in the class_list add it
            if class_name not in class_list:
                class_list.append(class_name)
        else: # add the class to the number of the model
            class_id = class_list.index(class_name)  # give class id based on class_list index
            class_dict[line[0]] = (class_id, class_name)
            modelcount += 1

    return class_dict, class_list


def read_info(file, shape):
    """
    It parses extra information from the .TXT file associated with a shape.
    ----------------------------
    Args:
        file: The .TXT file
        shape (obj: 'Shape'): The associated shape

    Returns:
        shape (obj: 'Shape): The updated shape

    Raises:
        ImportError
    """
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


def pick_file():
    """
    It opens a file picker window and lets the user choose a .OFF or .PLY file to import
    ----------------------------
    Returns:
        shape (obj: 'Shape'): The chosen shape

    Raises:
        FileNotFoundError
        FileExistsError
    """
    root = Tk()
    root.filename = askopenfilename(initialdir=s.DATA_SHAPES_PRICETON,
                                    filetypes =(("OFF files", "*.off"), ("PLY files", "*.ply"), ("All Files", "*.*")),
                                    title="Choose a file."
                                   )
    print(root.filename)
    root.destroy()
    try:
        with open(root.filename, 'r') as UseFile:
            UseFile.read()
    except:
        raise FileNotFoundError

    if str(root.filename).endswith(".off"):
        file = open(root.filename, 'r')
        verts, faces, n_verts, n_faces = read_off(file)
        mesh = trm.load_mesh(root.filename)

        shape = Shape(verts, faces, mesh)

    elif str(root.filename).endswith(".ply"):
        file = open(root.filename, "r")
        verts, faces, n_verts, n_faces = parse_ply(file)
        mesh = trm.load_mesh(root.filename)

        shape = Shape(verts, faces, mesh)

    else:
        raise FileExistsError

    return shape


def calculate_box(vertices: [[float]]) -> [float]:
    """
    It calculates the surrounding bounding box.
    ----------------------------
    Args:
        vertices (obj: 'list' of obj: 'list' of float): The list of vertices

    Returns:
        shape (obj: 'list' of floats): The vertices of the bounding box


    """
    x_coords = [x[0] for x in vertices]
    y_coords = [x[1] for x in vertices]
    z_coords = [x[2] for x in vertices]

    return [min(x_coords), min(y_coords), min(z_coords), max(x_coords), max(y_coords), max(z_coords)]


def remove_meshes(shapes: [Shape]) -> [Shape]:
    """
    It removes Trimesh mesh from a given set of Shape object to facilitate saving.
    ----------------------------
    Args:
       shapes (obj: 'list' of obj: 'Shape'): The list of shapes

    Returns:
       new_shapes (obj: 'list' of obj: 'Shape'): The list of shapes with Trimesh meshes removed

    """
    new_shapes = []
    for shape in shapes:
        new_shape = copy.deepcopy(shape)
        new_shape.delete_mesh()
        new_shapes.append(new_shape)

    return new_shapes


def write_off(path: str, shape: Shape):
    """
    It saves a Shape object at the specified path in .OFF format.
    ----------------------------
    Args:
        path (str): The global path
        shape (obj: 'Shape'): The shape to save

    """
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
        f.write( face + '\n')
    f.close()


def calc_eigenvectors(verts: [[float]]):
    """
    It computes eigenvectors from a set of vertices.
    ----------------------------
    Args:
       verts (obj: 'list' of obj: 'list' of float): The list of vertices

    Returns:
        eigenvalues (obj: 'list'): The list of eigenvalues
        eigenvectors (obj: 'list'): The list of eigenvectors

    """
    A = np.zeros((3, len(verts)))

    A[0] = np.array([x[0] for x in verts]) # First row is all the X_coords
    A[1] = np.array([x[1] for x in verts]) # second row is all the Y_coords
    A[2] = np.array([x[2] for x in verts]) # third row is all the z-coords
    
    A_cov = np.cov(A) # This is returns a 3x3
    eigenvalues, eigenvectors = np.linalg.eigh(A_cov)

    return eigenvalues, eigenvectors


# Compute the angle between 3 points
def compute_angle(a: [float], b: [float], c: [float]) -> float:
    """
    It computes the angle between three points.
    ----------------------------
    Args:
       a (float): The first vertex
       b (float): The second vertex
       c (float): The third vertex

    Returns:
       angle (float): The angle enclosed by the three vertices

    """
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.abs(cosine_angle))

    return angle


def normalize_hist(hist) -> [float]:
    """
    It normalizes the values of a histogram.
    ----------------------------
    Args:
       hist (float or int): The histogram to normalize

    Returns:
       newhist (obj: 'list' of float): The normalized histogram

    """
    hsum = np.sum(hist)
    newhist = []
    for hi in hist:
        newhist.append(hi/hsum)

    return newhist


def flatten_features_array(features: {}) -> []:
    """
    It flattens irregular features array of non-iterables and lists.
    ----------------------------
    Args:
       features (obj: 'dict'): The features in the form of a dictionary

    Returns:
       flattened (obj: 'list'): The features in the form of a flattened array

    """
    flattened = []
    flattened.append(features["volume"])
    flattened.append(features["area"])
    flattened.append(features["compactness"])
    flattened.append(features["bbox_volume"])
    flattened.append(features["diameter"])
    flattened.append(features["eccentricity"])
    for i in features["A3"][0]:
        flattened.append(i)
    for i in features["D1"][0]:
        flattened.append(i)
    for i in features["D2"][0]:
        flattened.append(i)
    for i in features["D3"][0]:
        flattened.append(i)
    for i in features["D4"][0]:
        flattened.append(i)

    return flattened


def convert_dict_to_arr(features: {}, labels: {}) -> ([], []):
    """
    It converts features and labels to array and save it to disk.
    ----------------------------
    Args:
       features (obj: 'dict'): The features in the form of a dictionary
       labels (obj: 'dict'): The labels in the form of a dictionary

    Returns:
        (obj: 'tuple' of (obj: 'list', obj: 'list'):
                    The features in the form of a flattened array, the labels in the form of a flattened array

    """

    features_arr = []
    labels_arr = []

    for id, featuresList in features.items():

        labels_arr.append([labels.get(str(id))[0], labels.get(str(id))[1], id])

        # Elementary features
        v = featuresList["volume"]
        a = featuresList["area"]
        c = featuresList["compactness"]
        bb = featuresList["bbox_volume"]
        d = featuresList["diameter"]
        e = featuresList["eccentricity"]
        elem_features = [v, a, c, bb, d, e]
        
        # Global features
        a3, d1, d2, d3, d4 = [], [], [], [], []
        for x in featuresList["A3"][0]:
            a3.append(x)
        for x in featuresList["D1"][0]:
            d1.append(x)
        for x in featuresList["D2"][0]:
            d2.append(x)
        for x in featuresList["D3"][0]:
            d3.append(x)
        for x in featuresList["D4"][0]:
            d4.append(x)
        glob_features = np.concatenate((a3, d1, d2, d3, d4))
        features_arr.append(np.concatenate((elem_features, glob_features)))

    np.savetxt(s.SAVED_DATA + 'features_arr.txt', np.asarray(features_arr), delimiter=',')

    return np.asarray(features_arr), np.asarray(labels_arr)


def tsne_plot(features: {}, labels: {}):
    """
    Computes and plot dimensionality reduction.
    ----------------------------
    Args:
       features (obj: 'dict'): The features
       labels (obj: 'dict'): The labels

    """
    # T-SNE computation
    tsne = TSNE(n_components=2, perplexity=40, n_iter=900, random_state=0)
    tsne_result = tsne.fit_transform(features)

    # Color specification
    n_labels = len(np.unique(labels[:,0]))
    label_set = np.unique(labels[:,1])
    palette = np.array(sns.color_palette('hls', n_labels))

    # Plot the scatter for each label seperatly
    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(n_labels):
        ax.scatter(tsne_result[labels[:,0].astype(np.int) == i,0], tsne_result[labels[:,0].astype(np.int) == i,1],  \
            linewidth=0, c=np.array([palette[i]]), label=label_set[i])
    ax.axis('tight')

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize=12)
    ax.set_title("Dimensionality reduction with T-SNE")
    plt.show()