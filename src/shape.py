from src.boundingbox import BoundingBox


class Shape:

    id: int
    vertices: [float]
    faces: [int]
    n_verts: int
    n_faces: int
    faces_types: int
    bbox: BoundingBox
    avg_depth: float
    center: (float, float, float)
    scale: float
    classification: str

    # to be determined if necessary -------------------------
    axis_x: (float, float, float)
    axis_y: (float, float, float)
    axis_z: (float, float, float)
    principle_values: (float, float, float)
    # -------------------------------------------------------

    def __init__(self, vertices, faces):
        self.id = None
        self.vertices = vertices
        self.faces = faces
        self.n_verts = len(vertices)
        self.n_faces = len(faces)
        self.faces_types = list(set([x[0] for x in faces]))[0]

    # ID

    def set_id(self, id):
        self.id = id

    def get_id(self) -> int:
        return self.id

    # Vertices

    def add_vertex(self, vertex):
        self.vertices.append(vertex)
        self.n_verts += 1

    def set_vertices(self, vertices):
        self.vertices = vertices
        self.n_verts = len(vertices)

    def get_vertices(self) -> [float]:
        return self.vertices

    # Faces

    def add_face(self, face):
        self.faces.append(face)
        self.n_faces += 1

    def set_faces(self, faces):
        self.faces = faces
        self.n_faces = len(faces)
        self.faces_types = list(set([x[0] for x in faces]))[0]

    def get_faces(self) -> [int]:
        return self.faces

    # Counts

    def get_n_faces(self) -> int:
        return self.n_faces

    def get_n_verts(self) -> int:
        return self.n_verts

    def get_faces_type(self) -> int:
        return self.faces_types


    # Extra Information

    axis_x: (float, float, float)
    axis_y: (float, float, float)
    axis_z: (float, float, float)
    principle_values: (float, float, float)

    def set_bounding_box(self, xmin, ymin, zmin, xmax, ymax, zmax):
        self.bbox.set_bbox(xmin, ymin, zmin, xmax, ymax, zmax)

    def get_buinding_box(self) -> BoundingBox:
        return self.bbox

    def set_avg_depth(self, avg_depth):
        self.avg_depth = avg_depth

    def get_avg_depth(self) -> float:
        return self.avg_depth

    def set_center(self, center: (float, float, float)):
        self.center = center

    def get_center(self) -> (float, float, float):
        return self.center

    def set_scale(self, scale):
        self.scale = scale

    def get_scale(self) -> float:
        return self.scale

    def set_classification(self, classification):
        self.classification = classification

    def get_classification(self) -> str:
        return self.classification

    # TODO: eventually implement getter and setter for x y z axis

