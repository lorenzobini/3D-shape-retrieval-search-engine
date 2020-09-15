
class Shape:

    vertices: [float]
    faces: [int]
    n_verts: int
    n_faces: int
    faces_types: int

    def __init__(self, vertices, faces):
        self.vertices = vertices
        self.faces = faces
        self.n_verts = len(vertices)
        self.n_faces = len(faces)
        self.faces_types = list(set([x[0] for x in faces]))[0]

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