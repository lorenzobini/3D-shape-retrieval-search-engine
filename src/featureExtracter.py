import numpy as np
import os
from shape import Shape

DATA_PATH = os.path.join(os.getcwd(), 'data') + os.sep
SAVED_DATA = DATA_PATH + 'cache' + os.sep

def surface_area(shapes):

    print("Calculating the surface . . .")

    volume_arr = [["id", "volume"]]

    for shape in shapes:
        
        # The mesh must be closed (watertight) to compute the volume
        if shape.is_watertight():

            #center = shape.get_center()
            verts = shape.get_vertices()
            faces = shape.get_faces()

            volume = float(0)

            # Loop through all the triangles in the shape
            for face in faces:
                # The three vertex coordinates of the triangle
                p1 = verts[face[0]]
                p2 = verts[face[1]]
                p3 = verts[face[2]]

                # Calculate the volume of the tetrahetron (triangle + origin)
                volume_temp = signed_volume_of_triangle(p1, p2, p3)
                volume = volume + volume_temp

            volume_arr = np.append(volume_arr, [[shape.get_id(), round(abs(volume),2)]], axis=0)
    
    print(volume_arr)

    #np.save(SAVED_DATA + 'features.npy', features)


def signed_volume_of_triangle(p1, p2, p3):
    
    # p[0] = x coordinate, p[1] = y coordinate, p[2] = z coordinate
    v321 = p3[0]*p2[1]*p1[2]
    v231 = p2[0]*p3[1]*p1[2]
    v312 = p3[0]*p1[1]*p2[2]
    v132 = p1[0]*p3[1]*p2[2]
    v213 = p2[0]*p1[1]*p3[2]
    v123 = p1[0]*p2[1]*p3[2]

    return 1/6*(-v321 + v231 + v312 - v132 - v213 + v123)