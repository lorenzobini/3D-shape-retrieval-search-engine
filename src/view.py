import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
from main import import_data
import numpy as np

vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;

void main()
{
    gl_Position = vec4(a_position, 1.0);
}
"""

def main():
    # create window to render 3d mesh
    window = create_window()

    # import data 
    shapes, shape_labels = import_data()

    # create vertices and indices from imported data
    vertices = np.array(shapes[0][0], dtype=np.float32)
    indices = np.array(shapes[0][1], dtype=np.uint32)

    # runs vertex and fragment shaders
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER))

    # vertex buffer object
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    # element buffer object
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)

    # refers to the location=0 in vertex_src
    glEnableVertexAttribArray(0)
    # tells how to read the buffer, each vertex contains 12 bytes (3 vertices * 4 bytes)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0)) # for 3 vertices

    glUseProgram(shader)
    # set background color of window
    glClearColor(0, 0.1, 0.1, 1)

    # the main application loop
    while not glfw.window_should_close(window):
        glfw.poll_events()
        
        glClear(GL_COLOR_BUFFER_BIT)

        glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(window)

    # terminate glfw, free up allocated resources
    glfw.terminate()

def create_window():
    # initializing glfw library
    if not glfw.init():
        raise Exception("glfw cannot be initialized!")
    
    # creating the window
    window = glfw.create_window(1280, 720, "OpenGL window", None, None)

    # Check if window was created
    if not window:
        glfw.terminate()
        raise Exception("glfw window cannot be created!")

    # set window's position
    glfw.set_window_pos(window, 400, 200)
    glfw.set_window_size_callback(window, window_resize)

    # make the context current
    glfw.make_context_current(window)

    return window

def window_resize(window, width, height):
    glViewport(0, 0, width, height)


main()