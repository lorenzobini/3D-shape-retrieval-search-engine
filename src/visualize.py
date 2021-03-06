import glfw # If getting error remove this line
from OpenGL.GL import *
import numpy as np
import time


# Define needed global variabels for rotation, translation, zooming
xrot, yrot, xspeed, yspeed = 0, 0, 0, 0
z = -5.0
xas, yas = 0.0, 0.0

# allows switching between panning and rotating. 
rotation = 1
DRAW_EDGES = False

# GLFW action variable
GLFW_PRESS = 1

# Keybind variables
GLFW_KEY_ESCAPE = 256
GLFW_KEY_DELETE = 261
GLFW_KEY_RIGHT = 262
GLFW_KEY_LEFT = 263
GLFW_KEY_DOWN = 264
GLFW_KEY_UP = 265
GLFW_KEY_ZOOM_IN = 45
GLFW_KEY_ZOOM_OUT = 61
GLFW_ENTER = 257
GLFW_KEY_V = 86

i = 0

vertices = None
indices = None
shapes = None
saves = []


def InitGL(width, height):
    """
    Initializes OpenGL window.
    """
    glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading
    glClearColor(1.0, 1.0, 1.0, 1.0)	# This Will Clear The Background Color To Black
    glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
    glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
    glDepthFunc(GL_LEQUAL)				# The Type Of Depth Test To Do
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST) # Really Nice Perspective Calculations 


def visualize(shapes_list):
    """
    Displays the 3D shape within the visualization window.
    ----------------------------
    Args:
        shapes_list (obj: 'list' of obj: 'Shape'): The list of shapes
    """
    global xrot, yrot, shapes, vertices, indices

    # create window to render 3d mesh
    window = create_window()
    InitGL(1280, 720)

    shapes = shapes_list
    vertices = np.array(shapes[i].vertices)
    indices = np.array(shapes[i].faces)
    # set background color of window
    glClearColor(0.9, 0.9, 0.9, 1)

    # the main application loop
    while not glfw.window_should_close(window):
        # create vertices and indices from imported data
        glfw.poll_events()

        width, height = glfw.get_framebuffer_size(window)
        glViewport(0,0,width, height)
        
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)		
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glFrustum(-1.0, 1.0, -1.0, 1.0, 2.5, 20.0)
    
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix() 
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 2)
        # Translate based on the xas, yas, and z values.
        glTranslatef(xas, yas, z)

        glRotatef(xrot, 1.0, 0.0, 0.0)			# Rotate On The X Axis
        glRotatef(yrot, 0.0, 1.0, 0.0)			# Rotate On The Y Axis
        for index in indices:
            # Allows handling of quads and triangles
            glBegin(GL_TRIANGLES)
            glColor3f(0.5,0.5,0.5)
            for number in index[1:]:
                glVertex3f(vertices[number][0], vertices[number][1], vertices[number][2])
            glEnd()

               
        if DRAW_EDGES: 
            for index in indices:
                # Draws the lines for the vertices
                glBegin(GL_LINES)
                glColor3f(0.0,0.0,0.0)
                for number in index[1:]:
                    glVertex3f(vertices[number][0], vertices[number][1], vertices[number][2])
                glEnd()
        glDisable(GL_POLYGON_OFFSET_FILL)
        glPopMatrix()
               
        # Add the speed to the rotation value, is increased by keypressing.
        xrot += xspeed		               
        yrot += yspeed	
        
        glfw.swap_buffers(window)

    # terminate glfw, free up allocated resources
    print(saves)
    glfw.terminate()


def create_window():
    """
    It creates a new window.
    """
    # initializing glfw library
    if not glfw.init():
        raise Exception("glfw cannot be initialized!")
    
    # creating the window
    window = glfw.create_window(1280, 720, "OpenGL window", None, None)

    # Check if window was created
    if not window:
        glfw.terminate()
        raise Exception("glfw window cannot be created!")

    # make the context current
    glfw.make_context_current(window)

    glfw.set_window_size_callback(window, window_resize)

    # Define the keycallback functions
    glfw.set_key_callback(window, key_callback)

    return window


def window_resize(window, width, height):
    """
    It resizes the current window.
    """
    if height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
        height = 1

    glViewport(0, 0, width, height)		# Reset The Current Viewport And Perspective Transformation
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glMatrixMode(GL_MODELVIEW)


def key_callback(window, key, scancode, action, mods):
    '''
    It executes an action based on pressed key.
    '''
    global z, xspeed, yspeed, rotation, xas, yas, vertices, indices, i , DRAW_EDGES, saves
    # avoid thrashing this procedure 
    time.sleep(0.01)

    # Only respond to pressing the button not releasing
    if action == 0: 
        return

    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        print("Shutting down.")
        # shut down our window 
        glfw.set_window_should_close(window, GL_TRUE)

    elif (key == 82 or key == 114) and action == GLFW_PRESS:
        if rotation == 0:
            rotation = 1
            print("You are now in rotation mode.")
        else:
            rotation = 0
            print("You are now in translation mode.")
        

    elif key == GLFW_KEY_DELETE and action == GLFW_PRESS: #Pressing Delete resets the figure
        print("Rotation is back to 0.0 and model is moved to center.")
        xrot = 0.0    # x rotation
        yrot = 0.0    # y rotation
        xspeed = 0.0  # x rotation speed
        yspeed = 0.0  # y rotation speed
        z = -5.0
        xas = 0.0
        yas = 0.0

    # Needed for the rotation, translation and zooming
    elif key == GLFW_KEY_ZOOM_IN:  # move the cube into the distance.
        z -= 0.05
    elif key == GLFW_KEY_ZOOM_OUT:  # move the cube closer.
        z += 0.05
    elif key == GLFW_KEY_UP:  # decrease x rotation speed or translate left along x-axis;
        if rotation == 1:  # Change between rotation mode and translation mode
            xspeed -= 0.2
            print("X-Rotation speed is now at " + str(round(xspeed, 2))+ ".")
        else:
            yas += 0.05 
    elif key == GLFW_KEY_DOWN:  # increase x rotation speed or translate right along x-axis;
        if rotation == 1:
            xspeed += 0.2
            print("X-Rotation speed is now at " + str(round(xspeed, 2))+ ".")
        else:
            yas -= 0.05  
    elif key == GLFW_KEY_LEFT:  # decrease y rotation speed or translate down along the y-axis;
        if rotation == 1:
            yspeed-=0.2
            print("Y-Rotation speed is now at " + str(round(yspeed, 2))+ ".")  
        else:
            xas -= 0.05
    elif key == GLFW_KEY_RIGHT:  # increase y rotation speed or translate up along the x-axis;
        if rotation == 1:
            yspeed += 0.2
            print("Y-Rotation speed is now at " + str(round(yspeed, 2))+ ".") 
        else:
            xas += 0.05 
    elif key == 340:  #Ignore the shift key
        pass
    elif key == GLFW_ENTER:  # For moving through the different meshes
        if i == (len(shapes)-1) or len(shapes) == 1:
            i = 0
        else:
            i += 1
        print("Next model which is: " + str(shapes[i].get_id()))
        vertices = np.array(shapes[i].vertices)
        indices = np.array(shapes[i].faces)
    elif key == GLFW_KEY_V:
        if DRAW_EDGES == False:
            print("Edges are drawn.")
            DRAW_EDGES = True
        else: 
            print("Edges are not drawn.")
            DRAW_EDGES = False
    else:
        print("Key %d pressed. No action there yet.\n"%(key))

