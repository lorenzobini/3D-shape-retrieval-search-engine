from OpenGL.GL import *
import time
import glfw
# The number of our GLUT window
window = None
ESCAPE = 27 # Keybinding of the escape key

xrot = 0   # x rotation
yrot = 0   # y rotation
xspeed = 0 # x rotation speed
yspeed = 0 # y rotation speed

z = -5.0  # depth into the screen.
xas = 0.0 # x translation
yas = 0.0 # y-translation


rotation = 1 # Define to switch between panning and rotating

# A general OpenGL initialization function.  Sets all of the initial parameters.
def InitGL(Width, Height):	
    glShadeModel(GL_SMOOTH)				# Enables Smooth Color Shading
    glClearColor(0.0, 0.0, 0.0, 0.5)	# This Will Clear The Background Color To Black
    glClearDepth(1.0)					# Enables Clearing Of The Depth Buffer
    glEnable(GL_DEPTH_TEST)				# Enables Depth Testing
    glDepthFunc(GL_LEQUAL)				# The Type Of Depth Test To Do
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST) # Really Nice Perspective Calculations 
    return True


# The function called when our window is resized 
def ReSizeGLScene(Width, Height):
	if Height == 0:						# Prevent A Divide By Zero If The Window Is Too Small 
		Height = 1

	glViewport(0, 0, Width, Height)		# Reset The Current Viewport And Perspective Transformation
	glMatrixMode(GL_PROJECTION)
	glLoadIdentity()
	# // field of view, aspect ratio, near and far
	# This will squash and stretch our objects as the window is resized.
	gluPerspective(45.0, float(Width)/float(Height), 0.1, 100.0)

	glMatrixMode(GL_MODELVIEW)
	glLoadIdentity()

# Function to print the text onto the screen
def glPrint(x, y, text):
    glColor3f(1.0, 1.0, 1.0)
    glWindowPos2f(x, y)
    for ch in text:
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, ord(ch))

# The main drawing function.
def DrawGLScene():
    global xrot, yrot

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)		
    glLoadIdentity()				

    # Translate based on the xas, yas, and z values.
    glTranslatef(xas, yas, z)       


    glRotatef(xrot,1.0,0.0,0.0)			# Rotate On The X Axis
    glRotatef(yrot,0.0,1.0,0.0)			# Rotate On The Y Axis

    # Code to draw a triangle for testing, 
    # TODO: change with code to display mesh
    glBegin(GL_TRIANGLES)

    glColor3f(1.0,0.0,0.0)			# Red
    glVertex3f( 0.0, 1.0, 0.0)		# Top Of Triangle(Front)
    glColor3f(0.0,1.0,0.0)			# Green
    glVertex3f(-1.0,-1.0, 1.0)		# Left Of Triangle(Front)
    glColor3f(0.0,0.0,1.0)			# Blue
    glVertex3f( 1.0,-1.0, 1.0)  
    glColor3f(1.0,0.0,0.0)			# Red
    glVertex3f( 0.0, 1.0, 0.0)		# Top Of Triangle(Right)
    glColor3f(0.0,0.0,1.0)			# Blue
    glVertex3f( 1.0,-1.0, 1.0)		# Left Of Triangle(Right)
    glColor3f(0.0,1.0,0.0)			# Green
    glVertex3f( 1.0,-1.0, -1.0)		# Right     
    glColor3f(1.0,0.0,0.0)			# Red
    glVertex3f( 0.0, 1.0, 0.0)		# Top Of Triangle(Back)
    glColor3f(0.0,1.0,0.0)			# Green
    glVertex3f( 1.0,-1.0, -1.0)		# Left Of Triangle(Back)
    glColor3f(0.0,0.0,1.0)			# Blue
    glVertex3f(-1.0,-1.0, -1.0)		# Right Of 
    
    
    glColor3f(1.0,0.0,0.0)			# Red
    glVertex3f( 0.0, 1.0, 0.0)		# Top Of Triangle(Left)
    glColor3f(0.0,0.0,1.0)			# Blue
    glVertex3f(-1.0,-1.0,-1.0)		# Left Of Triangle(Left)
    glColor3f(0.0, 1.0, 0.0)		# Green
    glVertex3f(-1.0,-1.0, 1.0)		# Right Of Triangle(Left)
    glEnd()	    
    
    # Add the speed to the rotation value, is increased by keypressing.
    xrot += xspeed		               
    yrot += yspeed		               

    # Change into appropriate text
    glPrint(10, 75, "X-rotation speed: " + str(xspeed) + "     Y-rotation speed:" + str(yspeed))
    glPrint(10, 55, "R/r:Switch between rotating and panning.        Delete: Reset the mesh to the initial position.      Esc: Close application.")
    glPrint(10, 40 ,"Left arrow: Move left/Decrease x-rotation.      Right Arrow: Move Right/Increase x-rotation.")
    glPrint(10, 25, "Up arrow: Move up/Increase y-rotation.          Down Arrow: Move down/Decrease y-rotation")
    glPrint(10, 10, "Pg up: Zoom-in                                  Pg dn: Zoom-out")
    # Add text to glut window
    glutSwapBuffers()


# The function called whenever a normal key is pressed. 
def keyPressed(key, x, y):
    global rotation
    
    key = ord(key)

    # avoid thrashing this procedure 
    time.sleep(0.01)

    if key == ESCAPE:
        # shut down our window 
	    glutDestroyWindow(window)
        # exit the program...normal termination. 
	    sys.exit()

    elif key == 82 or key == 114:
        print("Pressed rotation key")
        if rotation == 0:
            rotation = 1
        else:
            rotation = 0

    elif key == 127: # Pressing Delete resets the figure
        global xrot, yrot, xspeed, yspeed, z, xas, yas
        xrot = 0   # x rotation
        yrot = 0   # y rotation
        xspeed = 0 # x rotation speed
        yspeed = 0 # y rotation speed
        z = -5.0
        xas = 0.0
        yas = 0.0

    else:
        print("Key %d pressed. No action there yet.\n"%(key))
      
      

# The function called whenever a normal key is pressed. 
def specialKeyPressed(key, x, y):
    global z, xspeed, yspeed, rotation, xas, yas   
    # avoid thrashing this procedure 
    time.sleep(0.01)    

    if key == GLUT_KEY_PAGE_UP: # move the cube into the distance.
        z-=0.02 
    elif key == GLUT_KEY_PAGE_DOWN: # move the cube closer.
        z+=0.02 
    elif key == GLUT_KEY_UP: # decrease x rotation speed or translate left along x-axis;
        if rotation == 1: # Change between rotation mode and translation mode
            xspeed-=0.01
        elif rotation == 0:
            yas += 0.02 
    elif key == GLUT_KEY_DOWN: # increase x rotation speed or translate right along x-axis;
        if rotation == 1:
            xspeed+=0.01
        elif rotation == 0:
            yas -= 0.02   
    elif key == GLUT_KEY_LEFT: # decrease y rotation speed or translate down along the y-axis;
        if rotation == 1:
            yspeed-=0.01  
        elif rotation == 0:
            xas -= 0.02  
    elif key == GLUT_KEY_RIGHT: # increase y rotation speed or translate up along the x-axis;
        if rotation == 1:
            yspeed+=0.01  
        elif rotation == 0:
            xas += 0.02 

def key_callback(window, key, action):
    global z, xspeed, yspeed, rotation, xas, yas   
    # avoid thrashing this procedure 
    time.sleep(0.01)
    
    key = ord(key)

    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        # shut down our window 
	    glfw.SetWindowShouldClose(window, GL_TRUE)

    # elif key == 82 or key == 114:
    #     print("Pressed rotation key")
    #     if rotation == 0:
    #         rotation = 1
    #     else:
    #         rotation = 0

    # elif key == 127: # Pressing Delete resets the figure
    #     global xrot, yrot, xspeed, yspeed, z, xas, yas
    #     xrot = 0   # x rotation
    #     yrot = 0   # y rotation
    #     xspeed = 0 # x rotation speed
    #     yspeed = 0 # y rotation speed
    #     z = -5.0
    #     xas = 0.0
    #     yas = 0.0

    else:
        print("Key %d pressed. No action there yet.\n"%(key))
