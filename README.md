# INFOMR_system

> Created by:  
> Lars van Asseldonk (*6970907*) \
> Lorenzo Bini (*6924093*) \
> Lisa Salomons (*6951864*) \
> In fulfilment or the requirements of the INFOMR course at Utrecht University.

## Getting Started
Create a folder called "data" in `C:\Users\User\..`, make sure this data folder contains the Princeton shape benchmark dataset and a folder called cache, that contains a folder called processed_data. 

Such that when you open the data folder its path is `C:\Users\user\data` containing the subfolders: benchmark and cache, with the following paths: `C:\Users\user\data\benchmark` and `C:\Users\user\data\cache`. \
The cache folder contains the processed_data folder resulting in the path: `C:\Users\user\data\cache\processed_data`

In the cache map, there should also be an `exclude.txt` containing the numbers of the models our normalization method was not able to normalize properly. If not in the cache file the models are included in the database and will have a significant impact on the feature calculation and overall performance. 

The datapath to the cache folder can also be changed this can be done in `settings.py` by changing the **SAVED_DATA** path. 
Then proceed to run `main.py`.

The first run takes a long time as it normalizes the shapes and computes the features, saving it all in their own cache file. \
For the following runs, the models and features are reloaded from the cache files in the cache folder. When running the code make sure to read through `setting.py` to ensure that the right settings for your goal are selected. 


### Prerequisite
Required for the project are the following programs
 - [Python](https://www.python.org/downloads/release/python-370/) 
 - [GLFW](https://www.glfw.org/)\
 `pip install glfw `
 or
 `conda install -c conda-forge glfw`
 - [PyOpenGL](http://pyopengl.sourceforge.net/)\
 `pip install PyOpenGL `
 or
 `conda install -c anaconda pyopengl`
 - [MatPlotLib](https://matplotlib.org/users/installing.html)\
 `pip install matplotlib`
 or 
 `conda install -c conda-forge matplotlib`
 - [Numpy](https://numpy.org/doc/stable/user/setting-up.html)\
 `pip install numpy`
 or 
 `conda install numpy`
 - [Open3d](http://www.open3d.org/docs/release/)\
 `pip install open3d`
 or
 `conda install -c open3d-admin open3d`
 - [Trimesh](https://trimsh.org/trimesh.html)\
 `pip install trimesh`
 or 
 `conda install -c conda-forge trimesh`
 - [Annoy](https://pypi.org/project/annoy/)   
 `pip install annoy`
 or 
 `conda install -c conda-forge python-annoy`

## Usage 
From your Python interpreter run `main.py` to run the main file with the selected settings in `settings.py`.

### Keybindings for zooming, panning and rotation in the viewe
|                                           |                                                 |                        |
|-------------------------------------------|-------------------------------------------------|------------------------|
| R/r: Switch between rotating and panning  | Delete: Reset the mesh to the initial position  | Esc: Close application |
| Left Arrow: Move left/Decrease x-rotation | Right Arrow: Move right/Increase x-rotation     |                        |
| Up Arrow: Move up/Increase y-rotation     | Down Arrow: Move down/Decrease y-rotation       |                        |
| +: Zoom-in                                | -: Zoom-out                                     |                        |
| Enter: Go to next model                   | v: Toggle vertices                              |                        |

### Evaluation script
Run `evaluation.py` from your python interpreter command line.
Before running the evaluation script you should be sure to set one of the following to true: *USE_DISTANCE_METRIC* or *USE_RNN* in the settings.py file. If *USE_DISTANCE_METRIC* is set to **True** make sure that *USE_RNN*is set to **False** and vice versa. 
