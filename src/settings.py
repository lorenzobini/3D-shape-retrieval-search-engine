import os
import warnings

class Settings:

    '''
    FORCE IMPORT: if TRUE runs the offline part independently from the presence of normalised shapes
    in cache or already computed features
    '''
    FORCE_IMPORT: bool = False

    '''
    DISPLAY BATCH BEFORE NORMALISATION: if TRUE displays the shapes of every batch before being normalised
    '''
    DISPLAY_BATCH_BN: bool = False

    '''
    DISPLAY BATCH AFTER NORMALISATION: if TRUE displays the shapes of every batch after being normalised
    '''
    DISPLAY_BATCH_AN: bool = False

    '''
    DISPLAY PROGRESS: if TRUE prints computation progress in console
    '''
    DISPLAY_PROGRESS: bool = True

    '''
    INFINITE QUERY LOOP: if TRUE runs the query loop indefinite times until the program is forced to stop
    '''
    INFINITE_QUERY_LOOP: bool = False

    '''
    USE CUSTOM DISTANCE: if TRUE finds and displays similar shapes obtained by using custom distance function
    '''
    USE_CUSTOM_DISTANCE: bool = True

    '''
    USE K NEAREST NEIGHBORS: if TRUE finds and displays similar shapes obtained by using K-nearest neighbors
    '''
    USE_KNN: bool = True

    '''
    USE R NEAREST NEIGHBORS: if TRUE finds and displays similar shapes obtained by using R-nearest neighbors
    '''
    USE_RNN: bool = True

    '''
    K NEAREST NEIGHBORS SIZE: defines the number of shapes to return when computing KNN
    Default: 11
    '''
    KNN_SIZE: int = 11

    '''
    R NEAREST NEIGHBORS RANGE: defines the distance range inside which the shapes have to fall in order to 
    being returned when computing RNN
    ----- Requirement: 0 < RNN_RANGE <= 1
    Default: 0.5 
    '''
    RNN_RANGE: float = 0.5

    '''
    CATEGORIES: defines the number of categories into which dividing the feature space during ANN
    It reflects the number of categories of the shape labels
    ----- BASE:    131
    ----- COARSE1: 42
    ----- COARSE2: 7
    ----- COARSE3: 2
    '''
    CATEGORIES: int = 42

    '''
    NORMALISATION SETTINGS: defines the three vertices values used for normalisation
    Default AVG_VERTS: 2000
    Default Q1_VERTS: 1000
    Default Q3_VERTS: 3000 
    '''
    AVG_VERTS: int = 2000
    Q1_VERTS: int = 1000
    Q3_VERTS: int = 3000


    '''
    ##################### DO NOT CHANGE
    '''
    DATA_PATH: str = os.path.join(os.getcwd(), 'data') + os.sep
    DATA_SHAPES_PRICETON: str = DATA_PATH + 'benchmark' + os.sep + 'db' + os.sep
    DATA_CLASSIFICATION_PRINCETON = DATA_PATH + 'benchmark' + os.sep +\
                                    'classification' + os.sep + 'v1' + os.sep + 'coarse1' + os.sep
    SAVED_DATA: str = DATA_PATH + 'cache' + os.sep
    NORMALIZED_DATA: str = SAVED_DATA + 'processed_data' + os.sep
    '''
    ####################################
    '''

    def __init__(self):
        # Sanity check
        if self.RNN_RANGE > 1 or self.RNN_RANGE < 0:
            raise ValueError("Settings: RNN RANGE value not valid.")

        if self.CATEGORIES is not 131 and\
            self.CATEGORIES is not 42 and\
            self.CATEGORIES is not 7 and\
            self.CATEGORIES is not 2:
            warnings.warn("Settings: CATEGORIES value not optimal.", UserWarning)

        if "base" in self.DATA_CLASSIFICATION_PRINCETON and self.CATEGORIES is not 131:
            warnings.warn("Settings: CATEGORIES value does not match with classification set.", UserWarning)
        elif "coarse1" in self.DATA_CLASSIFICATION_PRINCETON and self.CATEGORIES is not 42:
            warnings.warn("Settings: CATEGORIES value does not match with classification set.", UserWarning)
        elif "coarse2" in self.DATA_CLASSIFICATION_PRINCETON and self.CATEGORIES is not 7:
            warnings.warn("Settings: CATEGORIES value does not match with classification set.", UserWarning)
        elif "coarse3" in self.DATA_CLASSIFICATION_PRINCETON and self.CATEGORIES is not 2:
            warnings.warn("Settings: CATEGORIES value does not match with classification set.", UserWarning)
