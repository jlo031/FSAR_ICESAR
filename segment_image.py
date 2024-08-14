import json
import pathlib

import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt



# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #

def load_folder_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    f.close()
    DATA_DIR = pathlib.Path(config['data_dir'])

    return DATA_DIR

# ---------- #

def load_plot_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    f.close()
    vv_vmin = config['vv_vmin']
    vv_vmax = config['vv_vmax']
    vh_vmin = config['vh_vmin']
    vh_vmax = config['vh_vmax']
    hv_vmin = config['hv_vmin']
    hv_vmax = config['hv_vmax']
    hh_vmin = config['hh_vmin']
    hh_vmax = config['hh_vmax']

    return vv_vmin, vv_vmax, vh_vmin, vh_vmax, hv_vmin, hv_vmax, hh_vmin, hh_vmax

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #

# load config files
DATA_DIR = load_folder_config('config/folder_config.json')
vv_vmin, vv_vmax, vh_vmin, vh_vmax, hv_vmin, hv_vmax, hh_vmin, hh_vmax = load_plot_config('config/plot_config.json')

# ---------- #

# define test images
##test_image_X = 'BaffinBay_1702_t01Xx10_ml_Stack_warped_cut.dat'
test_image_X = 'BaffinBay_1702_t01Cx10_ml_Stack_cut'
test_image_C = 'BaffinBay_1702_t01Cx10_ml_Stack_cut'
test_image_L = 'BaffinBay_1702_t01Lx10_ml_Stack_warped_cut.dat'

# build full paths to test image data
img_path_X = DATA_DIR / test_image_X
img_path_C = DATA_DIR / test_image_C
img_path_L = DATA_DIR / test_image_L

# load data
data_X = gdal.Open(img_path_X.as_posix()).ReadAsArray()
data_C = gdal.Open(img_path_C.as_posix()).ReadAsArray()
data_L = gdal.Open(img_path_L.as_posix()).ReadAsArray()

# extract sigma0 channels
X_sigma_VV = data_X[2,:,:]
X_sigma_HV = data_X[6,:,:]
X_sigma_HH = data_X[10,:,:]
C_sigma_VV = data_C[2,:,:]
C_sigma_HV = data_C[6,:,:]
C_sigma_HH = data_C[10,:,:]
L_sigma_VV = data_L[2,:,:]
L_sigma_HV = data_L[6,:,:]
L_sigma_HH = data_L[10,:,:]

# extract gamma0 channels
X_gamma_VV = data_X[4,:,:]
X_gamma_HV = data_X[8,:,:]
X_gamma_HH = data_X[13,:,:]
C_gamma_VV = data_C[4,:,:]
C_gamma_HV = data_C[8,:,:]
C_gamma_HH = data_C[13,:,:]
L_gamma_VV = data_L[4,:,:]
L_gamma_HV = data_L[8,:,:]
L_gamma_HH = data_L[13,:,:]

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #

# stack intensities to data cubes per wavelength
X_sigma = np.stack((X_sigma_VV, X_sigma_HV, X_sigma_HH),2)
C_sigma = np.stack((C_sigma_VV, C_sigma_HV, C_sigma_HH),2)
L_sigma = np.stack((L_sigma_VV, L_sigma_HV, L_sigma_HH),2)
X_gamma = np.stack((X_gamma_VV, X_gamma_HV, X_gamma_HH),2)
C_gamma = np.stack((C_gamma_VV, C_gamma_HV, C_gamma_HH),2)
L_gamma = np.stack((L_gamma_VV, L_gamma_HV, L_gamma_HH),2)

# train kmeans segmentation models









# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #

# ---- End of <segment_image.py> ---- # 
