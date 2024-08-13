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
X_VV = data_X[2,:,:]
X_HV = data_X[6,:,:]
X_HH = data_X[10,:,:]
C_VV = data_C[2,:,:]
C_HV = data_C[6,:,:]
C_HH = data_C[10,:,:]
L_VV = data_L[2,:,:]
L_HV = data_L[6,:,:]
L_HH = data_L[10,:,:]

# ---------- #

# create 8-bit false_color RGB

# new 8-bit min/max values
new_min = 0
new_max = 255

# linear map from sigma0 in dB to new_min and new_max
X_HH_scaled = (X_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
X_HV_scaled = (X_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
X_VV_scaled = (X_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min
C_HH_scaled = (C_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
C_HV_scaled = (C_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
C_VV_scaled = (C_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min
L_HH_scaled = (L_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
L_HV_scaled = (L_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
L_VV_scaled = (L_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min

# clip values
X_HH_scaled = np.clip(X_HH_scaled, new_min, new_max)
X_HV_scaled = np.clip(X_HV_scaled, new_min, new_max)
X_VV_scaled = np.clip(X_VV_scaled, new_min, new_max)
C_HH_scaled = np.clip(C_HH_scaled, new_min, new_max)
C_HV_scaled = np.clip(C_HV_scaled, new_min, new_max)
C_VV_scaled = np.clip(C_VV_scaled, new_min, new_max)
L_HH_scaled = np.clip(L_HH_scaled, new_min, new_max)
L_HV_scaled = np.clip(L_HV_scaled, new_min, new_max)
L_VV_scaled = np.clip(L_VV_scaled, new_min, new_max)

# stack scaled channels to false-color RGB
X_RGB = np.stack((X_HV_scaled,X_VV_scaled,X_HH_scaled),2)
C_RGB = np.stack((C_HV_scaled,C_VV_scaled,C_HH_scaled),2)
L_RGB = np.stack((L_HV_scaled,L_VV_scaled,L_HH_scaled),2)

VV_RGB = np.stack((X_VV_scaled,C_VV_scaled,L_VV_scaled),2)
HV_RGB = np.stack((X_HV_scaled,C_HV_scaled,L_HV_scaled),2)
HH_RGB = np.stack((X_HH_scaled,C_HH_scaled,L_HH_scaled),2)


# ---------- #

# visualize all polarization channels
fig, axes = plt.subplots(3, 3, sharex=True, sharey=True)
axes = axes.ravel()

axes[0].imshow(X_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[1].imshow(X_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[2].imshow(X_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[3].imshow(C_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[4].imshow(C_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[5].imshow(C_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[6].imshow(L_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[7].imshow(L_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[8].imshow(L_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')

axes[0].set_title('X-band VV')
axes[1].set_title('X-band HV')
axes[2].set_title('X-band HH')
axes[3].set_title('C-band VV')
axes[4].set_title('C-band HV')
axes[5].set_title('C-band HH')
axes[6].set_title('L-band VV') 
axes[7].set_title('L-band HV')
axes[8].set_title('L-band HH')

# ---------- #

# visualize false-color per wavelength
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.ravel()

axes[0].imshow(X_RGB/255)
axes[1].imshow(C_RGB/255)
axes[2].imshow(L_RGB/255)

axes[0].set_title('X-band (HV, VV, HH)')
axes[1].set_title('C-band (HV, VV, HH)')
axes[2].set_title('L-band (HV, VV, HH)')

# ---------- #

# visualize false-color per polarization
fig, axes = plt.subplots(3, 1, sharex=True, sharey=True)
axes = axes.ravel()

axes[0].imshow(VV_RGB/255)
axes[1].imshow(HV_RGB/255)
axes[2].imshow(HH_RGB/255)

axes[0].set_title('VV (X, C, L)')
axes[1].set_title('HV (X, C, L)')
axes[2].set_title('HH (X, C, L)')
