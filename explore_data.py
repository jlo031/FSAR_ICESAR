# ---- This is <explore_data.py> ---- # 

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #

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

# create 8-bit false_color RGB

# new 8-bit min/max values
new_min = 0
new_max = 255

# linear map from sigma0 in dB to new_min and new_max
X_sigma_HH_scaled = (X_sigma_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
X_sigma_HV_scaled = (X_sigma_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
X_sigma_VV_scaled = (X_sigma_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min
C_sigma_HH_scaled = (C_sigma_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
C_sigma_HV_scaled = (C_sigma_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
C_sigma_VV_scaled = (C_sigma_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min
L_sigma_HH_scaled = (L_sigma_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
L_sigma_HV_scaled = (L_sigma_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
L_sigma_VV_scaled = (L_sigma_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min

# clip values
X_sigma_HH_scaled = np.clip(X_sigma_HH_scaled, new_min, new_max)
X_sigma_HV_scaled = np.clip(X_sigma_HV_scaled, new_min, new_max)
X_sigma_VV_scaled = np.clip(X_sigma_VV_scaled, new_min, new_max)
C_sigma_HH_scaled = np.clip(C_sigma_HH_scaled, new_min, new_max)
C_sigma_HV_scaled = np.clip(C_sigma_HV_scaled, new_min, new_max)
C_sigma_VV_scaled = np.clip(C_sigma_VV_scaled, new_min, new_max)
L_sigma_HH_scaled = np.clip(L_sigma_HH_scaled, new_min, new_max)
L_sigma_HV_scaled = np.clip(L_sigma_HV_scaled, new_min, new_max)
L_sigma_VV_scaled = np.clip(L_sigma_VV_scaled, new_min, new_max)

# stack scaled channels to false-color RGB
X_sigma_RGB = np.stack((X_sigma_HV_scaled,X_sigma_VV_scaled,X_sigma_HH_scaled),2)
C_sigma_RGB = np.stack((C_sigma_HV_scaled,C_sigma_VV_scaled,C_sigma_HH_scaled),2)
L_sigma_RGB = np.stack((L_sigma_HV_scaled,L_sigma_VV_scaled,L_sigma_HH_scaled),2)

sigma_VV_RGB = np.stack((X_sigma_VV_scaled,C_sigma_VV_scaled,L_sigma_VV_scaled),2)
sigma_HV_RGB = np.stack((X_sigma_HV_scaled,C_sigma_HV_scaled,L_sigma_HV_scaled),2)
sigma_HH_RGB = np.stack((X_sigma_HH_scaled,C_sigma_HH_scaled,L_sigma_HH_scaled),2)

# ---------- #

# linear map from gamma0 in dB to new_min and new_max
X_gamma_HH_scaled = (X_gamma_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
X_gamma_HV_scaled = (X_gamma_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
X_gamma_VV_scaled = (X_gamma_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min
C_gamma_HH_scaled = (C_gamma_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
C_gamma_HV_scaled = (C_gamma_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
C_gamma_VV_scaled = (C_gamma_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min
L_gamma_HH_scaled = (L_gamma_HH - (hh_vmin)) * ((new_max - new_min) / ((hh_vmax) - (hh_vmin))) + new_min
L_gamma_HV_scaled = (L_gamma_HV - (hv_vmin)) * ((new_max - new_min) / ((hv_vmax) - (hv_vmin))) + new_min
L_gamma_VV_scaled = (L_gamma_VV - (hv_vmin)) * ((new_max - new_min) / ((vv_vmax) - (vv_vmin))) + new_min

# clip values
X_gamma_HH_scaled = np.clip(X_gamma_HH_scaled, new_min, new_max)
X_gamma_HV_scaled = np.clip(X_gamma_HV_scaled, new_min, new_max)
X_gamma_VV_scaled = np.clip(X_gamma_VV_scaled, new_min, new_max)
C_gamma_HH_scaled = np.clip(C_gamma_HH_scaled, new_min, new_max)
C_gamma_HV_scaled = np.clip(C_gamma_HV_scaled, new_min, new_max)
C_gamma_VV_scaled = np.clip(C_gamma_VV_scaled, new_min, new_max)
L_gamma_HH_scaled = np.clip(L_gamma_HH_scaled, new_min, new_max)
L_gamma_HV_scaled = np.clip(L_gamma_HV_scaled, new_min, new_max)
L_gamma_VV_scaled = np.clip(L_gamma_VV_scaled, new_min, new_max)

# stack scaled channels to false-color RGB
X_gamma_RGB = np.stack((X_gamma_HV_scaled,X_gamma_VV_scaled,X_gamma_HH_scaled),2)
C_gamma_RGB = np.stack((C_gamma_HV_scaled,C_gamma_VV_scaled,C_gamma_HH_scaled),2)
L_gamma_RGB = np.stack((L_gamma_HV_scaled,L_gamma_VV_scaled,L_gamma_HH_scaled),2)

gamma_VV_RGB = np.stack((X_gamma_VV_scaled,C_gamma_VV_scaled,L_gamma_VV_scaled),2)
gamma_HV_RGB = np.stack((X_gamma_HV_scaled,C_gamma_HV_scaled,L_gamma_HV_scaled),2)
gamma_HH_RGB = np.stack((X_gamma_HH_scaled,C_gamma_HH_scaled,L_gamma_HH_scaled),2)

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #

# visualize all polarization channels
# sigma0

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=((14,7)))
axes = axes.ravel()

axes[0].imshow(X_sigma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[1].imshow(X_sigma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[2].imshow(X_sigma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[3].imshow(C_sigma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[4].imshow(C_sigma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[5].imshow(C_sigma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[6].imshow(L_sigma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[7].imshow(L_sigma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[8].imshow(L_sigma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')

axes[0].set_title('X-band sigma VV')
axes[1].set_title('X-band sigma HV')
axes[2].set_title('X-band sigma HH')
axes[3].set_title('C-band sigma VV')
axes[4].set_title('C-band sigma HV')
axes[5].set_title('C-band sigma HH')
axes[6].set_title('L-band sigma VV') 
axes[7].set_title('L-band sigma HV')
axes[8].set_title('L-band sigma HH')

# ---------- #

# visualize all polarization channels
# gamma0

fig, axes = plt.subplots(3, 3, sharex=True, sharey=True, figsize=((14,7)))
axes = axes.ravel()

axes[0].imshow(X_gamma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[1].imshow(X_gamma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[2].imshow(X_gamma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[3].imshow(C_gamma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[4].imshow(C_gamma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[5].imshow(C_gamma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[6].imshow(L_gamma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[7].imshow(L_gamma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[8].imshow(L_gamma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')

axes[0].set_title('X-band gamma VV')
axes[1].set_title('X-band gamma HV')
axes[2].set_title('X-band gamma HH')
axes[3].set_title('C-band gamma VV')
axes[4].set_title('C-band gamma HV')
axes[5].set_title('C-band gamma HH')
axes[6].set_title('L-band gamma VV') 
axes[7].set_title('L-band gamma HV')
axes[8].set_title('L-band gamma HH')

# ---------- #

# visualize false-color per wavelength
# sigma0

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=((6,7)))
axes = axes.ravel()

axes[0].imshow(X_sigma_RGB/255)
axes[1].imshow(C_sigma_RGB/255)
axes[2].imshow(L_sigma_RGB/255)

axes[0].set_title('X-band (sigma HV, VV, HH)')
axes[1].set_title('C-band (sigma HV, VV, HH)')
axes[2].set_title('L-band (sigma HV, VV, HH)')

# ---------- #

# visualize false-color per wavelength
# gamma0

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=((6,7)))
axes = axes.ravel()

axes[0].imshow(X_gamma_RGB/255)
axes[1].imshow(C_gamma_RGB/255)
axes[2].imshow(L_gamma_RGB/255)

axes[0].set_title('X-band (gamma HV, VV, HH)')
axes[1].set_title('C-band (gamma HV, VV, HH)')
axes[2].set_title('L-band (gamma HV, VV, HH)')

# ---------- #

# visualize false-color per polarization
# sigma0

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=((6,7)))
axes = axes.ravel()

axes[0].imshow(sigma_VV_RGB/255)
axes[1].imshow(sigma_HV_RGB/255)
axes[2].imshow(sigma_HH_RGB/255)

axes[0].set_title('sigma VV (X, C, L)')
axes[1].set_title('sigma HV (X, C, L)')
axes[2].set_title('sigma HH (X, C, L)')

# ---------- #

# visualize false-color per polarization
# gamma0

fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=((6,7)))
axes = axes.ravel()

axes[0].imshow(gamma_VV_RGB/255)
axes[1].imshow(gamma_HV_RGB/255)
axes[2].imshow(gamma_HH_RGB/255)

axes[0].set_title('gamma VV (X, C, L)')
axes[1].set_title('gamma HV (X, C, L)')
axes[2].set_title('gamma HH (X, C, L)')

# ---------- #

# visualize sigma0 vs gamma0
# X-band

fig, axes = plt.subplots (2,3, sharex=True, sharey=True, figsize=((12,6)))
axes = axes.ravel()

axes[0].imshow(X_gamma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[1].imshow(X_gamma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[2].imshow(X_gamma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[3].imshow(X_sigma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[4].imshow(X_sigma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[5].imshow(X_sigma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')

axes[0].set_title('X-band gamma VV')
axes[1].set_title('X-band gamma HV')
axes[2].set_title('X-band gamma HH')
axes[3].set_title('X-band sigma VV')
axes[4].set_title('X-band sigma HV')
axes[5].set_title('X-band sigma HH')

# ---------- #

# visualize sigma0 vs gamma0
# C-band

fig, axes = plt.subplots (2,3, sharex=True, sharey=True, figsize=((12,6)))
axes = axes.ravel()

axes[0].imshow(C_gamma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[1].imshow(C_gamma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[2].imshow(C_gamma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[3].imshow(C_sigma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[4].imshow(C_sigma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[5].imshow(C_sigma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')

axes[0].set_title('C-band gamma VV')
axes[1].set_title('C-band gamma HV')
axes[2].set_title('C-band gamma HH')
axes[3].set_title('C-band sigma VV')
axes[4].set_title('C-band sigma HV')
axes[5].set_title('C-band sigma HH')


# ---------- #

# visualize sigma0 vs gamma0
# L-band

fig, axes = plt.subplots (2,3, sharex=True, sharey=True, figsize=((12,6)))
axes = axes.ravel()

axes[0].imshow(L_gamma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[1].imshow(L_gamma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[2].imshow(L_gamma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')
axes[3].imshow(L_sigma_VV, vmin=vv_vmin, vmax=vv_vmax, cmap='gray')
axes[4].imshow(L_sigma_HV, vmin=hv_vmin, vmax=hv_vmax, cmap='gray')
axes[5].imshow(L_sigma_HH, vmin=hh_vmin, vmax=hh_vmax, cmap='gray')

axes[0].set_title('L-band gamma VV')
axes[1].set_title('L-band gamma HV')
axes[2].set_title('L-band gamma HH')
axes[3].set_title('L-band sigma VV')
axes[4].set_title('L-band sigma HV')
axes[5].set_title('L-band sigma HH')

# ------------------------------------------------------------------------ #
# ------------------------------------------------------------------------ #

# ---- End of <explore_data.py> ---- # 
