### generate DBSI maps in one panel;

import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
def data_path():

    if not os.path.exists(proj_dir):
        print('project directory does not exist - creating...')
        os.makedirs(proj_dir)
        print('project directory created...')
    else:
        print('project directory already exists')

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------     
def generate_DBSI_maps():

    fig = plt.figure(figsize=(15, 8))

    for i in range(len(DBSI_maps)):
        
        map = DBSI_maps[i]
        
        data = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, map)).get_data()

        ax = fig.add_subplot(2, 4, i+1)
        
        if map == 'b0_map.nii':
            img = ax.imshow(data[:, :, k], cmap='gray', vmin=0, vmax=5, aspect='equal')
            plt.axis('off')
            plt.title(map[:-8], fontsize=fs_1, fontweight='bold')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(img, cax=cax, ticks=[0, 5])
            cbar.ax.set_yticklabels(['0', '5'], fontsize=fs_2, fontweight='bold')

        elif map == 'dti_adc_map.nii':
            img = ax.imshow(data[:, :, k], cmap='gray', vmin=0, vmax=2, aspect='equal')
            plt.axis('off')
            plt.title(map[:-8], fontsize=fs_1, fontweight='bold')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(img, cax=cax, ticks=[0, 2])
            cbar.ax.set_yticklabels(['0', '2'], fontsize=fs_2, fontweight='bold')
            
        else:
            img = ax.imshow(data[:, :, k], interpolation=interp, cmap='jet', vmin=0, vmax=1, aspect='equal')
            plt.axis('off')
            plt.title(map[:-8], fontsize=fs_1, fontweight='bold')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(img, cax=cax, ticks=[0, 1])
            cbar.ax.set_yticklabels(['0', '1'], fontsize=fs_2, fontweight='bold')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.01)
              
    plt.savefig(
                os.path.join(proj_dir, sample_dir, 'maps.png'),
                bbox_inches='tight',
                facecolor=fig.get_facecolor()
                )
    plt.show()
    plt.close()
    
# ----------------------------------------------------------------------------------
# model hyper parameters
# ----------------------------------------------------------------------------------
if __name__ == '__main__':    

    k      = 3                   #choose slice number to show
    fs_1   = 15
    fs_2   = 8
    interp = 'gaussian'

    proj_dir   = r"\\10.39.42.102\temp\2018_Legacy_Project\MRI"
    dbsi_dir   = 'DBSI_results_0.2_0.2_0.8_0.8_2_2'
    sample_dir = 'L008_TN2'
            
    DBSI_maps = [
                 'b0_map.nii',
                 'dti_adc_map.nii',
                 'dti_fa_map.nii',
                 'restricted_ratio_1_map.nii',
                 'restricted_ratio_2_map.nii',
                 'hindered_ratio_map.nii',
                 'water_ratio_map.nii',
                 'fiber_ratio_map.nii'
                 ]

    data_path()
    generate_DBSI_maps()
    print('generate DBSI maps complete')



              
