### generate DBSI maps in one panel;
### created by YZZ on 9/6/2018;

import numpy as np
import nibabel as nib
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size

def generating_DBSI_maps():

    print("data parameters and folder path")
    # parameters
    k = 3                                           # choose slice number to show
    cmap_1 = 'gray'
    cmap_2 = 'jet'
    fontsize_1 = 10
    fontsize_2 = 12
    interpolation = 'gaussian'

    # data folder path
    proj_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2018_HTAN\MRI'
    dbsi_dir = 'DBSI_results_0.1_0.1_0.8_0.8_1.5_1.5'

    sample_dir = 's_2019030602'

    print('generate DBSI maps: start...')

    fig = plt.figure(figsize=(15, 8))

    # b0 map
    ax = fig.add_subplot(2, 4, 1)
    img = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, 'b0_map.nii')).get_data()
    plt.imshow(img[:, :, k], cmap=cmap_1, vmin=0, vmax=1, aspect='equal')
    ax.set_title('b0', fontsize=fontsize_1, fontweight='bold')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 1, 2])
    cbar.set_clim(0, 2)
    cbar.ax.set_yticklabels(['0', '1', '2'], fontsize=fontsize_2, fontweight='bold')

    # ADC map
    ax = fig.add_subplot(2, 4, 2)
    img = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, 'dti_adc_map.nii')).get_data()
    plt.imshow(img[:, :, k], cmap=cmap_1, vmin=0, vmax=2, aspect='equal')
    ax.set_title('DTI ADC', fontsize=fontsize_1, fontweight='bold')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 1, 2])
    cbar.set_clim(0, 2)
    cbar.ax.set_yticklabels(['0', '1', '2'], fontsize=fontsize_2, fontweight='bold')

    # DTI FA map
    ax = fig.add_subplot(2, 4, 3)
    img = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, 'dti_fa_map.nii')).get_data()
    plt.imshow(img[:, :, k], interpolation=interpolation, cmap=cmap_2, vmin=0, vmax=1, aspect='equal')
    plt.axis('off')
    plt.title('DTI FA', fontsize=fontsize_1, fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.set_clim(0, 1)
    cbar.ax.set_yticklabels(['0', '0.5', '1'], fontsize=fontsize_2, fontweight='bold')

    # DBSI Highly Restricted Fraction Map
    ax = fig.add_subplot(2, 4, 4)
    img = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, 'restricted_ratio_1_map.nii')).get_data()
    ax.imshow(img[:,:,k], interpolation=interpolation, cmap=cmap_2, vmin=0, vmax=1, aspect='equal')
    plt.axis('off')
    plt.title('DBSI Highly Restricted Fraction', fontsize=fontsize_1, fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.set_clim(0, 1)
    cbar.ax.set_yticklabels(['0', '0.5', '1'], fontsize=fontsize_2, fontweight='bold')

    # DBSI Restricted Fraction Map
    ax = fig.add_subplot(2, 4, 5)
    img = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, 'restricted_ratio_2_map.nii')).get_data()
    ax.imshow(img[:,:,k], interpolation=interpolation, cmap=cmap_2, vmin=0, vmax=1, aspect='equal')
    plt.axis('off')
    plt.title('DBSI Restricted Fraction', fontsize=fontsize_1, fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.set_clim(0, 1)
    cbar.ax.set_yticklabels(['0', '0.5', '1'], fontsize=fontsize_2, fontweight='bold')

    # DBSI Hindered Fraction Map
    ax = fig.add_subplot(2, 4, 6)
    img = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, 'hindered_ratio_map.nii')).get_data()
    ax.imshow(img[:,:,k], interpolation=interpolation, cmap=cmap_2, vmin=0, vmax=1, aspect='equal')
    plt.axis('off')
    plt.title('DBSI Hindered Fraction', fontsize=fontsize_1, fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.set_clim(0, 1)
    cbar.ax.set_yticklabels(['0', '0.5', '1'], fontsize=fontsize_2, fontweight='bold')

    # DBSI Free Fraction Map
    ax = fig.add_subplot(2, 4, 7)
    img = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, 'water_ratio_map.nii')).get_data()
    ax.imshow(img[:, :, k], interpolation=interpolation, cmap=cmap_2, vmin=0, vmax=1, aspect='equal')
    plt.axis('off')
    plt.title('DBSI Free Fraction', fontsize=fontsize_1, fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.set_clim(0, 1)
    cbar.ax.set_yticklabels(['0', '0.5', '1'], fontsize=fontsize_2, fontweight='bold')

    # DBSI Fiber Fraction Map
    ax = fig.add_subplot(2, 4, 8)
    img = nib.load(os.path.join(proj_dir, sample_dir, dbsi_dir, 'fiber_ratio_map.nii')).get_data()
    ax.imshow(img[:, :, k],interpolation=interpolation, cmap=cmap_2, vmin=0, vmax=1, aspect='equal')
    plt.axis('off')
    plt.title('DBSI Anisotropic Fraction', fontsize=fontsize_1, fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.set_clim(0, 1)
    cbar.ax.set_yticklabels(['0', '0.5', '1'], fontsize=fontsize_2, fontweight='bold')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.01)
    plt.savefig(os.path.join(proj_dir, sample_dir, 'maps.png'), bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()
    plt.close()
    print('generating DBSI maps: complete!!!')

generating_DBSI_maps()