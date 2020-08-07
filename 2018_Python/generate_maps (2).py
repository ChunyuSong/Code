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

    print('generate DBSI maps')

    legacy_dir = r"\\10.39.42.102\temp\Zezhong_Ye\2018_Legacy_Project"
    if not os.path.exists(legacy_dir):
        print ("root folder does not existed ", legacy_dir)
        sys.exit("%s folder does not existed!" % (legacy_dir))

    dbsi_dir = 'DBSI_results_0.2_0.2_0.8_0.8_2_2'
    sample_dir = 'L003_N1'
    k = 3 #choose slice number to show

    fig = plt.figure(figsize=(15,8))
    ax = fig.add_subplot(2,4,1)
    img = nib.load(os.path.join(legacy_dir, sample_dir, dbsi_dir, 'dti_adc_map.nii')).get_data()
    plt.imshow(img[:,:,k], cmap='gray', vmin=0, vmax=2, aspect='equal')
    ax.set_title('ADC', fontsize=15, fontweight='bold')
    ax.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['0', '1.0', '2.0'], fontsize=12, fontweight='bold')

    ax = fig.add_subplot(2,4,2)
    img = nib.load(os.path.join(legacy_dir, sample_dir, dbsi_dir, 'dti_fa_map.nii')).get_data()
    plt.imshow(img[:,:,k], interpolation='gaussian', cmap='jet', vmin=0, vmax=1, aspect='equal')
    plt.axis('off')
    plt.title('FA', fontsize=15, fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=12, fontweight='bold')

    ax = fig.add_subplot(2,4,3)
    img = nib.load(os.path.join(legacy_dir,sample_dir,dbsi_dir,'restricted_ratio_1_map.nii')).get_data()
    ax.imshow(img[:,:,k], interpolation='gaussian', cmap='jet', vmin=0, vmax=1, aspect='equal')
    plt.axis('off')
    plt.title('highly restricted',fontsize=15, fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=12, fontweight='bold')

    ax = fig.add_subplot(2,4,4)
    img = nib.load(os.path.join(legacy_dir,sample_dir,dbsi_dir,'restricted_ratio_2_map.nii')).get_data()
    ax.imshow(img[:,:,k],interpolation='gaussian',cmap='jet',vmin=0, vmax=1,aspect='equal')
    plt.axis('off')
    plt.title('restricted',fontsize=15,fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=12, fontweight='bold')

    ax = fig.add_subplot(2,4,5)
    img = nib.load(os.path.join(legacy_dir,sample_dir,dbsi_dir,'hindered_ratio_map.nii')).get_data()
    ax.imshow(img[:,:,k],interpolation='gaussian',cmap='jet',vmin=0, vmax=0.5,aspect='equal')
    plt.axis('off')
    plt.title('hindered',fontsize=15,fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=12, fontweight='bold')

    ax = fig.add_subplot(2,4,6)
    img = nib.load(os.path.join(legacy_dir,sample_dir,dbsi_dir,'water_ratio_map.nii')).get_data()
    ax.imshow(img[:,:,k],interpolation='gaussian',cmap='jet',vmin=0, vmax=1,aspect='equal')
    plt.axis('off')
    plt.title('free',fontsize=15,fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=12, fontweight='bold')

    ax = fig.add_subplot(2,4,7)
    img = nib.load(os.path.join(legacy_dir,sample_dir,dbsi_dir,'fiber_ratio_map.nii')).get_data()
    ax.imshow(img[:,:,k],interpolation='gaussian',cmap='jet',vmin=0, vmax=1,aspect='equal')
    plt.axis('off')
    plt.title('anisotropic',fontsize=15,fontweight='bold')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, ticks=[0, 0.5, 1])
    cbar.ax.set_yticklabels(['0', '0.5', '1.0'], fontsize=12, fontweight='bold')

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.01)
    plt.savefig(os.path.join(legacy_dir,sample_dir,'maps.png'),bbox_inches='tight',facecolor=fig.get_facecolor())
    plt.show()
    plt.close()
    print('generating DBSI maps complete')

generating_DBSI_maps()