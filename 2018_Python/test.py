### generate DBSI maps in one panel;
### created by YZZ on 9/6/2018;

import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size


plt.subplot(241)
img = nib.load('dti_adc_map.nii')
data = img.get_data()
data = data[:,:,3]
# ax = plt.gca()
map = plt.imshow(data,cmap='gray',vmin=0, vmax=1.5)
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
plt.axis('off')
plt.title('ADC',fontsize=20)
# plt.colorbar(map, cax=cax)
# cbar = plt.colorbar(cax, ticks=[0, 1, 2])
# cbar.ax.set_yticklabels(['0', '1', '2'])

plt.subplot(242)
img = nib.load('dti_fa_map.nii')
data = img.get_data()
data = data[:,:,3]
map = plt.imshow(data,interpolation='nearest',cmap='jet',vmin=0, vmax=1)
# plt.colorbar(map)
plt.axis('off')
plt.title('FA',fontsize=20)

plt.subplot(243)
img = nib.load('restricted_ratio_1_map.nii')
data = img.get_data()
data = data[:,:,3]
map = plt.imshow(data,interpolation='nearest',cmap='jet',vmin=0, vmax=0.5)
# plt.colorbar(map)
plt.axis('off')
plt.title('highly restricted',fontsize=20)

plt.subplot(244)
img = nib.load('restricted_ratio_2_map.nii')
data = img.get_data()
data = data[:,:,3]
map = plt.imshow(data,interpolation='nearest',cmap='jet',vmin=0, vmax=1)
# plt.colorbar(map)
plt.axis('off')
plt.title('restricted',fontsize=20)

plt.subplot(245)
img = nib.load('hindered_ratio_map.nii')
data = img.get_data()
data = data[:,:,3]
map = plt.imshow(data,interpolation='nearest',cmap='jet',vmin=0, vmax=1)
# plt.colorbar(map)
plt.axis('off')
plt.title('hindered',fontsize=20)

plt.subplot(246)
img = nib.load('water_ratio_map.nii')
data = img.get_data()
data = data[:,:,3]
map = plt.imshow(data,interpolation='nearest',cmap='jet',vmin=0, vmax=1)
# plt.colorbar(map)
plt.axis('off')
plt.title('free',fontsize=20)


plt.subplot(247)
img = nib.load('fiber_ratio_map.nii')
data = img.get_data()
data = data[:,:,3]
map = plt.imshow(data,interpolation='nearest',cmap='jet',vmin=0, vmax=1)
# plt.colorbar(map)
plt.axis('off')
plt.title('anisotropic',fontsize=20)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.05, hspace=.05)
plt.show()