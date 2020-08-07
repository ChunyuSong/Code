#----------------------------------------------------------------------
# extract voxel information
# get metric values from MRI
#
# Author: Zezhong Ye & Ajit George, Washington University School of Medicine
# Copyright: Copyright 2020, Washington University School of Medicine
# Email: ze-zhong@wustl.edu
# Date: 01.16.2020   
#     
#-------------------------------------------------------------------------------------------

import os
import nibabel as nib
import numpy as np
import pandas as pd
import glob
import ntpath
import timeit
from datetime import datetime
from time import gmtime, strftime


# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
def data_path():

    if not os.path.exists(result_dir):
        print('result directory does not exist - creating...')
        os.makedirs(result_dir)
        print('log directory created...')
    else:
        print('result directory already exists ...')

    if not os.path.exists(log_dir):
           print('log directory does not exist - creating...')
           os.makedirs(log_dir)
           os.makedirs(log_dir + '/train')

           os.makedirs(log_dir + '/validation')
           print('log directory created.')
    else:
        print('log directory already exists...')

# ----------------------------------------------------------------------------------
# class
# ----------------------------------------------------------------------------------

class voxel(object):

    def __init__(
                 self,
                 proj_dir,
                 result_dir,
                 DBSI_folder,
                 thresholds,
                 csv_name,
                 roi_1
                 ):

        self.proj_dir   = proj_dir
        self.result_dir = result_dir
        self.DBSI_foler = DBSI_folder
        self.thresholds = thresholds
        self.csv_name   = csv_name
        self.roi_1      = roi_1
        
    def map_list(self):

        param_list = [
                      'fiber1_extra_axial_map',
                      'fiber1_extra_radial_map',
                      'fiber1_extra_ratio_map',
                      'fiber1_intra_axial_map',
                      'fiber1_intra_radial_map',
                      'fiber1_intra_ratio_map',
                     ]

        return param_list

    def map_id(self):

        param_id = [
                    'fiber1_extra_axial',
                    'fiber1_extra_radial',
                    'fiber1_extra_ratio',
                    'fiber1_intra_axial',
                    'fiber1_intra_radial',
                    'fiber1_intra_ratio',
                    ]

        return param_id


    def voxel_extraction(self):

        dirs = []

        param_list = self.map_list()
        param_id   = self.map_id()
        
        for dirName, subdirList, fileList in os.walk(self.proj_dir):
            
            for dirname in range(len(subdirList)):
                
                if subdirList[dirname] == self.DBSI_foler % (
                                                             self.thresholds[0],
                                                             self.thresholds[0],
                                                             self.thresholds[1],
                                                             self.thresholds[1]
                                                             ):
                                                             dirs.append(os.path.join(dirName, subdirList[dirname]))                        
                                                        
        
        col_list = param_id  # insert voxel index and information into each column
        
        col_list.insert(0, 'ROI_Class')
        col_list.insert(0, 'ROI_ID')
        col_list.insert(0, 'Voxel')
        col_list.insert(0, 'Z')
        col_list.insert(0, 'Y')
        col_list.insert(0, 'X')
        col_list.insert(0, 'Sub_ID')
        
        stat_df = pd.DataFrame([], columns=col_list)

        for dir in dirs:
            
            sub_id = os.path.basename(os.path.dirname(dir))
            print(sub_id)
            roi_path = dir

            roi = os.path.join(roi_path, self.roi_1)
            print(roi)
                                                                 
            try:
                atlas = nib.load(roi).get_data()
                
            except:
                print('No roi')
                continue

            roi_folder, roi_name = os.path.split(roi)
            current_dir = dir
            
            if len(np.unique(atlas[atlas>0])) > 0:
                
                roi_ids = np.unique(atlas)[1:]
                print(roi_ids)               

            for roi_id in roi_ids:

                stat = []

                idx = np.asarray(np.where(atlas == roi_id))

                for item in range(len(param_list)):
                    
                    img = nib.load(glob.glob(os.path.join(current_dir, param_list[item] + '.nii'))[0]).get_data()
                    sub_data = img[atlas == roi_id]
                    stat.append(sub_data)

                val = np.asarray(stat)
                val = np.concatenate((np.repeat(roi_name[:-7], len(sub_data))[np.newaxis], val), axis=0) # -4 for .nii file, -7 for nii.gz file
                val = np.concatenate((np.repeat(roi_id, len(sub_data))[np.newaxis], val), axis=0)
                val = np.concatenate((np.asarray(range(0, len(sub_data)))[np.newaxis], val), axis=0)
                val = np.concatenate((idx, val), axis=0)
                val = np.concatenate((np.repeat(sub_id, len(sub_data))[np.newaxis], val), axis=0)
                val = np.transpose(val)

                df = pd.DataFrame(val, columns=col_list)
                stat_df = pd.concat([stat_df, df])

        csv_filename = str(self.csv_name) + '_' + \
                       strftime('%d_%b_%Y_%H_%M_%S', gmtime()) + \
                       '.csv'
                               
        stat_df.to_csv(os.path.join(self.result_dir, csv_filename), index=False)

        return stat_df

# ----------------------------------------------------------------------------------
# parameters and main fuction
# ----------------------------------------------------------------------------------
 
if __name__ == '__main__':

    proj_dir    = r'\\10.39.42.102\temp\Spees\MS_Brain'
    result_dir  = r'\\10.39.42.102\temp\Spees\MS_Brain'
    log_dir     = r'\\10.39.42.102\temp\Spees\MS_Brain'
    DBSI_folder = 'DBSI_IA_results_%s_%s_%s_%s'     
    csv_name    = 'Brain_lesion_DBSI_IA'
    roi_1       = 'mask.nii.gz'            
    thresholds  = [0.3, 2]


    # ----------------------------------------------------------------------------------
    # run main function
    # ----------------------------------------------------------------------------------

    print("extract voxel values and index from ROI and DBSI data: start...")
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    start = timeit.default_timer()

    data_path()
    
    stat_df = voxel(
                    proj_dir,
                    result_dir,
                    DBSI_folder,
                    thresholds,
                    csv_name,
                    roi_1
                    ).voxel_extraction()

    stop = timeit.default_timer()
    running_seconds = np.around(stop-start, 0)
    
    print('running time:', running_seconds, 'seconds')
    print("extract voxels from ROI and DBSI: complete!!!")





















