#----------------------------------------------------------------------
# extract voxel information
# get metric values from MRI
#     
#-------------------------------------------------------------------------------------------

import os
import nibabel as nib
import numpy as np
import pandas as pd
import scipy.stats as stats
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
                 roi_1,
                 roi_2
                 ):

        self.proj_dir   = proj_dir
        self.result_dir = result_dir
        self.DBSI_foler = DBSI_folder
        self.thresholds = thresholds
        self.csv_name   = csv_name
        self.roi_1      = roi_1
        self.roi_2      = roi_2

        

    def map_list(self):

        param_list = [
                      'b0_map',
                      'dti_adc_map',
                      'dti_axial_map',
                      'dti_fa_map',
                      'dti_radial_map',
                      'fiber_ratio_map',
                      'fiber1_axial_map',
                      'fiber1_fa_map',
                      'fiber1_fiber_ratio_map',
                      'fiber1_radial_map',
                      'fiber2_axial_map',
                      'fiber2_fa_map',
                      'fiber2_fiber_ratio_map',
                      'fiber2_radial_map',
                      'hindered_adc_map',
                      'hindered_ratio_map',
                      'iso_adc_map',
                      'restricted_adc_1_map',
                      'restricted_adc_2_map',
                      'restricted_ratio_1_map',
                      'restricted_ratio_2_map',
                      'water_adc_map',
                      'water_ratio_map'
                     ]

##        param_list = [
##                      'b0_map',
##                      'dti_adc_map',
##                      'dti_axial_map',
##                      'dti_fa_map',
##                      'dti_radial_map',
##                      'fiber_ratio_map',
##                      'fiber1_axial_map',
##                      'fiber1_fa_map',
##                      'fiber1_radial_map',
##                      'hindered_ratio_map',
##                      'iso_adc_map',
##                      'restricted_ratio_1_map',
##                      'restricted_ratio_2_map',
##                      'water_ratio_map'
##                     ]



        return param_list

    def map_id(self):

        param_id = [
                    'b0',
                    'dti_adc',
                    'dti_axial',
                    'dti_fa',
                    'dti_radial',
                    'fiber_ratio',
                    'fiber1_axial',
                    'fiber1_fa',
                    'fiber1_fiber_ratio',
                    'fiber1_radial',
                    'fiber2_axial',
                    'fiber2_fa',
                    'fiber2_fiber_ratio',
                    'fiber2_radial',
                    'hindered_adc',
                    'hindered_ratio',
                    'iso_adc',
                    'restricted_adc_1',
                    'restricted_adc_2',
                    'restricted_ratio_1',
                    'restricted_ratio_2',
                    'water_adc',
                    'water_ratio'
                   ]

##        param_id = [
##                    'b0',
##                    'dti_adc',
##                    'dti_axial',
##                    'dti_fa',
##                    'dti_radial',
##                    'fiber_ratio',
##                    'fiber1_axial',
##                    'fiber1_fa',
##                    'fiber1_radial',
##                    'hindered_ratio',
##                    'iso_adc',
##                    'restricted_ratio_1',
##                    'restricted_ratio_2',
##                    'water_ratio'
##                   ]


        return param_id


    def mean_voxel_extraction(self):

        dirs = []

        param_list = self.map_list()
        param_id   = self.map_id()
        
        for dirName, subdirList, fileList in os.walk(self.proj_dir):
            
            for dirname in range(len(subdirList)):
                
                if subdirList[dirname] == self.DBSI_foler % ( #look for correct thresholds of files; append to dirs
                                                             self.thresholds[0],
                                                             self.thresholds[0],
                                                             self.thresholds[1],
                                                             self.thresholds[1],
                                                             self.thresholds[2],
                                                             self.thresholds[2]
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

        for dir in dirs: #contains all folders with correct thresholds
            
            sub_id = os.path.basename(os.path.dirname(dir))
            print(sub_id)
            roi_path = dir

            rois = [os.path.join(roi_path, self.roi_1)] + \
                   [os.path.join(roi_path, self.roi_2)]
                                     
            for roi in rois: #looking at each roi individually
                
                stat = []
                
                try:
                    atlas = nib.load(roi).get_data() #data from roi
                    
                except:
                    print('No roi')
                    continue

                roi_folder, roi_name = os.path.split(roi) #split into folder and file name
                current_dir = dir
                
                if len(np.unique(atlas[atlas>0])) > 0: #find all rois per file, look at the first one
                    # print("unique atlases: ", np.unique(atlas[atlas>0]))
                    roi_id = np.unique(atlas[atlas>0])[0]
                    
                idx = np.asarray(np.where(atlas == roi_id))
                # print(param_list)
                for item in range(len(param_list)):
                    
                    img = nib.load(glob.glob(os.path.join(current_dir, param_list[item] + '.nii'))[0]).get_data()
                    sub_data = img[atlas == roi_id]
                    stat.append(sub_data)
                    # print("subdata: ", sub_data)
                val  = np.asarray(stat)
                avg  = np.mean(val, axis = 1) #find mean of all the voxels for all metrics
                med  = np.median(val, axis = 1) #find median of all the voxels for all metrics
                skew = np.array(stats.skew(val, axis = 1)) #find skew of all the voxels for all metrics
                IQR  = np.array(stats.iqr(val, axis = 1)) #find skew of all the voxels for all metrics

                order = ["average", "median", "skew", "IQR"]
                ROI_met = np.array([avg, med, skew, IQR])
                new_met = [[],[],[],[]]

                for i in range(len(ROI_met)): #add metrics
                    new_met[i] = np.concatenate((np.array([roi_name[:-7]]), ROI_met[i]))
                    new_met[i] = np.concatenate((np.array([roi_id]), new_met[i]))
                    new_met[i] = np.concatenate((np.array([order[i]]), new_met[i]))
                    new_met[i] = np.concatenate((np.array([0,0,0]), new_met[i])) #X,Y,Z irrelevant with summary metrics
                    new_met[i] = np.concatenate((np.array([sub_id]), new_met[i]))
                # new_met = np.transpose(new_met)
                # -4 for .nii file, -7 for nii.gz file
                new_met = np.array(new_met)
                df = pd.DataFrame(new_met, columns=col_list)
                stat_df = pd.concat([stat_df, df])

        csv_filename = str(self.csv_name) + '_' + \
                       strftime('%d_%b_%Y_%H_%M_%S', gmtime()) + \
                       '.csv'
                               
        stat_df.to_csv(os.path.join(self.result_dir, csv_filename), index=False)

        return stat_df


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
                                                             self.thresholds[1],
                                                             self.thresholds[2],
                                                             self.thresholds[2]
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

            rois = [os.path.join(roi_path, self.roi_1)] + \
                   [os.path.join(roi_path, self.roi_2)] 

                                     
            for roi in rois:
                
                stat = []
                
                try:
                    atlas = nib.load(roi).get_data()
                    
                except:
                    print('No roi')
                    continue

                roi_folder, roi_name = os.path.split(roi)
                current_dir = dir
                
                if len(np.unique(atlas[atlas>0])) > 0:
                    roi_id = np.unique(atlas[atlas>0])[0]
                    
                idx = np.asarray(np.where(atlas == roi_id))

                for item in range(len(param_list)):
                    
                    img = nib.load(glob.glob(os.path.join(current_dir, param_list[item] + '.nii'))[0]).get_data()
                    sub_data = img[atlas == roi_id]
                    stat.append(sub_data)

                val = np.asarray(stat)
                # -4 for .nii file, -7 for nii.gz file
                val = np.concatenate((np.repeat(roi_name[:-7], len(sub_data))[np.newaxis], val), axis=0)
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

    proj_dir    = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\Non_PCa_without_biopsy'
    result_dir  = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\Non_PCa_without_biopsy'
    log_dir     = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\Non_PCa_without_biopsy\log'
    DBSI_folder = 'DBSI_results_%s_%s_%s_%s_%s_%s'      # 'DHISTO_results_%s_%s_%s_%s_%s_%s'
    csv_name    = 'benign_mpMRI_voxel'
    roi_1       = 'c.nii.gz'            
    roi_2       = 'p.nii.gz'
    thresholds  = [0.1, 0.8, 2.3]


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
                    roi_1,
                    roi_2
                    ).mean_voxel_extraction()

    stop = timeit.default_timer()
    running_seconds = np.around(stop-start, 0)
    
    print('DNN running time:', running_seconds, 'seconds')
    print("extract voxels from ROI and DBSI: complete!!!")





















