import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from Tkinter import Tk
from tkFileDialog import askopenfilename
import pickle
import sys


print 'generate ROI reports: start'

proj_dir = r"\\10.39.42.100\temp\Spees\frog_nerve" 

roi_name = raw_input("Please enter the name of roi: ") # "roi_map_????_??.nii.gz"
print "you ROI naming convension is: ", roi_name        

print "checking project folder %s...... " % proj_dir
if not os.path.exists(proj_dir):
    print "project folder does not existed ", proj_dir
    sys.exit("%s folder does not existed!" % (proj_dir))


#maps_list = pd.Series(['dti_fa_map.nii','dti_adc_map.nii','dti_axial_map.nii','dti_radial_map.nii',
                       #'fiber_ratio_map.nii','fiber1_fa_map.nii','fiber1_axial_map.nii','fiber1_radial_map.nii',
                       #'restricted_ratio_map.nii','hindered_ratio_map.nii','water_ratio_map.nii'], 
                      #index=['DTI FA','DTI ADC','DTI Axial','DTI Radial',
                             #'Fiber Fraction','Fiber FA','Fiber Axial','Fiber Radial',
                             #'Restricted Fraction','Hindered Fraction','Water Fraction'])

maps_list = pd.Series(['dti_fa_map.nii','dti_adc_map.nii','dti_axial_map.nii','dti_radial_map.nii',
                       'fiber_ratio_map.nii','fiber1_fa_map.nii','fiber1_axial_map.nii','fiber1_radial_map.nii',
                       'restricted_ratio_map.nii','hindered_ratio_map.nii','water_ratio_map.nii', 'restricted_adc_map.nii','hindered_adc_map.nii','water_adc_map.nii'], 
                      index=['DTI FA','DTI ADC','DTI Axial','DTI Radial',
                             'Fiber Fraction','Fiber FA','Fiber Axial','Fiber Radial',
                             'Restricted Fraction','Hindered Fraction','Water Fraction','Restricted ADC','Hindered ADC','Water ADC'])


roi_files = []
for roi_file in glob.glob(os.path.join(proj_dir,'s_*','dbsi*','DBSI_results_*',roi_name)):
    roi_files.append(roi_file)
    print "ROI file %s found"%(roi_file) 
        
df = pd.DataFrame() 
mastercsvfile = os.path.join(proj_dir,"roi_map_ham_101718.csv")
        
for roi_file in roi_files:        

    data_set = os.path.dirname(roi_file)
    roi_file_name = os.path.basename(roi_file)
    case_id = roi_file.split('\\')[-4]
    acq_id = roi_file.split('\\')[-3]

    if all([glob.glob(os.path.join(data_set, map_file)) for map_file in maps_list]):
        print data_set+': data is complete'
    else:
        print data_set+': data is not complete'  
        continue

    #mask = np.fliplr(nib.load(roi_file).get_data())
    mask = nib.load(roi_file).get_data()
    
    for roi_id in range(1,20):   
        inx = np.where(mask==roi_id) 
        if len(inx[0])>0:
            roi_results = pd.Series()
            roi_results['CASE ID'] = case_id
            roi_results['ACQ ID'] = acq_id
            roi_results['ROI ID'] = roi_id
            roi_results['VOXELS'] = len(inx[0])
            for dbsi_index, dbsi_file in maps_list.iteritems():
                dbsimap = nib.load(os.path.join(data_set,dbsi_file)).get_data()
                roi_results[dbsi_index] = np.nanmean(dbsimap[inx])
            
            roi_results['Fiber Volume'] = roi_results['Fiber Fraction']*roi_results['VOXELS']        
            roi_results['Restricted Volume'] = roi_results['Restricted Fraction']*roi_results['VOXELS']
            roi_results['Hindered Volume'] = roi_results['Hindered Fraction']*roi_results['VOXELS']
            roi_results['Water Volume'] = roi_results['Water Fraction']*roi_results['VOXELS']
    
            if df.empty:
                df = pd.DataFrame(roi_results).transpose()
            else:
                df = df.append(roi_results, ignore_index=True)
                
if not df.empty:       
    df.to_csv(mastercsvfile,header=True, index=False)  
        
print 'generate ROI reports: end'
