import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
#from Tkinter import Tk
#import Tkinter, tkFileDialog
#import pickle

print ('generate ROI reports: start')

#proj_dir = r"\\10.39.42.101\temp\Victor Song\Spinal cord study_Jenny"
proj_dir = r"C:\Users\zye01\Desktop\GBM"

#Tk().withdraw() # hide root
#proj_dir = tkFileDialog.askdirectory(title='locate the project directory')

#roi_name = 'TMEV???_ROI.nii' 
#roi_name = 'TMEV???_hp_ROI.nii' 

roi_name = input("Please enter the name of roi: ")
print ("you roi file name is (.nii.gz)", roi_name)

csvfilename = input("Please enter the output csv file prefix: ")
print ("the prefix of your output csv file name will be: ", csvfilename)

print ("checking project folder %s...... " % proj_dir)
if not os.path.exists(proj_dir):
    print ("project folder does not existed ", proj_dir)
    sys.exit("%s folder does not existed!" % (proj_dir))



maps_list = pd.Series(['dti_fa_map.nii','dti_adc_map.nii','dti_axial_map.nii','dti_radial_map.nii',
                       'fiber_ratio_map.nii','fiber1_fiber_ratio_map.nii','fiber1_fa_map.nii','fiber1_axial_map.nii','fiber1_radial_map.nii',
                       'fiber2_fiber_ratio_map.nii','fiber2_fa_map.nii','fiber2_axial_map.nii','fiber2_radial_map.nii',
                       'restricted_ratio_map.nii','hindered_ratio_map.nii','water_ratio_map.nii',], 
                      index=['DTI FA','DTI ADC','DTI Axial','DTI Radial',
                             'Fiber Fraction','Fiber1 Fraction','Fiber FA','Fiber Axial','Fiber Radial',
                             'Fiber2 Fraction','Fiber2 FA','Fiber2 Axial','Fiber2 Radial',
                             'Restricted Fraction','Hindered Fraction','Water Fraction'])


roi_files = []
for roi_file in glob.glob(os.path.join(proj_dir,'*','*','*',roi_name)):
    if not os.path.exists(roi_file.replace('.nii','.csv')):  #"%s.csv" % (os.path.splitext(roi_file)[0])
        roi_files.append(roi_file)
        print ("ROI file %s found"%(roi_file))
        
for roi_id in range(1,10):
    df = pd.DataFrame([],columns=['CASE ID','ROI ID','VOXELS']+ list(maps_list.index)+['Fiber Volume','Restricted Volume','Hindered Volume','Water Volume'])  
    mastercsvfile = os.path.join(proj_dir,"%s_roi_%s_report_.csv"%(csvfilename,roi_id))
    for roi_file in roi_files:
        data_set = os.path.dirname(roi_file)
        roi_file_name = os.path.basename(roi_file)
        case_id = roi_file.split('\\')[-4]
    
    
        if all([glob.glob(os.path.join(data_set, map_file)) for map_file in maps_list]):
            print (data_set+': data is complete')
        else:
            print (data_set+': data is not complete')
            continue
    
        #mask = np.fliplr(nib.load(roi_file).get_data())
        mask = nib.load(roi_file).get_data()
    
        inx = np.where(mask==roi_id) 
        if len(inx[0])>0:
            roi_results = pd.Series()
            roi_results['CASE ID'] = case_id
            roi_results['ROI ID'] = roi_id
            roi_results['VOXELS'] = len(inx[0])
            for dbsi_index, dbsi_file in maps_list.iteritems():
                dbsimap = nib.load(os.path.join(data_set,dbsi_file)).get_data()
                roi_results[dbsi_index] = np.mean(dbsimap[inx])
            
            roi_results['Fiber Volume'] = roi_results['Fiber Fraction']*roi_results['VOXELS']        
            roi_results['Restricted Volume'] = roi_results['Restricted Fraction']*roi_results['VOXELS'] 
            roi_results['Hindered Volume'] = roi_results['Hindered Fraction']*roi_results['VOXELS']
            roi_results['Water Volume'] = roi_results['Water Fraction']*roi_results['VOXELS']
    
            df = df.append(roi_results, ignore_index=True)
    if not df.empty:       
        df.to_csv(mastercsvfile,header=True, index=False)  
        
print ('generate ROI reports: end')