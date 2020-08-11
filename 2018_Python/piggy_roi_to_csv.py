#!/usr/bin/env python

"""this function is a library to extra roi numbers to csv."""

import os
import glob, copy
import nibabel as nib
import numpy as np
import pandas as pd
from Tkinter import Tk
import Tkinter, tkFileDialog
from tkFileDialog import askopenfilename
import pickle
import fnmatch
import datetime
import sys
import matplotlib.pyplot as plt

def locate(pattern, root=os.curdir):
    '''Locate all files matching supplied filename pattern in and below
    supplied root directory.'''
    for path, dirs, files in os.walk(os.path.abspath(root)):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)

def extract_voxels(proj_dir, roi_name, csvfilename):
    
    print 'generate Voxel reports: start'
   
    print "checking project folder %s...... " % proj_dir
    if not os.path.exists(proj_dir):
        print "project folder does not existed ", proj_dir
        sys.exit("%s folder does not existed!" % (proj_dir))
    
    maps_list = pd.Series(['dti_fa_map.nii','dti_adc_map.nii','dti_axial_map.nii','dti_radial_map.nii',
                           'fiber_ratio_map.nii','fiber1_fa_map.nii','fiber1_axial_map.nii','fiber1_radial_map.nii',
                           'restricted_ratio_map.nii','hindered_ratio_map.nii','water_ratio_map.nii'], 
                          index=['DTI FA','DTI ADC','DTI Axial','DTI Radial',
                                 'Fiber Fraction','Fiber FA','Fiber Axial','Fiber Radial',
                                 'Restricted Fraction','Hindered Fraction','Water Fraction'])
    
    
    roi_files = []
    #for roi_file in locate(roi_name, proj_dir):
        #roi_files.append(roi_file)
        #print "ROI found: %s" %(roi_file)
    for roi_file in glob.glob(os.path.join(proj_dir,'*','*','*',roi_name)):
        if not os.path.exists(roi_file.replace('.nii','.csv')):  #"%s.csv" % (os.path.splitext(roi_file)[0])
            roi_files.append(roi_file)
            print "ROI file %s found"%(roi_file)  
            
    for roi_id in range(1,50):
        df = pd.DataFrame([],columns=['GROUP ID','CASE ID','ROI ID','VOXELS']+ list(maps_list.index)+['Fiber Volume','Restricted Volume','Hindered Volume','Water Volume'])  
        mastercsvfile = os.path.join(proj_dir,"%s_voxel_%s_report.csv"%(csvfilename,roi_id))
        for roi_file in roi_files:
            data_set = os.path.dirname(roi_file)
            group_id = roi_file.split('\\')[-5]
            case_id = roi_file.split('\\')[-4]
        
        
            if all([glob.glob(os.path.join(data_set, map_file)) for map_file in maps_list]):
                print data_set+': data is complete'
            else:
                print data_set+': data is not complete'  
                continue
        
            #mask = np.fliplr(nib.load(roi_file).get_data())
            mask = nib.load(roi_file).get_data()
        
            inx = np.where(mask==roi_id) 
            if len(inx[0])>0:
                roi_results = pd.Series()
                roi_results['GROUP ID'] = group_id
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
            
    print 'generate Voxel reports: end'

def extract_rois(proj_dir, roi_name):
    
    print 'generate ROI reports: start'
   
    print "checking project folder %s...... " % proj_dir
    if not os.path.exists(proj_dir):
        print "project folder does not existed ", proj_dir
        sys.exit("%s folder does not existed!" % (proj_dir))
    
    maps_list = pd.Series(['dti_fa_map.nii','dti_adc_map.nii','dti_axial_map.nii','dti_radial_map.nii',
                           'fiber_ratio_map.nii','fiber1_fa_map.nii','fiber1_axial_map.nii','fiber1_radial_map.nii',
                           'restricted_ratio_map.nii','hindered_ratio_map.nii','water_ratio_map.nii'],
                          index=['DTI FA','DTI ADC','DTI Axial','DTI Radial',
                                 'Fiber Fraction','Fiber FA','Fiber Axial','Fiber Radial',
                                 'Restricted Fraction','Hindered Fraction','Water Fraction'])
    
    
    roi_files = []
    #for roi_file in locate(roi_name, proj_dir):
        #roi_files.append(roi_file)
        #print "ROI found: %s" %(roi_file)
    for roi_file in glob.glob(os.path.join(proj_dir,'*','*','*','*',roi_name)):
        roi_files.append(roi_file)
        print "ROI file %s found"%(roi_file)  
    
    if True:
        QC(proj_dir, roi_files)
     
    mastercsvfile = os.path.join(proj_dir,"roi_report_%s.csv"%(datetime.datetime.now().strftime("%Y_%B_%d_%I%M%p")))     
    df = pd.DataFrame([],columns=['CASE ID','SCAN DATE','ROI ID','VOXELS']+ list(maps_list.index)+['Non-Restricted Fraction','Fiber Volume','Restricted Volume','Hindered Volume','Water Volume']) #+['Fiber Volume','Restricted Volume','Hindered Volume','Water Volume'])  
    for roi_file in roi_files:  
        data_set = os.path.dirname(roi_file)
        scan_date = roi_file.split(os.sep)[-4]
        case_id = roi_file.split(os.sep)[-5]
    
        if all([glob.glob(os.path.join(data_set, map_file)) for map_file in maps_list]):
            print data_set+': data is complete'
        else:
            print data_set+': data is not complete'  
            continue
    
        #mask = np.fliplr(nib.load(roi_file).get_data())
        mask = nib.load(roi_file).get_data()        

        for roi_id in range(1,50):
            inx = np.where(mask==roi_id) 
            if len(inx[0])>0:
                roi_results = pd.Series()
                roi_results['CASE ID'] = case_id
                roi_results['SCAN DATE'] = scan_date
                roi_results['ROI ID'] = roi_id
                roi_results['VOXELS'] = len(inx[0])
                for dbsi_index, dbsi_file in maps_list.iteritems():
                    dbsimap = nib.load(os.path.join(data_set,dbsi_file)).get_data()
                    roi_results[dbsi_index] = np.mean(dbsimap[inx])
                
                roi_results['Fiber Volume'] = roi_results['Fiber Fraction']*roi_results['VOXELS']        
                roi_results['Restricted Volume'] = roi_results['Restricted Fraction']*roi_results['VOXELS']
                roi_results['Hindered Volume'] = roi_results['Hindered Fraction']*roi_results['VOXELS']
                roi_results['Water Volume'] = roi_results['Water Fraction']*roi_results['VOXELS']
                roi_results['Non-Restricted Fraction'] = roi_results['Hindered Fraction']+roi_results['Water Fraction']
                df = df.append(roi_results, ignore_index=True)
                
    if not df.empty:       
        df.to_csv(mastercsvfile,header=True, index=False)  
            
    print 'generate ROI reports: end'

def QC(proj_dir, roi_files):
    QC_dir = os.path.join(proj_dir,'QC')
    if not os.path.exists(QC_dir):
        os.makedirs(QC_dir)
        
    for i,roi_file in enumerate(roi_files):
        rois = nib.load(roi_file).get_data()
        #rois = np.fliplr(nib.load(roi_file).get_data())
        roi_path,roi_name = os.path.split(roi_file)
        
        current_dir = roi_path
        for roi_id in range(1,100):
            if not any(rois.ravel() == roi_id):
                continue

            idx = np.asarray(np.where(rois==roi_id))      
            
            temp_atlas = copy.deepcopy(rois)
            temp_atlas = temp_atlas.astype(float)
            temp_atlas[temp_atlas[:]==0]=np.nan
            
            fig = plt.figure(figsize=(24,15))
            ax_1 = fig.add_subplot(2,1,1)
            ax_1.axis('off')
            img_dti_adc = nib.load(glob.glob(os.path.join(current_dir,'dti_adc_map.nii'))[0]).get_data()
            ax_1.imshow(img_dti_adc[:,:,idx[2,0]],cmap='gray',vmin=0.2,vmax=2.0,aspect='equal')
            ax_1.imshow(temp_atlas[:,:,idx[2,0]],cmap='rainbow',alpha=0.8,aspect='equal')
            ax_1.set_title('DTI ADC',fontweight='bold')
            ax_2 = fig.add_subplot(2,2,1)
            ax_2.axis('off')
            ax_2.set_title('DTI FA',fontweight='bold')
            img_dti_fa = nib.load(glob.glob(os.path.join(current_dir,'dti_fa_map.nii'))[0]).get_data()
            ax_2.imshow(img_dti_fa[:,:,idx[2,0]],cmap='gray',vmin=0.2,vmax=0.8,aspect='equal')
            ax_2.imshow(temp_atlas[:,:,idx[2,0]],cmap='rainbow',alpha=0.8,aspect='equal')
            plt.savefig(os.path.join(QC_dir,'qc_%s_%s.png'%(roi_name[:-7],roi_id)),format='png')
            plt.close()    

if __name__ == "__main__":
    proj_dir = r"\\10.39.42.101\temp\Tsen\Kidney_Hua"
    #roi_name = 'HD????.nii' 
    #Tk().withdraw() # hide root
    #proj_dir = tkFileDialog.askdirectory(title='locate the project directory')
    
    roi_name = raw_input("Please enter the name of roi: ")
    print "you roi file name is (.nii.gz)", roi_name           
    
    extract_rois(proj_dir, roi_name )
    
    