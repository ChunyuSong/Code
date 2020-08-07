import os
import tkFileDialog
import re
import Tkinter
import nibabel as nib
import numpy as np
import pandas as pd
import glob
import ntpath
import ROI_Plot_Prostate_Log_Hist
import ROI_Plot_Prostate_Linear_Hist

def main():

    data_folder = tkFileDialog.askdirectory(title="Select Main Directory")
    # threshold = raw_input("Please enter DBSI Isotropic Thresholds (separated by commas)\n")
    # thresholds = [x.strip() for x in threshold.split(',')]
    thresholds = [0.1,0.8,2.3]
    plot = raw_input("Would you like to generate plot(s)? :\n 1: Yes\n 2: No\n")
    if plot == '1':
        plot_type = raw_input("Type of Histogram plot :\n 1: Log\n 2: Linear\n")
        bin = float(raw_input("Please enter number of bins for Histogram plot\n"))
    dirs = []
    for dirName, subdirList, fileList in os.walk(data_folder):
        for dirname in range(len(subdirList)):
            if subdirList[dirname] == 'DBSI_results_%s_%s_%s_%s_%s_%s'%(thresholds[0],thresholds[0],thresholds[1],thresholds[1],thresholds[2],thresholds[2]):
                dirs.append(os.path.join(dirName,subdirList[dirname]))
    dti_dbsi_param_list = ['dti_fa_map','dti_adc_map','dti_axial_map','dti_radial_map','fiber_ratio_map','fiber1_fiber_ratio_map','fiber1_fa_map','fiber1_axial_map','fiber1_radial_map','fiber2_fiber_ratio_map','fiber2_fa_map','fiber2_axial_map','fiber2_radial_map','restricted_ratio_1_map','restricted_ratio_2_map','hindered_ratio_map','water_ratio_map','restricted_adc_1_map','restricted_adc_2_map','hindered_adc_map','water_adc_map']
    param_id = ['FA','ADC','Axial', 'Radial', 'Fiber Fraction', 'Fiber Fraction 1','Fiber FA 1','Fiber Axial 1', 'Fiber Radial 1', 'Fiber Fraction 2', 'Fiber FA 2','Fiber Axial 2', 'Fiber Radial 2','Restricted Fraction 1','Restricted Fraction 2','Hindered Fraction','Water Fraction','Restricted ADC 1','Restricted ADC 2','Hindered ADC','Water ADC']
    col_list = param_id
    col_list.insert(0,"ROI Name")
    col_list.insert(0,"ROI ID")
    col_list.insert(0,"Voxel")
    col_list.insert(0,"Z")
    col_list.insert(0,"Y")
    col_list.insert(0,"X")
    col_list.insert(0,"Sub_ID")
    for dir in dirs:
        type = os.path.basename(os.path.dirname(os.path.dirname(dir)))
        sub_id = os.path.basename(os.path.dirname(dir))
        stat_df = pd.DataFrame([],columns=col_list)
        if type == 'Non_PCa_with_biopsy' or type == 'Non_PCa_without_biopsy':
            groups = re.compile(r'(^[0-9]+)(\_)(\D*)([0-9]*)')
            string_groups = groups.search(sub_id)
            roi_path = os.path.dirname(os.path.dirname(dir)).replace('DBSI','ROI')
            rois = glob.glob(os.path.join(roi_path,'p%s_*.nii'%string_groups.group(1)))
        elif type == 'PCa_with_biopsy':
            groups = re.compile(r'(^[0-9]+)(\_)(\D*)([0-9]*)')
            string_groups = groups.search(sub_id)
            roi_path = os.path.dirname(os.path.dirname(dir)).replace('DBSI','ROI')
            rois = glob.glob(os.path.join(roi_path,'pca_%s_*.nii'%string_groups.group(1)))
        for roi in rois:
            stat = []
            atlas = nib.load(roi).get_data()
            for row_num in range(1,100):
                num = row_num
                if any(atlas.ravel() == num):
                    continue
                else:
                    break
            roi_num = num
            roi_folder,roi_name = ntpath.split(roi)
            for row_num in range(1,roi_num):
                current_dir = dir
                roi_id = row_num
                for file in range(len(dti_dbsi_param_list)):
                    img = nib.load(glob.glob(os.path.join(current_dir,dti_dbsi_param_list[file]+'.nii'))[0]).get_data()
                    idx = np.asarray(np.where(atlas==roi_id))
                    sub_data = img[atlas==roi_id]
                    stat.append(sub_data)
                val = np.asarray(stat)
                val = np.concatenate((np.repeat(roi_name[:-4],len(sub_data))[np.newaxis], val), axis=0)
                val = np.concatenate((np.repeat(roi_id,len(sub_data))[np.newaxis], val), axis=0)
                val = np.concatenate((np.asarray(range(1,len(sub_data)+1))[np.newaxis], val), axis=0)
                val = np.concatenate((idx,val),axis=0)
                val = np.concatenate((np.repeat(sub_id,len(sub_data))[np.newaxis], val), axis=0)
                val = np.transpose(val)
                df = pd.DataFrame(val,columns=col_list)
                stat_df = pd.concat([stat_df, df])
        stat_df.to_csv((os.path.join((os.path.dirname(dir)),'%s.csv'%sub_id )),index=False)
        if plot == '1':
            if plot_type == '1':
                ROI_Plot_Prostate_Log_Hist.main(os.path.join((os.path.dirname(dir)),'%s.csv'%sub_id ),bin)
            elif plot_type =='2':
                ROI_Plot_Prostate_Linear_Hist.main(os.path.join((os.path.dirname(dir)),'%s.csv'%sub_id ),bin)

if __name__ == '__main__':
    main()
