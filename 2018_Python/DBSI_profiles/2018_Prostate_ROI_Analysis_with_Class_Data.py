import os
import tkFileDialog
import re
import nibabel as nib
import numpy as np
import pandas as pd
import glob
import ntpath
import ROI_Plot_Prostate_Log_Hist_with_class_data
from scipy.io import loadmat
import matlab.engine
import argparse
from scipy import stats
import copy
import matplotlib.pyplot as plt
import h5py



def main(data_folder='',sub_list=''):

    if data_folder == '':
        data_folder = tkFileDialog.askdirectory(title="Select Main Directory")
    if sub_list == '':
        sub_list_file = tkFileDialog.askopenfilename(title="Select Subject List to combine subjects",initialdir=data_folder)
        if sub_list_file:
            sub_df = pd.read_csv(sub_list_file)
            sub_list = sub_df[sub_df.columns[0]].tolist()

    # plot = raw_input("Would you like to generate plot(s)? :\n 1: Yes\n 2: No\n")
    plot = '1'
    if plot == '1':
        # bin = float(raw_input("Please enter number of bins for Histogram plot\n"))
          bin = '40'
    files = []
    if sub_list == '':
        for dirName, subdirList, fileList in os.walk(data_folder):
            for fname in range(len(fileList)):
                if fileList[fname]=="DBSIClassData.mat":
                    files.append(os.path.join(dirName,fileList[fname]))
    else:
        for sub in sub_list:
            files.append(os.path.join(data_folder,sub,"DBSIClassData.mat"))
    eng = matlab.engine.start_matlab()
    eng.addpath('/bmrs090temp/dhisto_release/','/bmrs090temp/dhisto_release/Misc/','/bmrs090temp/dhisto_release/Misc/NIfTI_20140122/','/bmrs090temp/Ajit_George/fsl/Prostate/ALL_DBSI_DATA_new_thresh_all/')

    min_adc = 0.15
    max_adc = 3.5 + min_adc
    adc_constant = min_adc
    thr = 0

    col_list = ["ADC_Linear_Scale"]
    col_list.insert(0,"ADC")
    col_list.insert(0,"Count")
    col_list.insert(0,"Fractions")
    col_list.insert(0,"ROI Name")
    col_list.insert(0,"ROI ID")
    col_list.insert(0,"Sub_ID")
    if sub_list != '':
        stat_df = pd.DataFrame([],columns=col_list)
    for dir in files:
        eng.create_mat(dir,nargout=0)
        type = os.path.basename(os.path.dirname(os.path.dirname(dir)))
        sub_id = os.path.basename(os.path.dirname(dir))
        if sub_list == '':
            stat_df = pd.DataFrame([],columns=col_list)
        groups = re.compile(r'(^[0-9]+)(\_)(\D*)([0-9]*)')
        string_groups = groups.search(sub_id)
        roi_path = os.path.dirname(os.path.dirname(dir)).replace('ALL_DBSI_raw_data','All_ROI_downsampled')
        # roi_path = os.path.dirname(os.path.dirname(dir)).replace('ALL_DBSI_DATA_new_thresh_all','All_ROI_downsampled')
        if type == 'Non_PCa_with_biopsy' or type == 'Non_PCa_without_biopsy':
            rois = glob.glob(os.path.join(roi_path,'p%s_*.nii'%string_groups.group(1)))
        elif type == 'PCa_with_biopsy':
            rois = glob.glob(os.path.join(roi_path,'pca_%s_*.nii'%string_groups.group(1)))

        for roi in rois:
            atlas = nib.load(roi).get_data()
            atlas = atlas.astype(float)
            for row_num in range(1,100):
                num = row_num
                if any(atlas.ravel() == num):
                    continue
                else:
                    break
            roi_num = num
            roi_folder,roi_name = ntpath.split(roi)
            for row_num in range(1,roi_num):
                roi_id = row_num
                adc_data = h5py.File(os.getcwd() + os.sep + 'ADC.mat','r')
                # adc_data = loadmat(os.getcwd() + os.sep + 'ADC.mat')
                # coordinates = adc_data['Coordinates']
                coordinates =  np.asarray(adc_data['Coordinates']).transpose()
                idx = np.asarray(np.where(atlas==roi_id))
########################################################################################################################
######################### qc check #####################################################################################
                co = coordinates - 1
                img_dti_adc = np.zeros([atlas.shape[0],atlas.shape[1],atlas.shape[2]])
                img_dti_fa = np.zeros([atlas.shape[0],atlas.shape[1],atlas.shape[2]])
                temp_atlas = copy.deepcopy(atlas)
                temp_atlas[temp_atlas[:]==0]=np.nan
                dti_adc_data = np.asarray(adc_data['DTI_ADC']).transpose()
                dti_fa_data = np.asarray(adc_data['DTI_FA']).transpose()
                for img_idx in range(co.shape[1]):
                    imdx = list(co[:,img_idx].astype(int))
                    img_dti_adc[imdx[0],imdx[1],imdx[2]] = dti_adc_data[0,img_idx]
                    img_dti_fa[imdx[0],imdx[1],imdx[2]] = dti_fa_data[0,img_idx]
                fig = plt.figure(figsize=(24,15))
                ax = fig.add_subplot(1,1,1)
                # ax_1 = fig.add_subplot(2,1,1)
                ax.axis('off')
                ax.imshow(img_dti_adc[:,:,idx[2,1]],cmap='gray',vmin=0.2,vmax=2.0,aspect='equal')
                ax.imshow(temp_atlas[:,:,idx[2,1]],cmap='rainbow',alpha=0.35,aspect='equal')
                ax.set_title('DTI ADC',fontweight='bold')
                # ax_2 = fig.add_subplot(2,2,1)
                # ax_2.axis('off')
                # ax_2.set_title('DTI FA',fontweight='bold')
                # ax_2.imshow(img_dti_fa[:,:,idx[2,1]],cmap='gray',vmin=0.3,vmax=0.8,aspect='equal')
                # ax_2.imshow(temp_atlas[:,:,idx[2,1]],cmap='rainbow',alpha=0.35,aspect='equal')
                plt.savefig(os.path.join(os.path.dirname(dir),'qc_%s.tiff'%roi_name[:-4]),format='tiff', dpi=100)
                plt.close()
########################################################################################################################
########################################################################################################################                
                # adc = adc_data['ADC']
                adc = np.asarray(adc_data['ADC']).transpose()
                fractions_data = np.asarray(adc_data['Fractions']).transpose()
                img_fr = np.zeros([atlas.shape[0],atlas.shape[1],atlas.shape[2],adc.shape[1]])
                for img_idx in range(co.shape[1]):
                    imdx = list(co[:,img_idx].astype(int))
                    img_fr[imdx[0],imdx[1],imdx[2]] = fractions_data[:,img_idx]
                fr = img_fr[idx[0],idx[1],idx[2],:]
                fr = fr.transpose()
                mean_fr = np.mean(fr,axis=1)
                adc = adc.flatten() * 1000
                if bin == 0:
                    bin = len(adc)
                adc = np.log10(adc + adc_constant)
                bins = np.linspace(np.log10(min_adc),np.log10(max_adc), bin)
                bin_sum, bin_edges, binnumber = stats.binned_statistic(adc,mean_fr, statistic='sum',bins=bins,range=(np.log10(min_adc),np.log10(max_adc)))
                bin_edges = copy.deepcopy(bin_edges[0:-1])
                adc = copy.deepcopy(bin_edges)
                adc_lin = copy.deepcopy(bin_edges)
                adc_lin = np.power(10,adc_lin) - min_adc
                mean_fr = copy.deepcopy(bin_sum)
                val_count = copy.deepcopy(mean_fr)
                val_count[val_count>thr] = 1
                val_count[val_count<=thr] = 0
                sub_data = [mean_fr,val_count,adc,adc_lin]
                val = np.asarray(sub_data)
                if sub_list == '':
                    val = np.concatenate((np.repeat(roi_name[:-4],val.shape[1])[np.newaxis], val), axis=0)
                elif type == 'Non_PCa_without_biopsy':
                    grp = re.compile(r'(^[a-zA-Z]+)([0-9]+)(\_)([a-zA-Z]+)')
                    str_grps = grp.search(roi_name[:-4])
                    new_roi_name = re.sub(str_grps.group(2),'',roi_name[:-4])
                    val = np.concatenate((np.repeat(new_roi_name,val.shape[1])[np.newaxis],val), axis=0)
                elif type == 'Non_PCa_with_biopsy':
                    grp = re.compile(r'(^[a-zA-Z]+)([0-9]+)(\_)([a-zA-Z]+)')
                    str_grps = grp.search(roi_name[:-4])
                    new_roi_name = re.sub(str_grps.group(2),'',roi_name[:-4])
                    val = np.concatenate((np.repeat(new_roi_name,val.shape[1])[np.newaxis], val), axis=0)
                elif type == 'PCa_with_biopsy':
                    val = np.concatenate((np.repeat('pca',val.shape[1])[np.newaxis], val), axis=0)
                val = np.concatenate((np.repeat(roi_id,val.shape[1])[np.newaxis], val), axis=0)
                val = np.concatenate((np.repeat(sub_id,val.shape[1])[np.newaxis], val), axis=0)
                val = np.transpose(val)
                df = pd.DataFrame(val,columns=col_list)
                stat_df = pd.concat([stat_df, df])
        if adc_data.fid.valid:
            adc_data.close()
            os.remove(os.getcwd() + os.sep + 'ADC.mat')
        if sub_list == '':
            stat_df.to_csv((os.path.join((os.path.dirname(dir)),'%s.csv'%sub_id )),index=False)
            if plot == '1':
                ROI_Plot_Prostate_Log_Hist_with_class_data.main(os.path.join((os.path.dirname(dir)),'%s.csv'%sub_id ),bin)
            else:
                continue
    if sub_list != '':
        roi_names = stat_df['ROI Name'].unique().tolist()
        sub_names = stat_df['Sub_ID'].unique().tolist()
        fraction = np.zeros([int(bin-1),len(sub_names)])
        count = np.zeros([int(bin-1),len(sub_names)])
        for roi in roi_names:
            df_roi = stat_df[stat_df['ROI Name'] == roi]
            for k,sub in enumerate(sub_names):
                df_sub = df_roi[df_roi['Sub_ID'] == sub]
                fraction[:,k] = df_sub['Fractions']
                count[:,k] = df_sub['Count']
                adc = df_sub['ADC']
                adc_lin = df_sub['ADC_Linear_Scale']
                roi_name = df_sub['ROI Name']
            count_val = np.sum(count,axis=1)
            sum_val = np.sum(fraction,axis=1)
            np.seterr(divide='ignore')
            mean_val = np.divide(sum_val,count_val)
            mean_val[np.isnan(mean_val)]=0
            all_val = np.concatenate((mean_val[:,np.newaxis],count_val[:,np.newaxis],sum_val[:,np.newaxis],adc[:,np.newaxis],adc_lin[:,np.newaxis],roi_name[:,np.newaxis]),axis=1)
            mean_stat_df = pd.DataFrame(all_val,columns=['Fractions','Count','Sum','ADC','ADC_Linear_Scale','ROI Name'])
            mean_stat_df.to_csv((os.path.join(os.path.dirname(data_folder),'%s_%s.csv'%(type,roi) )),index=False)
            if plot == '1':
                ROI_Plot_Prostate_Log_Hist_with_class_data.main(os.path.join(os.path.dirname(data_folder),'%s_%s.csv'%(type,roi)),bin)

    # if sub_list != '':
    #     filename = raw_input("Please enter filename to store ROI Data\n")
    #     stat_df.to_csv((os.path.join(data_folder,'%s.csv'%filename )),index=False)
    #     if plot == '1':
    #         ROI_Plot_Prostate_Log_Hist_with_class_data.main(os.path.join(data_folder,'%s.csv'%filename ),bin)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', required=False, default='')
    parser.add_argument('--sub_list', required=False, default='')
    args = parser.parse_args()
    main()
