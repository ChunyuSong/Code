import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import filedialog
import os
import ntpath
import seaborn as sns
import argparse
import itertools
from scipy import stats
import matplotlib.patches as mpatches

def main(file_path = '',bin=128):
    if file_path == '':
        #file_path = filedialog.askopenfilename(title="Select Statistics File")
        #bin = float(raw_input("Please enter number of bins for Histogram plot\n"))
        file_path = 'Z:\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\PCa_with_biopsy\011_XU_YUN_LIN'
        bin = 100

    pathname, filename = ntpath.split(file_path)
    stat_df  = pd.read_csv(file_path)
    exclude_list = ['FA','Axial', 'Radial', 'Fiber Fraction', 'Fiber Fraction 1','Fiber FA 1','Fiber Axial 1', 'Fiber Radial 1', 'Fiber Fraction 2', 'Fiber FA 2','Fiber Axial 2', 'Fiber Radial 2']
    stat_df.drop(exclude_list, axis = 1, inplace = True)
    roi_name = stat_df['ROI Name'].unique().tolist()
    plt_counter = 0
    marker = ['D','s','o','^']
    marker = np.asarray(marker)
    max_adc = 3.5
    min_adc = 0
    for i in range(len(roi_name)):
        type = ["Restricted 1","Restricted 2","Hindered", "Water"]
        type = np.asarray(type)
        index = stat_df.index[stat_df['ROI Name'] == roi_name[i]].tolist()
        rfr_adc_1_df = stat_df.loc[index,['Restricted Fraction 1', 'Restricted ADC 1']]
        rfr_adc_2_df = stat_df.loc[index,['Restricted Fraction 2','Restricted ADC 2']]
        hfr_adc_df = stat_df.loc[index,['Hindered Fraction','Hindered ADC']]
        wfr_adc_df = stat_df.loc[index,['Water Fraction','Water ADC']]
        dti_adc_df = stat_df.loc[index,['ADC']]
        rfr_adc_1_df.dropna(inplace=True)
        rfr_adc_2_df.dropna(inplace=True)
        hfr_adc_df.dropna(inplace=True)
        wfr_adc_df.dropna(inplace=True)
        dti_adc_df.dropna(inplace=True)

        idx = [len(rfr_adc_1_df.index),len(rfr_adc_2_df.index),len(hfr_adc_df.index),len(wfr_adc_df.index)]
        non_zero_idx = [n for n in range(len(idx)) if idx[n]!=0]
        palette = itertools.compress(sns.hls_palette(len(idx), l=.3, s=.8),idx)

        if all(value == 0 for value in idx):
            print('No data to plot for ROI Name -%s' %(roi_name[i]))
            continue
        else:
            plt_counter = plt_counter + 1
            sns.set(font_scale=1.5)
            sns.set_style("white")
            type = np.repeat(type,idx,axis=0)
            d = {'Type': type,'Fraction': np.concatenate((rfr_adc_1_df['Restricted Fraction 1'],rfr_adc_2_df['Restricted Fraction 2'],hfr_adc_df['Hindered Fraction'],wfr_adc_df['Water Fraction']),axis=0), 'ADC': np.concatenate((rfr_adc_1_df['Restricted ADC 1'],rfr_adc_2_df['Restricted ADC 2'],hfr_adc_df['Hindered ADC'],wfr_adc_df['Water ADC']),axis=0)}
            df = pd.DataFrame(data=d,index=range(len(type)))
            plt.switch_backend('TkAgg')
            plt.rc('font', weight='bold')
            plt.rc('font', style='normal')
            plt.rc('xtick', labelsize=14)
            ax = sns.lmplot(data=df,x='ADC',y='Fraction',hue='Type',scatter=True,fit_reg=False,ci=None,markers=marker[non_zero_idx].tolist(),scatter_kws={"s": 100},palette=palette)
            ax.set_xlabels(fontweight='bold')
            ax.set_ylabels(fontweight='bold')
            ax.set(xlim=(-0.1,4))
            ax.fig.set_figwidth(24)
            ax.fig.set_figheight(15)
            adc = np.concatenate((rfr_adc_1_df['Restricted ADC 1'],rfr_adc_2_df['Restricted ADC 2'],hfr_adc_df['Hindered ADC'],wfr_adc_df['Water ADC']),axis=0)
            fraction =  np.concatenate((rfr_adc_1_df['Restricted Fraction 1'],rfr_adc_2_df['Restricted Fraction 2'],hfr_adc_df['Hindered Fraction'],wfr_adc_df['Water Fraction']),axis=0)
            dti_adc = np.asarray(dti_adc_df['ADC'])
            ax.savefig(os.path.join(pathname,'Plot_scatter_%s.tiff'%(roi_name[i])),format='tiff', dpi=300)
            plt.close()
            plt.switch_backend('TkAgg')
            plt.rc('font', weight='bold')
            plt.rc('font', style='normal')
            plt.rc('xtick', labelsize=14)
            fig = plt.figure(figsize=(24,15))
            ax = fig.add_subplot(2,1,1)
            ax_dti = fig.add_subplot(2,1,2)
            bin_width = (max_adc-min_adc)/bin
            bin_means, bin_edges, binnumber = stats.binned_statistic(adc,fraction, statistic='mean',bins=bin,range=(min_adc,max_adc))
            bin_count, dti_bin_edges, dti_binnumber = stats.binned_statistic(dti_adc, dti_adc, statistic='count',bins=bin,range=(min_adc,max_adc))
            new_idx = [len(bin_edges[bin_edges<rfr_adc_1_df['Restricted ADC 1'].max()]),len(bin_edges[bin_edges<rfr_adc_2_df['Restricted ADC 2'].max()]) - len(bin_edges[bin_edges<rfr_adc_1_df['Restricted ADC 1'].max()]),len(bin_edges[bin_edges<hfr_adc_df['Hindered ADC'].max()]) - len(bin_edges[bin_edges<rfr_adc_2_df['Restricted ADC 2'].max()]),len(bin_edges[bin_edges<wfr_adc_df['Water ADC'].max()]) - len(bin_edges[bin_edges<hfr_adc_df['Hindered ADC'].max()])]
            new_idx = [0 if n <=0 else n for n in new_idx]
            palette_hist = sns.hls_palette(4, l=.3, s=.8)
            color = list(palette_hist)
            RF1 = mpatches.Patch(color=color[0], label='Restricted 1')
            RF2 = mpatches.Patch(color=color[1], label='Restricted 2')
            HF = mpatches.Patch(color=color[2], label='Hindered')
            WF = mpatches.Patch(color=color[3], label='Water')
            handles=[RF1,RF2,HF,WF]
            handles = np.asarray(handles)
            color = np.asarray(color)
            color = np.repeat(color,new_idx,axis=0)
            ax.bar(bin_edges[0:len(bin_means)],bin_means,bin_width,align='edge',color=color)
            ax.set_ylim(0,1)
            ax.set_xlim(-0.1,4)
            ax.set_xlabel('ADC',fontweight='bold')
            ax.set_ylabel('Fraction Bin Means',fontweight='bold')
            ax.legend(handles=handles[non_zero_idx].tolist(), loc=1)
            DTI_ADC = mpatches.Patch(color='blue', label='DTI ADC')
            dti_handle = [DTI_ADC]
            ax_dti.bar(bin_edges[0:len(bin_means)],bin_count,bin_width,align='edge',color='blue')
            ax_dti.set_xlim(-0.1,4)
            ax_dti.set_xlabel('ADC',fontweight='bold')
            ax_dti.set_ylabel('Count',fontweight='bold')
            ax_dti.legend(handles=dti_handle, loc=1)
            plt.savefig(os.path.join(pathname,'Plot_linear_hist_%s.tiff'%(roi_name[i])),format='tiff', dpi=300)
            plt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path",dest="file_path",help='Working Folder',required=False,default='')
    parser.add_argument("--bin",required=False,default=128)
    args = parser.parse_args()
    main(args.file_path,args.bin)
