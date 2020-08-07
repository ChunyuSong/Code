import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tkFileDialog
import os
import ntpath
import seaborn as sns
import argparse
import itertools
from scipy import stats
import matplotlib.patches as mpatches

def main(file_path = '', bin=128):
    if file_path == '':
        file_path = tkFileDialog.askopenfilename(title="Select Statistics File")
        #file_path = r"\\Z:10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\ALL_DBSI_DATA_new_thresh_all\PCa_with_biopsy\011_XU_YUN_LIN"
        #bin = float(raw_input("Please enter number of bins for Histogram plot\n"))
        bin = '100'
        pathname, filename = ntpath.split(file_path)
        stat_df = pd.read_csv(file_path)
        exclude_list = ['FA','Axial', 'Radial', 'Fiber Fraction', 'Fiber Fraction 1','Fiber FA 1','Fiber Axial 1', 'Fiber Radial 1', 'Fiber Fraction 2', 'Fiber FA 2','Fiber Axial 2', 'Fiber Radial 2']
        stat_df.drop(exclude_list, axis = 1, inplace = True)
        roi_name = stat_df['ROI Name'].unique().tolist()
        plt_counter = 0
        marker = ['D','s','o','^']
        marker = np.asarray(marker)
        min_adc = 0.15
        max_adc = 3.5 + min_adc
        adc_constant = min_adc
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
                sns.set(font_scale=2)
                sns.set_style("white")
                type = np.repeat(type,idx,axis=0)
                d = {'Type': type,'Fraction': np.concatenate((rfr_adc_1_df['Restricted Fraction 1'],rfr_adc_2_df['Restricted Fraction 2'],hfr_adc_df['Hindered Fraction'],wfr_adc_df['Water Fraction']),axis=0), 'ADC': np.concatenate((rfr_adc_1_df['Restricted ADC 1'],rfr_adc_2_df['Restricted ADC 2'],hfr_adc_df['Hindered ADC'],wfr_adc_df['Water ADC']),axis=0)}
                df = pd.DataFrame(data=d,index=range(len(type)))

                ### plot scatter plot

                plt.switch_backend('TkAgg')
                plt.rc('font', weight='bold')
                plt.rc('font', style='normal')
                plt.rc('xtick', labelsize=50)
                plt.rcParams['axes.linewidth'] = 7
                ax = sns.lmplot(data=df,x='ADC',y='Fraction',hue='Type',scatter=True,fit_reg=False,ci=None,markers=marker[non_zero_idx].tolist(),scatter_kws={"s": 100},palette=palette)
                ax.set_xlabels(fontweight='bold', fontsize=50)
                ax.set_ylabels(fontweight='bold', fontsize=50)
                ax.set(xlim=(-0.1,4))
                ax.fig.set_figwidth(24)
                ax.fig.set_figheight(15)
                adc = np.concatenate((rfr_adc_1_df['Restricted ADC 1'],rfr_adc_2_df['Restricted ADC 2'],hfr_adc_df['Hindered ADC'],wfr_adc_df['Water ADC']),axis=0)
                fraction = np.concatenate((rfr_adc_1_df['Restricted Fraction 1'],rfr_adc_2_df['Restricted Fraction 2'],hfr_adc_df['Hindered Fraction'],wfr_adc_df['Water Fraction']),axis=0)
                dti_adc = np.asarray(dti_adc_df['ADC'])
                ax.savefig(os.path.join(pathname,'Plot_scatter_%s.tiff'%(roi_name[i])),format='tiff', dpi=300)
                plt.close()

                ### Plot DBSI profiles histogram

                plt.switch_backend('TkAgg')
                plt.rc('font', weight='bold')
                plt.rc('font', style='normal')
                plt.rc('xtick', labelsize=50)
                plt.rc('ytick', labelsize=50)
                #fig = plt.figure(figsize=(26,18))
                ax = fig.add_subplot(1,1,1)
                adc_nonneg_idx = [a for a in range(len(adc)) if adc[a]>=0]
                adc_nonneg = adc[adc_nonneg_idx]
                adc_log = np.log10(adc_nonneg + adc_constant)
                dti_adc_nonneg_idx = [a for a in range(len(dti_adc)) if dti_adc[a]>=0]
                dti_adc_nonneg = dti_adc[dti_adc_nonneg_idx]
                dti_adc_log = np.log10(dti_adc_nonneg + adc_constant)
                bins = np.linspace(np.log10(min_adc),np.log10(max_adc), bin)
                bin_width = bins[1]-bins[0]
                bin_means, bin_edges, binnumber = stats.binned_statistic(adc_log,fraction[adc_nonneg_idx], statistic='mean',bins=bins,range=(np.log10(min_adc),np.log10(max_adc)))
                bin_count, bin_edges, binnumber = stats.binned_statistic(adc_log,fraction[adc_nonneg_idx], statistic='count',bins=bins,range=(np.log10(min_adc),np.log10(max_adc)))
                dti_bin_count, dti_bin_edges, dti_binnumber = stats.binned_statistic(dti_adc_log, dti_adc_log, statistic='count',bins=bins,range=(np.log10(min_adc),np.log10(max_adc)))
                new_idx = [len(bin_edges[bin_edges<np.log10(rfr_adc_1_df['Restricted ADC 1'].max() + adc_constant)]),len(bin_edges[bin_edges<np.log10(rfr_adc_2_df['Restricted ADC 2'].max() + adc_constant)]) - len(bin_edges[bin_edges<np.log10(rfr_adc_1_df['Restricted ADC 1'].max() + adc_constant)]),len(bin_edges[bin_edges<np.log10(hfr_adc_df['Hindered ADC'].max() + adc_constant)]) - len(bin_edges[bin_edges<np.log10(rfr_adc_2_df['Restricted ADC 2'].max() + adc_constant)]),len(bin_edges[bin_edges<np.log10(wfr_adc_df['Water ADC'].max() + adc_constant)]) - len(bin_edges[bin_edges<np.log10(hfr_adc_df['Hindered ADC'].max() + adc_constant)])]

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

                ### Plot frequency vs. Isotropic ADC

                ax.bar(bin_edges[0:len(bin_count)],bin_count/len(adc_nonneg),bin_width,align='edge',color=color)
                ax.set_ylim(0,0.1)
                #ticks = np.arange(0, 0.7, 0.2)
                ax.set_xlim(np.log10(min_adc),np.log10(max_adc+min_adc))
                fig.canvas.draw()
                ticks = [item.get_text() for item in ax.get_xticklabels()]
                ticks = [s.replace(u'\u2212', '-') for s in ticks]
                ticks = [float(s) for s in ticks if s!=u'']
                ticks = np.power(10,np.asarray(ticks)) - min_adc
                ticks_1 = np.round(np.asarray(ticks[0]),decimals=2)
                ticks_2 = np.round(np.asarray(ticks[1:]),decimals=1)
                ticks_1 = [ticks_1]
                ticks_2 = ticks_2.tolist()
                ticks = ticks_1 + ticks_2
                ticks.insert(0,u'\u200a')
                ticks.insert(len(ticks),round(np.power(10,np.log10(max_adc)) - min_adc,1))
                # ticks = ['%.2f'%s for s in ticks]
                ax.set_xticklabels(ticks)
                ax.set_xlabel('Isotropic ADC', fontweight='bold', fontsize=60)
                ax.set_ylabel('Frequency', fontweight='bold', fontsize=60)
                #ax.legend(handles=handles[non_zero_idx].tolist(), loc=2, fontsize=40)
                #plt.savefig(os.path.join(pathname,'Plot_log_hist_freq_%s.tiff'%(roi_name[i])),format='tiff', dpi=300)
                plt.savefig(os.path.join(pathname, 'Plot_log_hist_freq_%s.tiff' % (roi_name[i])), format='png')
                plt.close()

                ### Plot fractions per bin vs. Isotropic ADC

                fig = plt.figure(figsize=(26,18))
                ax = fig.add_subplot(1,1,1)
                ax.bar(bin_edges[0:len(bin_count)],bin_means,bin_width,align='edge',color=color)
                ax.set_xlim(np.log10(min_adc),np.log10(max_adc+min_adc))
                ax.set_ylim(0,1)
                #ticks = np.arange(0, 0.7, 0.2)
                ax.set_xticklabels(ticks)
                ax.set_xlabel('Isotropic ADC',fontweight='bold', fontsize=60)
                ax.set_ylabel('Fraction Bin Means',fontweight='bold', fontsize=60)
                #ax.legend(handles=handles[non_zero_idx].tolist(), loc=2, fontsize=40)
                plt.savefig(os.path.join(pathname,'Plot_log_hist_mean_%s.tiff'%(roi_name[i])),format='tiff', dpi=300)
                plt.close()

                ### plot fraction*frequncy per bin vs. Isotropic ADC

                fig = plt.figure(figsize=(26,18))
                ax = fig.add_subplot(1,1,1)
                ax.bar(bin_edges[0:len(bin_count)],bin_means*(bin_count/len(adc_nonneg)),bin_width,align='edge',color=color)
                ax.set_xlim(np.log10(min_adc),np.log10(max_adc+min_adc))
                ax.set_ylim(0,0.06)
                #ticks = np.arange(0, 0.7, 0.2)
                ax.set_xticklabels(ticks)
                ax.set_xlabel('Isotropic ADC',fontweight='bold', fontsize=60)
                ax.set_ylabel('Fraction Bin Means*Frequency',fontweight='bold',fontsize=60)
                #ax.legend(handles=handles[non_zero_idx].tolist(), loc=2, fontsize=40)
                #plt.savefig(os.path.join(pathname,'Plot_log_hist_mean_freq_%s.tiff'%(roi_name[i])),format='tiff', dpi=300)
                plt.savefig(os.path.join(pathname, 'Plot_log_hist_mean_freq_%s.tiff'%(roi_name[i])), format='png')
                plt.close()

                ### plot frequency vs. DTI ADC

                fig = plt.figure(figsize=(26,18))
                ax_dti = fig.add_subplot(1,1,1)
                DTI_ADC = mpatches.Patch(color='blue', label='DTI ADC')
                dti_handle = [DTI_ADC]
                ax_dti.bar(bin_edges[0:len(bin_count)],dti_bin_count/len(adc_nonneg),bin_width,align='edge',color='blue')
                ax_dti.set_xlim(np.log10(min_adc),np.log10(max_adc))
                ax.set_ylim(0,0.1)
                #ticks = np.arange(0, 0.7, 0.2)
                ax_dti.set_xticklabels(ticks)
                ax_dti.set_xlabel('DTI ADC',fontweight='bold',fontsize=60)
                ax_dti.set_ylabel('Frequency',fontweight='bold', fontsize=60)
                #ax_dti.legend(handles=dti_handle, loc=2, fontsize=40)
                plt.savefig(os.path.join(pathname,'Plot_log_hist_dti_freq_%s.tiff'%(roi_name[i])),format='png')
                plt.close()

                number = np.linspace(1,len(bin_means),len(bin_means))
                hist_df = pd.DataFrame({'Bin Number':number, 'Frequency':bin_count/len(adc_nonneg), 'Fraction Bin Means':bin_means, 'Fraction Bin MeansxFrequency':bin_means*(bin_count/len(adc_nonneg)), 'DTI Frequency':dti_bin_count/len(adc_nonneg)})
                hist_df.to_csv((os.path.join(pathname,'Bin_data_%s.csv' %(roi_name[i]))),index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path",dest="file_path",help='Working Folder',required=False,default='')
    parser.add_argument("--bin",required=False,default=128)
    args = parser.parse_args()
    main(args.file_path,args.bin)
