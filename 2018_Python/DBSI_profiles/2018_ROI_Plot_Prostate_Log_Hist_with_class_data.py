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
import Tkinter
from mpl_toolkits.mplot3d import Axes3D
import copy

def main(file_path = '',bin=128):
    mul_file = 0
    plt_counter = 0
    min_adc = 0.15
    max_adc = 3.5 + min_adc

    if file_path == '':
        root = Tkinter.Tk()
        file_path = tkFileDialog.askopenfilenames(title="Select Statistics File")
        bin = float(raw_input("Please enter number of bins for Histogram plot\n"))
        if len(file_path) > 1:
            mul_file = 1
        else:
            mul_file = 0
            file_path = file_path[0]

    if mul_file == 0:
        pathname, filename = ntpath.split(file_path)
        stat_df  = pd.read_csv(file_path)
        roi_name = stat_df['ROI Name'].unique().tolist()
        for i in range(len(roi_name)):
            index = stat_df.index[stat_df['ROI Name'] == roi_name[i]].tolist()
            fr_adc_df = stat_df.loc[index,["Fractions","ADC"]]
            fr_adc_df.dropna(inplace=True)
            idx = [len(fr_adc_df.index)]

            if all(value == 0 for value in idx):
                print('No data to plot for ROI Name -%s' %(roi_name[i]))
                continue
            else:
                plt_counter = plt_counter + 1
                sns.set(font_scale=3)
                sns.set_style("white")
                adc = np.asarray(fr_adc_df['ADC'])
                fraction =  np.asarray(fr_adc_df['Fractions'])
                plt.switch_backend('TkAgg')
                plt.rc('font', weight='bold')
                plt.rc('font', style='normal')
                plt.rc('xtick', labelsize=35)
                plt.rc('ytick', labelsize=35)
                fig = plt.figure(figsize=(24,15))
                ax = fig.add_subplot(1,1,1)
                bin_width = adc[1]-adc[0]
                fraction[-1] = 0
                ax.bar(adc,fraction,bin_width,align='edge')

                ax.set_ylim(0,0.08)
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
                ax.set_xticklabels(ticks)
                ax.set_xlabel('ADC',fontweight='bold',fontsize=35)
                ax.set_ylabel('Fraction Bin Means',fontweight='bold',fontsize=35)
                plt.savefig(os.path.join(pathname,'Plot_log_hist_mean_%s_%s.tiff'%(filename[:-4],roi_name[i])),format='tiff', dpi=100)
                plt.close()
    else:
        fraction = np.zeros([int(bin-1),len(file_path)])
        files = list(file_path)
        fname = [ntpath.split(f)[1] for f in files]
        for k,file in enumerate(file_path):
            pathname, filename = ntpath.split(file)
            stat_df  = pd.read_csv(file)
            roi_name = stat_df['ROI Name'].unique().tolist()
            for i in range(len(roi_name)):
                index = stat_df.index[stat_df['ROI Name'] == roi_name[i]].tolist()
                fr_adc_df = stat_df.loc[index,["Fractions","ADC"]]
                fr_adc_df.dropna(inplace=True)
                idx = [len(fr_adc_df.index)]

                if all(value == 0 for value in idx):
                    print('No data to plot for ROI Name -%s' %(roi_name[i]))
                    continue
                else:
                    plt_counter = plt_counter + 1
                    sns.set(font_scale=3)
                    sns.set_style("white")
                    fraction[:,k] = fr_adc_df['Fractions']
        adc = np.asarray(fr_adc_df['ADC'])
        plt.switch_backend('TkAgg')
        plt.rc('font', weight='bold')
        plt.rc('font', style='normal')
        plt.rc('xtick', labelsize=35)
        plt.rc('ytick', labelsize=35)
        color = ['y','g','b','r']*len(file_path)
        bin_width = adc[1]-adc[0]
        sum_adc = np.zeros([1,len(file_path) + len(file_path)])
        fr_idx = 0
        fig = plt.figure(figsize=(24,15))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(10,-60)
        for fr_num,z in zip(range(0,len(file_path)),[1000*nbr for nbr in range(0,len(file_path))]):
            fraction[-1,fr_num] = 0
            ax.bar(adc,fraction[:,fr_num],zs=z*np.ones([len(fraction[:,fr_num])]),zdir='y',width=bin_width,color=color[fr_num],alpha=0.7)
            sum_adc[:,fr_idx] = np.sum(fraction[:,fr_num])
            fr_idx = fr_idx + 1
            sum_adc[:,fr_idx] = (np.sum(np.multiply(fraction[:,fr_num],(np.power(10,adc)) - min_adc)))/np.sum(fraction[:,fr_num])
            fr_idx = fr_idx + 1
        groups_orig = [name[:-4] for name in fname]
        grps = 0
        for p in range(0,len(file_path)):
            if p ==0:
                txt = '%s\nSum_Fraction - %s\nMean_ADC - %s\n\n' %(groups_orig[p],sum_adc[0,grps],sum_adc[0,grps+1])
            else:
                txt = txt + '%s\nSum_Fraction - %s\nMean_ADC - %s\n\n' %(groups_orig[p],sum_adc[0,grps],sum_adc[0,grps+1])
            grps = grps + 2
        ax.set_yticks([1000*nbr for nbr in range(0,len(file_path))])
        ax.set_yticklabels([name[:-4] for name in fname],fontsize=35)
        ax.set_zlim(0,0.05)
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
        # ticks.insert(0,u'\u200a')
        ticks.insert(len(ticks),round(np.power(10,np.log10(max_adc)) - min_adc,1))
        ax.set_xticklabels(ticks)
        ax.set_xlabel('ADC',fontweight='bold')
        ax.set_zlabel('Fraction Bin Means',fontweight='bold')
        plt.savefig(os.path.join(os.path.dirname(pathname),'Plot_log_hist_mean.tiff'),format='tiff', dpi=50)
        plt.close()
        fig = plt.figure(figsize=(24,15))
        ax = fig.add_subplot(111)
        for fr_num in (range(0,len(file_path))):
            fraction[-1,fr_num] = 0
            ax.plot(adc,fraction[:,fr_num],color=color[fr_num],alpha=0.7)
        ax.set_ylim(0,0.1)
        ax.set_xlim(np.log10(min_adc),np.log10(max_adc+min_adc))
        ticks1 = copy.deepcopy(ticks)
        ticks1.insert(0,u'\u200a')
        ax.set_xticklabels(ticks1)
        ax.set_xlabel('ADC',fontweight='bold',fontsize=35)
        ax.set_ylabel('Fraction Bin Mean',fontweight='bold',fontsize=35)
        ax.legend([name[:-4] for name in fname])
        ax.grid()
        font = {'family': 'serif',
                'color':  'darkred',
                'weight': 'normal',
                'size': 15,
                }
        plt.text(0.01,0.03,txt,fontdict=font, withdash=False)
        plt.savefig(os.path.join(os.path.dirname(pathname),'Plot_log_hist_mean_2d.tiff'),format='tiff', dpi=100)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path",dest="file_path",help='Working Folder',required=False,default='')
    parser.add_argument("--bin",required=False,default=128)
    args = parser.parse_args()
    main(args.file_path,args.bin)
