#----------------------------------------------------------------------
# deep learning classifier using a multiple layer perceptron (MLP)
# batch normalization was used
#-------------------------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import seaborn as sns
import glob2 as glob
import nibabel as nib
from functools import partial
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle
from math import sqrt

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
# construct dataset
# ----------------------------------------------------------------------------------
def data_prepare():
    
    maps_list = [
                 'ADC',                        #08
                 'FA',                         #09
                 'fiber fraction',             #10
                 'Highly restricted fraction', #11
                 'restricted fraction',        #12
                 'hindered fraction',          #13
                 'water fraction',             #14
                ]

    df = pd.read_csv(os.path.join(proj_dir, train_file))

    df.loc[(df['Sub_ID'] == 'week_04'), 'y_cat'] = 'week_04'
    df.loc[(df['Sub_ID'] == 'week_12'), 'y_cat'] = 'week_12'
    df.loc[(df['Sub_ID'] == 'week_16'), 'y_cat'] = 'week_16'
    df.loc[(df['Sub_ID'] == 'week_28'), 'y_cat'] = 'week_28'
    df.loc[(df['Sub_ID'] == 'week_32'), 'y_cat'] = 'week_32'

    class1 = df[df['y_cat'] == 'week_04']
    class2 = df[df['y_cat'] == 'week_12']
    class3 = df[df['y_cat'] == 'week_16']
    class4 = df[df['y_cat'] == 'week_28']
    class5 = df[df['y_cat'] == 'week_32']

    df2 = pd.concat([class1, class2, class3, class4, class5])

    return df2

# ----------------------------------------------------------------------------------
# construct dataset
# ----------------------------------------------------------------------------------
def boxplot():

    df2.head()          

    fig = plt.figure(figsize=(16, 8))

    for i, j, k, ylabel in zip(
                               range(len(metric_list)),
                               metric_list,
                               ytick_list,
                               label_list
                               ):

        x_value = df2['y_cat']
        y_value = df2.iloc[:, j]

        sns.set(style="darkgrid")

        ax = fig.add_subplot(2, 4, i+1)

##        ax = sns.boxplot(
##                         x=x_value,
##                         y=y_value,
##                         palette='Set1',
##                         width=0.5,
##                         orient='v',
##                         linewidth=1.5,
##                         whis=[5, 95],
##                         notch=None,
##                         showfliers=False,
##                         ax=ax
##                         )

        ax = sns.swarmplot(
                           x=x_value,
                           y=y_value,
                           color='black',
                           #edgecolor='gray',
                           size=3
                           )

        plt.xticks(fontsize=10, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(k, fontsize=10, fontweight='bold')

        for _,s in ax.spines.items():
            s.set_linewidth(1.5)
            s.set_color('black')

        ax.tick_params(direction='out', length=4, width=2, colors='k')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plt.xlabel('', fontweight='bold', fontsize=10)
        plt.ylabel(ylabel, fontweight='bold', fontsize=10)
        plt.grid(True)
        plt.tight_layout()

##    plt.savefig(
##                os.path.join(result_dir, 'boxplot.png'),
##                bbox_inches='tight',
##                facecolor=fig.get_facecolor()
##                )

    plt.show()
    plt.close()

# ----------------------------------------------------------------------------------
# model hyper parameters
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    metric_list = [8, 9, 10, 11, 12, 13, 14]

    ytick_list = [
                  [0, 0.5, 1, 1.5, 2, 2.5],
                  [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  [0, 0.2, 0.4, 0.6, 0.8, 1.0],
                  [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                  ]

    label_list = [
                  'DTI ADC',
                  'DTI FA',
                  'Fiber Fraction',
                  'Highly Restricted Fraction',
                  'Restricted Fraction',
                  'Hindered Fraction',
                  'Free Fraction'
                  ]
                  
    proj_dir    = r'\\10.39.42.102\temp\Zezhong_Ye\Human_Brain_Tumor_in_vivo\Glioma_WashU\C1_004'
    result_dir  = r'\\10.39.42.102\temp\Zezhong_Ye\Human_Brain_Tumor_in_vivo\Glioma_WashU\C1_004'
    log_dir     = r'\\10.39.42.102\temp\Zezhong_Ye\Human_Brain_Tumor_in_vivo\Glioma_WashU\C1_004'
    train_file  = 'gbm_voxel_2.csv'
    
    data_path()

    df2 = data_prepare()

    boxplot()

    print('boxplot: complete!')

    




