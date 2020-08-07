#----------------------------------------------------------------------
# SVM for brain tumor DHI classification
#
# Patient stratification was used
#
# Author: Zezhong Ye, Anthony_Wu
#
# Contact: ze-zhong@wustl.edu, atwu@wustl.edu
#
# Date: 08-14-2019
#-------------------------------------------------------------------------------------------

import os
import string
import timeit
import numpy as np
import itertools
import pandas as pd               
import seaborn as sn
import glob2 as glob
import matplotlib.pyplot as plt
from matplotlib.pylab import *
from time import gmtime, strftime
from itertools import cycle, combinations
import nibabel as nib

from sklearn.svm import SVC
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import auc, precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier



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
# sample stratification
# ----------------------------------------------------------------------------------
def unique(list1):
    
    unique_list = []
    
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
            
    return unique_list

# ----------------------------------------------------------------------------------
# manual sample stratification
# ----------------------------------------------------------------------------------
class GBM_data(object):

    def __init__(
                 self,
                 proj_dir,
                 DBSI_file,
                 sample_1,
                 sample_2,
                 sample_3,
                 sample_4,
                 x_range,
                 y_range,
                 pred_dir,
                 pred_file,
                 x_range_pred
                 ):

        self.proj_dir     = proj_dir
        self.DBSI_file    = DBSI_file
        self.sample_1     = sample_1
        self.sample_2     = sample_2
        self.sample_3     = sample_3
        self.sample_4     = sample_4
        self.x_range      = x_range
        self.y_range      = y_range
        self.pred_dir     = pred_dir
        self.pred_file    = pred_file
        self.x_range_pred = x_range_pred

    def data_load(self):

        df = pd.read_csv(os.path.join(self.proj_dir, self.DBSI_file))

        df.loc[df['ROI_class'] == '1',  'y_cat'] = 1
        df.loc[df['ROI_class'] == '2',  'y_cat'] = 2
        df.loc[df['ROI_class'] == '3',  'y_cat'] = 3

        class1 = df[df.iloc[:, 0] == 1]
        class2 = df[df.iloc[:, 0] == 2]
        class3 = df[df.iloc[:, 0] == 3]

        df2 = pd.concat([class1, class2, class3])
                                                                      
        return df2

    def sample_strat(self):

        df2 = self.data_load()

        train_names = []

        test_names  = [
                       self.sample_1,
                       self.sample_2,
                       self.sample_3,
                       self.sample_4
                       ]
             
        df_train = pd.DataFrame({'A': []})
        df_test  = pd.DataFrame({'A': []})
        
        unique_names = unique(df2.iloc[:, column])
        
        for name in unique_names:
            
            if name not in test_names:
                train_names.append(name)

        for name in test_names:
            
            if df_test.empty:
                df_test = df2[df2.iloc[:, column] == name]
            else:
                df_test = df_test.append(df2[df2.iloc[:, column] == name])

        for name in train_names:
            
            if df_train.empty:
                df_train = df2[df2.iloc[:, column] == name]
                
            else:
                df_train = df_train.append(df2[df2.iloc[:, column] == name])
                
        print('\ntrain samples:\n', train_names)
        print('\ntest samples:\n', test_names)

        return df_train, df_test, test_names

    def data_split(self):

        df_train, df_test, test_names = self.sample_strat()

        x_train = df_train.iloc[:, self.x_range]
        y_train = df_train.iloc[:, self.y_range]

        x_test  = df_test.iloc[:, self.x_range]
        y_test  = df_test.iloc[:, self.y_range]
 
        return x_train, y_train, x_test, y_test

    def data_predict(self):
        
        df_pred = pd.read_csv(os.path.join(self.pred_dir, self.pred_file))
        x_pred  = df_pred.iloc[:, self.x_range_pred]

        return df_pred, x_pred

# ----------------------------------------------------------------------------------
# trainning SVM model
# ----------------------------------------------------------------------------------
def SVM_pred():
    
    svm_model = SVC(
                    C=c,
                    cache_size=1000,
                    class_weight='balanced',
                    coef0=0.0,
                    decision_function_shape=None,
                    degree=Degree,
                    gamma=Gamma,
                    kernel=kernel_function,
                    max_iter=-1,
                    probability=False,
                    random_state=None,
                    shrinking=True,
                    tol=0.001,
                    verbose=False
                    ).fit(x_train, y_train.values.ravel())

    y_pred   = svm_model.predict(x_test)
    accuracy = svm_model.score(x_test, y_test)

    y_prediction = svm_model.predict(x_pred)
##    y_prediction_label = np.argmax(y_prediction, axis=1)
    
    return y_pred, accuracy, y_prediction

# ----------------------------------------------------------------------------------
# calculate confusion matrix
# ----------------------------------------------------------------------------------
def calculate_cm():

    cm = confusion_matrix(y_test, y_pred)
   
    cm_norm = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], 2)

    return cm, cm_norm

# ----------------------------------------------------------------------------------
# plot confusion matrix
# ----------------------------------------------------------------------------------                  
def plot_cm():
    
    ax = sn.heatmap(
                    cm_list[1],
                    annot=True,
                    annot_kws={"size": 18, "weight": 'bold'},
                    cmap="Blues",
                    linewidths=.5
                    )

    ax.set_ylim(3, 0)
    ax.set_aspect('equal')
    #plt.figure(figsize = (10,7))
    #sn.set(font_scale=1.4)#for label size
    plt.tight_layout()

    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=3, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=3, color='k', linewidth=4)

    cm_filename = 'cm' + '_' + \
                  str(c) + \
                  str(Degree) + '_' + \
                  str(Gamma) + '_' + \
                  strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
            
    plt.savefig(os.path.join(result_dir, cm_filename), format='png', dpi=600)
    plt.show()
    plt.close()

# ----------------------------------------------------------------------------------
# create prediction
# ----------------------------------------------------------------------------------
def GBM_predict():

    x_index = np.asarray(df_pred.iloc[:, [1]])[:, 0]
    y_index = np.asarray(df_pred.iloc[:, [2]])[:, 0]
    z_index = np.asarray(df_pred.iloc[:, [3]])[:, 0]

    img = np.zeros(shape=(256, 256, 224))

    for i in range(x_index.shape[0]):
        
            img[x_index[i], y_index[i], z_index[i]] = y_prediction[i]
            
    aff = nib.load(os.path.join(
                                pred_dir,
                                DBSI_folder,
                                overlaid_map,
                                )
                   ).get_affine()
    
    GBM_prediction = nib.Nifti1Image(img, aff)

    filename = str(pred_map_name) + '_' + \
               strftime('%d_%b_%Y_%H_%M_%S', gmtime()) + \
               '.nii'
    
    nib.save(GBM_prediction, os.path.join(pred_dir, filename))

    return GBM_prediction
    
# ----------------------------------------------------------------------------------
# model hyper parameters
# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    
    c            = 150
    Degree       = 4
    Gamma        = 0.5
    column_DTI   = 3
    column_DBSI  = 4
    column       = column_DBSI
    x_range_DTI  = [0, 1]
    y_range_DTI  = [2]
    x_range_DBSI = [1, 2, 3]
    y_range_DBSI = [0]
    x_range_pred = [19, 16, 12]
    
    x_range      = x_range_DBSI
    y_range      = y_range_DBSI
    
    sample_1      = 'B_122_4_5'
    sample_2      = ''
    sample_3      = ''
    sample_4      = ''
 
    proj_dir      = r'\\10.39.42.102\temp\Zezhong_Ye\Human_Brain_Biopsy_Tissue_Albert_kim\GBM_ML'
    result_dir    = r'\\10.39.42.102\temp\Zezhong_Ye\Human_Brain_Biopsy_Tissue_Albert_kim\GBM_ML'
    pred_dir      = r'\\10.39.42.102\temp\Zezhong_Ye\Human_Brain_Biopsy_Tissue_Albert_kim\GBM_ML'

    DBSI_file     = 'all_data_brain.csv'
    DTI_file      = 'DTI_data_labeled.csv'
    pred_file     = 'GBM_voxel_04_Feb_2020_07_11_36.csv'
    overlaid_map  = 'b0_map.nii'
    pred_map_name = 'histology_pred_map'
    DBSI_folder   = 'DBSI_results_0.1_0.1_1.5_1.5_2.5_2.5' 

    kernel_function = 'poly'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------

    print("SVM classification: start...")

    start = timeit.default_timer()

    GBM_Data = GBM_data(
                        proj_dir,
                        DBSI_file,
                        sample_1,
                        sample_2,
                        sample_3,
                        sample_4,
                        x_range,
                        y_range,
                        pred_dir,
                        pred_file,
                        x_range_pred
                        )

    x_train, y_train, x_test, y_test = GBM_Data.data_split()

    df_pred, x_pred = GBM_Data.data_predict()

    y_pred, accuracy, y_prediction = SVM_pred()

    cm, cm_norm = calculate_cm()

    predict_map = GBM_predict()

    print(x_pred)
    print(y_prediction)
    df3 = pd.concat([x_pred, y_prediction])
    print(df3)

    print('test acc:', np.around(accuracy, 3))
    print(cm)
    print(cm_norm)
    
    print('SVM C value: ', c)
    print('SVM degree:  ', Degree)
    print('SVM gamma:   ', Gamma)
 
    stop = timeit.default_timer()
    running_seconds = np.around(stop-start, 0)
    print('SVM Running Time:', running_seconds, 'seconds')


