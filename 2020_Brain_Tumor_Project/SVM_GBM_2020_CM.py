#----------------------------------------------------------------------
# SVM for brain tumor DHI classification
#
# Patient stratification was used
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
                 y_range
                 ):

        self.proj_dir  = proj_dir
        self.DBSI_file = DBSI_file
        self.sample_1  = sample_1
        self.sample_2  = sample_2
        self.sample_3  = sample_3
        self.sample_4  = sample_4
        self.x_range   = x_range
        self.y_range   = y_range

    def data_load(self):

        df = pd.read_csv(os.path.join(self.proj_dir, self.DBSI_file))

        return df

    def sample_strat(self):

        df = self.data_load()

        train_names = []

        test_names  = [
                       self.sample_1,
                       self.sample_2,
                       self.sample_3,
                       self.sample_4
                       ]
             
        df_train = pd.DataFrame({'A': []})
        df_test  = pd.DataFrame({'A': []})
        
        unique_names = unique(df.iloc[:, column])
        
        for name in unique_names:
            
            if name not in test_names:
                train_names.append(name)

        for name in test_names:
            
            if df_test.empty:
                df_test = df[df.iloc[:, column] == name]
            else:
                df_test = df_test.append(df[df.iloc[:, column] == name])

        for name in train_names:
            
            if df_train.empty:
                df_train = df[df.iloc[:, column] == name]
                
            else:
                df_train = df_train.append(df[df.iloc[:, column] == name])
                
        print('\ntrain samples:\n', train_names)
        print('\ntest samples:\n', test_names)

        return df_train, df_test, test_names

    def data_split(self):

        df_train, df_test, test_names = self.sample_strat()

        x_train = df_train.iloc[:, self.x_range]
        y_train = df_train.iloc[:, self.y_range]

        x_test = df_test.iloc[:, self.x_range]
        y_test = df_test.iloc[:, self.y_range]

        x_test1 = df_test.iloc[:, self.x_range].loc[df_test['ID'] == self.sample_1]
        y_test1 = df_test.iloc[:, self.y_range].loc[df_test['ID'] == self.sample_1]

        x_test2 = df_test.iloc[:, self.x_range].loc[df_test['ID'] == self.sample_2]
        y_test2 = df_test.iloc[:, self.y_range].loc[df_test['ID'] == self.sample_2]

        x_test3 = df_test.iloc[:, self.x_range].loc[df_test['ID'] == self.sample_3]
        y_test3 = df_test.iloc[:, self.y_range].loc[df_test['ID'] == self.sample_3]

        x_test4 = df_test.iloc[:, self.x_range].loc[df_test['ID'] == self.sample_4]
        y_test4 = df_test.iloc[:, self.y_range].loc[df_test['ID'] == self.sample_4]

        x_test_list = [x_test, x_test1, x_test2, x_test3, x_test4]
        y_test_list = [y_test, y_test1, y_test2, y_test3, y_test4]
    

##        x_test_list = df_test.iloc[:, self.x_range]
##        y_test_list = df_test.iloc[:, self.y_range]
##
##        for sample in test_names:
##
##            x_test1 = df_test.iloc[:, self.x_range].loc[df_test['ID'] == sample]
##            y_test1 = df_test.iloc[:, self.y_range].loc[df_test['ID'] == sample]
##
##            x_test_list.append(x_test1)
##            y_test_list.append(y_test1)

        return x_train, y_train, x_test_list, y_test_list

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
                    ).fit(x_train, y_train)

    y_pred   = svm_model.predict(x_test)
    accuracy = svm_model.score(x_test, y_test)
    
    return y_pred, accuracy

# ----------------------------------------------------------------------------------
# calculate confusion matrix
# ----------------------------------------------------------------------------------
def calculate_cm():

    cm = confusion_matrix(y_test, y_pred)
   
    cm_norm  = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], 2)

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
# model hyper parameters
# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    
    c            = 500
    Degree       = 5
    Gamma        = 0.5
    column_DTI   = 3
    column_DBSI  = 4
    column       = column_DBSI
    x_range_DTI  = [0, 1]
    y_range_DTI  = 2

    x_range_DBSI = [1, 2, 3]
    y_range_DBSI = [0]
    x_range      = x_range_DBSI
    y_range      = y_range_DBSI
    
    sample_1     = 'B_127_1'
    sample_2     = 'B_127_2'
    sample_3     = 'B_124_5'
    sample_4     = 'B_124_7'

    proj_dir   = r'\\10.39.42.102\temp\Anthony_Wu\Zezhong_assist\JAAH'
    result_dir = r'\\10.39.42.102\temp\\Anthony_Wu\Zezhong_assist\JAAH\2019_yzz'
    DBSI_file  = 'all_data_brain.csv'
    DTI_file   = 'DTI_data_labeled.csv'

    kernel_function = 'poly'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------

    print("SVM classification: start...")


    start = timeit.default_timer()

    x_train, y_train, x_test_list, y_test_list = GBM_data(
                                                          proj_dir,
                                                          DBSI_file,
                                                          sample_1,
                                                          sample_2,
                                                          sample_3,
                                                          sample_4,
                                                          x_range,
                                                          y_range
                                                          ).data_split()

    cm_list      = []
    cm_norm_list = []

    for x_test, y_test in zip(x_test_list, y_test_list):

        y_pred, accuracy = SVM_pred()

        cm, cm_norm = calculate_cm()

        print(cm)

        cm_list.append(cm)
        
        cm_norm_list.append(cm_norm)

    print(sample_1, '\n', cm_list[0])
    print(sample_2, '\n', cm_list[1])
    print(sample_3, '\n', cm_list[2])
    print(sample_4, '\n', cm_list[3])
    print(sample_1, '\n', cm_norm_list[0])
    print(sample_2, '\n', cm_norm_list[1])
    print(sample_3, '\n', cm_norm_list[2])
    print(sample_4, '\n', cm_norm_list[3])
    
    print('\nSVM C value: ', c)
    print('SVM degree:  ', Degree)
    print('SVM gamma:   ', Gamma)
 
    stop = timeit.default_timer()
    running_seconds = np.around(stop-start, 0)
    print('SVM Running Time:', running_seconds, 'seconds')


