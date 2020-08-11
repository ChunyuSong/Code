#----------------------------------------------------------------------
# SVM for brain tumor DHI classification
#-------------------------------------------------------------------------------------------

import pandas as pd
import os
import glob2 as glob
import numpy as np
import itertools
import matplotlib.pyplot as plt                     
import seaborn as sn
import string
from matplotlib.pylab import *
from itertools import combinations
import timeit
from time import gmtime, strftime

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC

# ----------------------------------------------------------------------------------
# sample stratification
# ----------------------------------------------------------------------------------

def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def patient_stratification(df, test_frac):
    #print(df)
    unique_names = unique(df.iloc[:, 4])
    #print(unique_names)
    #tpi = test patient indices 
    tpi_num = int(np.ceil(len(unique_names)*test_frac))
    tpi = np.random.choice(len(unique_names), size=tpi_num, replace=False)
    test_names = []
    train_names = []
    df_test = pd.DataFrame({'A' : []})
    df_train = pd.DataFrame({'A' : []})
    
    for index in tpi:
        test_names.append(unique_names[index])
        if df_test.empty:
            df_test = df[df.iloc[:, column] == unique_names[index]]
        else:
            df_test = df_test.append(df[df.iloc[:, column] == unique_names[index]])
            
    for index in range(len(unique_names)):
        if index not in tpi:
            train_names.append(unique_names[index])
            if df_train.empty:
                df_train = df[df.iloc[:, column] == unique_names[index]]
            else:
                df_train = df_train.append(df[df.iloc[:, column] == unique_names[index]])
                
    #print(df_test.shape)
    #print(df_train.shape)
    return df_train, df_test

def patient_strat_manual(df, test_names):
    train_names = []
    df_test = pd.DataFrame({'A' : []})
    df_train = pd.DataFrame({'A' : []})
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
            
    #print(train_names)
    #print(test_names)
    #print(df_train.shape)
    #print(df_test.shape)
            
    return df_train, df_test 

def data_balancing(df, class0_ratio, class1_ratio, class2_ratio):
    
    class0        = df[df.iloc[:,0] == 1]
    class0_sample = class0.sample(int(class0.shape[0]*class0_ratio))
    class1        = df[df.iloc[:,0] == 2]
    class1_sample = class1.sample(int(class1.shape[0]*class1_ratio))
    class2        = df[df.iloc[:,0] == 3]
    class2_sample = class2.sample(int(class2.shape[0]*class2_ratio))

    df_2 = pd.concat(
                    [class0_sample,
                    class1_sample,
                    class2_sample,]
                    )

    return df_2

# ----------------------------------------------------------------------------------
# loading data
# ----------------------------------------------------------------------------------

def GBM_data(project_dir, sample_list):
    
    df = pd.DataFrame()
    # for f in filenames:
    #     data = pd.read_excel(f,'Sheet1',header = None)
    #     data.iloc[:, 1:] = (data.iloc[:, 1:]) / (data.iloc[:, 1:].max())
    #     data = data.iloc[data[data.iloc[:, 0] != 0].index]
    #     df = df.append(data)

    data = pd.read_csv(project_dir)
    #data = data_balancing(data, 0.5, 1, 1)

    # data.iloc[:, 7:] = (data.iloc[:, 7:]) / (data.iloc[:, 7:].max())
    #data = data.iloc[data[data.iloc[:, 0] != 0].index]
    df = df.append(data)

    # Do patient stratification, with around 4:1 ratio tts

    train, test = patient_strat_manual(df, sample_list)

    #train, test = patient_stratification(df, 0.2)
    #train, test = train_test_split(df, test_size=0.2)

    x_train = train.iloc[:, x_range]
    y_train = train.iloc[:, y_range]

    x_test  = test.iloc[:, x_range]
    y_test  = test.iloc[:, y_range]

    return x_train, y_train, x_test, y_test

# ----------------------------------------------------------------------------------
# trainning SVM model
# ----------------------------------------------------------------------------------

def SVM_prediction(c, Degree, Gamma, kernel_function):
    
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

    svm_predictions = svm_model.predict(x_test)
    accuracy        = svm_model.score(x_test, y_test)

    cm      = confusion_matrix(y_test, svm_predictions)
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 2)

    acc_class1  = cm[0, 0]/(cm[0, 0] + cm[0, 1] + cm[0, 2])
    acc_class2  = cm[1, 1]/(cm[1, 0] + cm[1, 1] + cm[1, 2])
    acc_class3  = cm[2, 2]/(cm[2, 0] + cm[2, 1] + cm[2, 2])
    acc_overall = accuracy

    b122_4_5_class1 = svm_predictions[0: 473]
    b122_4_5_class2 = svm_predictions[473: 692]
    b122_4_5_class3 = svm_predictions[692: 843]
    b95_1_class1    = svm_predictions[843: 1349]
    b95_1_class2    = svm_predictions[1349: 1394]
    b128_class1     = svm_predictions[1394: 1786]
    b94_class3      = svm_predictions[1786: 1963]

    acc_122_4_5_class1 = b122_4_5_class1[b122_4_5_class1 == 1].shape[0] / b122_4_5_class1.shape[0]
    acc_122_4_5_class2 = b122_4_5_class2[b122_4_5_class2 == 2].shape[0] / b122_4_5_class2.shape[0]
    acc_122_4_5_class3 = b122_4_5_class3[b122_4_5_class3 == 3].shape[0] / b122_4_5_class3.shape[0]
    acc_95_1_class1    = b95_1_class1[b95_1_class1 == 1].shape[0] / b95_1_class1.shape[0]
    acc_95_1_class2    = b95_1_class2[b95_1_class2 == 2].shape[0] / b95_1_class2.shape[0]
    acc_128_class1     = b128_class1[b128_class1 == 1].shape[0] / b128_class1.shape[0]
    acc_94_class3      = b94_class3[b94_class3 == 3].shape[0] / b94_class3.shape[0]

    acc_class1         = np.around(acc_class1, 3)
    acc_class2         = np.around(acc_class2, 3)
    acc_class3         = np.around(acc_class3, 3)
    acc_overall        = np.around(acc_overall, 3)
    acc_122_4_5_class1 = np.around(acc_122_4_5_class1, 3)
    acc_122_4_5_class2 = np.around(acc_122_4_5_class2, 3)
    acc_122_4_5_class3 = np.around(acc_122_4_5_class3, 3)
    acc_95_1_class1    = np.around(acc_95_1_class1, 3)
    acc_95_1_class2    = np.around(acc_95_1_class2, 3)
    acc_128_class1     = np.around(acc_128_class1, 3)
    acc_94_class3      = np.around(acc_94_class3, 3)

    stat_list = [
                 acc_class1,
                 acc_class2,
                 acc_class3,
                 acc_overall,
                 acc_122_4_5_class1,
                 acc_122_4_5_class2,
                 acc_122_4_5_class3,
                 acc_95_1_class1,
                 acc_95_1_class2,
                 acc_128_class1,
                 acc_94_class3
                ]

    stat_df = pd.DataFrame(
                           stat_list,
                           index=[
                                  'class1 overall acc',
                                  'class2 overall acc',
                                  'class3 overall acc',
                                  'overall acc',
                                  'B122_4_5 class1 acc',
                                  'B122_4_5 class2 acc',
                                  'B122_4_5 class3 acc',
                                  'B95_1 class1 acc',
                                  'B95_1 class2 acc',
                                  'B128 class1 acc',
                                  'B94 class3 acc'
                                  ],
                            columns=['accuracy']
                            )
                              
    ### plot confusion matrix heat map

    ax = sn.heatmap(
                    cm_norm,
                    annot=True,
                    cbar=True,
                    annot_kws={'size': 20, 'fontweight': 'bold'},
                    cmap="Blues",
                    linewidths=.5,
                    fmt=''
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
            
    #plt.savefig(os.path.join(result_dir, cm_filename), format='png', dpi=600)

    plt.show()

    return stat_df, cm, cm_norm

# ----------------------------------------------------------------------------------
# ROC and AUC
# ----------------------------------------------------------------------------------

def SVM_ROC(c, Degree, Gamma, kernel_function):

    SVM_model = OneVsRestClassifier(
                                    SVC(
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
                                        )
                                    ).fit(x_train, y_train)
            
    y_score   = SVM_model.decision_function(x_test)
    y_predict = SVM_model.predict(x_test)
    accuracy  = SVM_model.score(x_test, y_test)

    # Compute ROC curve and AUC for each class
    fpr       = dict()
    tpr       = dict()
    threshold = dict()
    roc_auc   = dict()

    for i in range(3):
        
        fpr[i], tpr[i], threshold[i] = roc_curve(
                                                 y_test,
                                                 y_score[:, i],
                                                 pos_label=i+1
                                                 )
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        fig = plt.figure(figsize=(7,7))
        
        plt.plot(
                 fpr[i],
                 tpr[i],
                 'b',
                 label='AUC=%0.3f'%roc_auc[i],
                 linewidth=6
                 )
        
        #plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([-0.03, 1])
        plt.ylim([0, 1.03])
        
        plt.axhline(y=0, color='k', linewidth=4)
        plt.axhline(y=1.03, color='k', linewidth=4)
        plt.axvline(x=-0.03, color='k', linewidth=4)
        plt.axvline(x=1, color='k', linewidth=4)
        
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=24, fontweight='bold')
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=24, fontweight='bold')
        #plt.xlabel('False Positive Rate', fontweight='bold', fontsize=25)
        #plt.ylabel('True Positive Rate', fontweight='bold', fontsize=25)
        plt.legend(loc='lower right', prop={'size': 24, 'weight': 'bold'}) 
        plt.grid(True)
        #plt.show()

        ROC_filename = 'ROC' + '_' + \
                        str(c) + \
                        str(Degree) + '_' + \
                        str(Gamma) + '_' + \
                        strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
            
        plt.savefig(os.path.join(result_dir, ROC_filename), format='png', dpi=600)

        roc_auc[i] = np.around(roc_auc[i], 3)

    return roc_auc

# ----------------------------------------------------------------------------------
# model hyper parameters
# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    
    c      = 150
    Degree = 5
    Gamma  = 0.5
    column_DTI   = 3
    column_DBSI  = 4
    column       = column_DBSI
    x_range_DTI  = [0, 1]
    y_range_DTI  = 2
    x_range_DBSI = [1, 2, 3]
    y_range_DBSI = [0]
    x_range      = x_range_DBSI
    y_range      = y_range_DBSI
    
    kernel_function = 'poly'
    sample_list     = ['B_122_4_5', 'B_95_1', 'B_128', 'B_94']

    DBSI_dir    = r'Z:\Anthony_Wu\Zezhong_assist\JAAH\all_data_brain.csv'
    DTI_dir     = r'z:\Anthony_Wu\Zezhong_assist\JAAH\DTI_data_labeled.csv'  
    result_dir  = r'Z:\Zezhong_Ye\2017_Science_Translational_Medicine\2019_yzz'
    project_dir = DBSI_dir

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------

    print("SVM classification: start...")

    start = timeit.default_timer()

    x_train, y_train, x_test, y_test = GBM_data(project_dir, sample_list)

    stat_df, cm, cm_norm = SVM_prediction(c, Degree, Gamma, kernel_function)

    roc_auc = SVM_ROC(c, Degree, Gamma, kernel_function)

    print('confusion matrix:\n', cm_norm)
    print('confusion matrix:\n', cm)
    print(stat_df)
    print('AUC for each class:\n', roc_auc)
    print('SVM C value: ', c)
    print('SVM degree:  ', Degree)
    print('SVM gamma:   ', Gamma)
 
    stop = timeit.default_timer()
    running_seconds = np.around(stop - start, 0)
    #running_minutes = np.around(running_seconds/60, 0)
    print('\nSVM Running Time:', running_seconds, 'seconds')


