#----------------------------------------------------------------------
# SVM for brain tumor DHI classification
#
# Patient stratification was used
#-------------------------------------------------------------------------------------------
import pandas as pd
import os
import glob2 as glob
import numpy as np
import itertools
from itertools import cycle, combinations
import matplotlib.pyplot as plt                     
import seaborn as sn
import string
from matplotlib.pylab import *
import timeit
from time import gmtime, strftime

from sklearn import datasets, svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.multiclass import OneVsRestClassifier
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

# ----------------------------------------------------------------------------------
# sample stratification with ratio
# ----------------------------------------------------------------------------------
def patient_stratification(df, test_frac):
    
    #print(df)
    unique_names = unique(df.iloc[:, 4])
    #print(unique_names)
    #tpi = test patient indices 
    tpi_num = int(np.ceil(len(unique_names) * test_frac))
    tpi = np.random.choice(len(unique_names), size=tpi_num, replace=False)
    test_names = []
    train_names = []
    df_test = pd.DataFrame({'A': []})
    df_train = pd.DataFrame({'A': []})
    
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

# ----------------------------------------------------------------------------------
# manual sample stratification
# ----------------------------------------------------------------------------------
def sample_strat_manual():

    df = pd.read_csv(os.path.join(proj_dir, DBSI_file))
    
    train_names = []
    
    df_test = pd.DataFrame({'A': []})
    df_train = pd.DataFrame({'A': []})
    
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
            
    print('train samples:\n', train_names)
    print('\ntest samples:\n', test_names)
    print(df_train.shape)
    print(df_test.shape)
    print(df_test)
    print(df_test.loc[df_test['ID'] == 'B_122_4_5'].shape[0])

    x_train = df_train.iloc[:, x_range]
    y_train = df_train.iloc[:, y_range]

    x_test  = df_test.iloc[:, x_range].loc[df_test['ID'] == 'B_122_4_5']
    y_test  = df_test.iloc[:, y_range].loc[df_test['ID'] == 'B_122_4_5']

    print(x_test, y_test)

    return x_train, y_train, x_test, y_test

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
    
    return y_pred, accuracy

# ----------------------------------------------------------------------------------
# calculate confusion matrix
# ----------------------------------------------------------------------------------
def calculate_cm():

    cm  = confusion_matrix(y_test, y_pred)
    cm1 = confusion_matrix(y_test[0: 843], y_pred[0: 843])
    cm2 = confusion_matrix(y_test[843: 1394], y_pred[843: 1394])
    cm3 = confusion_matrix(y_test[1394: 1786], y_pred[1394: 1786])
    cm4 = confusion_matrix(y_test[1786: 1963], y_pred[1786: 1963])
   
    cm_norm  = np.around(cm.astype('float')/cm.sum(axis=1)[:, np.newaxis], 2)
    cm_norm1 = np.around(cm1.astype('float')/cm1.sum(axis=1)[:, np.newaxis], 2)
    cm_norm2 = np.around(cm2.astype('float')/cm2.sum(axis=1)[:, np.newaxis], 2)
    cm_norm3 = np.around(cm3.astype('float')/cm3.sum(axis=1)[:, np.newaxis], 2)
    cm_norm4 = np.around(cm4.astype('float')/cm4.sum(axis=1)[:, np.newaxis], 2)

    cm_list = [
               cm, cm_norm,
               cm1, cm_norm1,
               cm2, cm_norm2,
               cm3, cm_norm3,
               cm4, cm_norm4
               ]

    return cm_list

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
# ROC and AUC
# ----------------------------------------------------------------------------------
def SVM_ROC():

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
                                    ).fit(x_train, y_train.values.ravel())
            
    y_score   = SVM_model.decision_function(x_test)
    y_predict = SVM_model.predict(x_test)
    accuracy  = SVM_model.score(x_test, y_test)

    return y_score

# ----------------------------------------------------------------------------------
# ROC 
# ----------------------------------------------------------------------------------
def DNN_ROC():

    fpr       = dict()
    tpr       = dict()
    threshold = dict()
    roc_auc   = dict()

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
        
    colors = cycle(['red', 'royalblue', 'green'])

    for i, color in zip(range(3), colors):
        
        fpr[i], tpr[i], threshold[i] = roc_curve(
                                                 y_test,
                                                 y_score[:, i],
                                                 pos_label = i+1
                                                 )
        roc_auc[i] = auc(fpr[i], tpr[i])

        print('ROC AUC %.2f' % roc_auc[i])
        
        plt.plot(
                 fpr[i],
                 tpr[i],
                 color=color,
                 label = 'AUC %0.2f' % roc_auc[i],
                 linewidth = 4
                 )
                
    plt.xlim([-0.03, 1])
    plt.ylim([0, 1.03])  
    plt.axhline(y=0, color='k', linewidth=4)
    plt.axhline(y=1.03, color='k', linewidth=4)
    plt.axvline(x=-0.03, color='k', linewidth=4)
    plt.axvline(x=1, color='k', linewidth=4)  
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    #plt.xlabel('False Positive Rate', fontweight='bold', fontsize=25)
    #plt.ylabel('True Positive Rate', fontweight='bold', fontsize=25)
    plt.legend(loc='lower right', prop={'size': 16, 'weight': 'bold'}) 
    plt.grid(True)

    ROC_filename = 'ROC' + '_' + \
                    str(c) + \
                    str(Degree) + '_' + \
                    str(Gamma) + '_' + \
                    strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
        
    plt.savefig(os.path.join(result_dir, ROC_filename), format='png', dpi=600)
    plt.show()
    plt.close()

# ----------------------------------------------------------------------------------
# precision recall curve
# ----------------------------------------------------------------------------------
def DNN_PRC():
    
    precision = dict()
    recall    = dict()
    threshold = dict()
    prc_auc   = []

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
        
    colors = cycle(['red', 'royalblue', 'green'])

    for i, color in zip(range(3), colors):

        precision[i], recall[i], _ = precision_recall_curve(
                                                            y_test,
                                                            y_score[:, i],
                                                            pos_label=i+1
                                                            )
        
        RP_2D = np.array([recall[i], precision[i]])
        RP_2D = RP_2D[np.argsort(RP_2D[:,0])]

        prc_auc.append(auc(RP_2D[1], RP_2D[0]))
        
        print('PRC AUC %.2f' % auc(RP_2D[1], RP_2D[0]))
                
        plt.plot(
                 recall[i],
                 precision[i],
                 color=color,
                 linewidth=3,
                 label='AUC %0.2f' % prc_auc[i]
                 )

    plt.xlim([0, 1.03])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.03, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=1.03, color='k', linewidth=4) 
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    #plt.xlabel('recall', fontweight='bold', fontsize=16)
    #plt.ylabel('precision', fontweight='bold', fontsize=16)
    plt.legend(loc='lower left', prop={'size': 16, 'weight': 'bold'}) 
    plt.grid(True)

    PRC_filename = 'PRC' + '_' + \
                   str(c) + \
                   str(Degree) + '_' + \
                   str(Gamma) + '_' + \
                   strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
     
##    plt.savefig(
##                os.path.join(result_dir, PRC_filename),
##                format='png',
##                dpi=600
##                )
    
    plt.show()
    plt.close()
    
# ----------------------------------------------------------------------------------
# model hyper parameters
# ----------------------------------------------------------------------------------
if __name__ == '__main__':
    
    c            = 150
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
    
    kernel_function = 'poly'
    test_names = ['B_122_4_5', 'B_122_3', 'B_127_1', 'B_127_2']

    proj_dir   = r'\\10.39.42.102\temp\JAAH'
    result_dir = r'\\10.39.42.102\temp\2017_Science_Translational_Medicine\2019'
    DBSI_file  = 'all_data_brain.csv'
    DTI_file   = 'DTI_data_labeled.csv'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------

    print("SVM classification: start...")

    start = timeit.default_timer()

    x_train, y_train, x_test, y_test = sample_strat_manual()

    y_pred, accuracy = SVM_pred()

    cm_list = calculate_cm()

##    plot_cm()

    y_score = SVM_ROC()

##    DNN_ROC()
##    DNN_PRC()

    #print('confusion matrix:\n', cm_norm)
    print(cm_list[4])
    print('SVM C value: ', c)
    print('SVM degree:  ', Degree)
    print('SVM gamma:   ', Gamma)
 
    stop = timeit.default_timer()
    running_seconds = np.around(stop-start, 0)
    print('\nSVM Running Time:', running_seconds, 'seconds')


