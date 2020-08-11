#----------------------------------------------------------------------
# SVM for brain tumor DHI classification
# Patient stratification was used
--------------------------------------------------------------------

import pandas as pd
import glob2 as glob
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import timeit
import scipy.stats
import seaborn as sn
import string
from matplotlib.pylab import *
from itertools import combinations
from time import strftime, gmtime
import warnings

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
import gc

# ----------------------------------------------------------------------------------
# sample stratification
# ----------------------------------------------------------------------------------
j       = 7000
deg     = 4
counter = 0
for i in [500]:
    k = 0
    l = 'poly'
    m = 0.5
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
        tpi_num = int(np.ceil(len(unique_names) * test_frac))
        tpi = np.random.choice(len(unique_names), size=tpi_num, replace=False)
        test_names  = []
        train_names = []
        df_test  = pd.DataFrame({'A' : []})
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

        return df_train, df_test

    # def patient_strat_loop(df, iter_num):
        
    #     tpi = total_comb[iter_num]
    #     print(tpi)
    #     unique_names = unique(df.iloc[:, column]) #this gives set of unique samples
    #     test_names = []
    #     train_names = []
    #     df_test = pd.DataFrame({'A' : []})
    #     df_train = pd.DataFrame({'A' : []})
        
    #     for index in tpi:
    #         test_names.append(unique_names[index])
    #         if df_test.empty:
    #             df_test = df[df.iloc[:, column] == unique_names[index]]
    #         else:
    #             df_test = df_test.append(df[df.iloc[:, column] == unique_names[index]])
                
    #     for index in range(len(unique_names)):
    #         if index not in tpi:
    #             train_names.append(unique_names[index])
    #             if df_train.empty:
    #                 df_train = df[df.iloc[:, column] == unique_names[index]]
    #             else:
    #                 df_train = df_train.append(df[df.iloc[:, column] == unique_names[index]])
    #     #print(df_test.shape)
    #     #print(df_train.shape)
    #     return df_train, df_test

    def patient_strat_loop(df, iter_num): #this version splits by patient, not by sample
        
        tpi = total_comb[iter_num]
        print(tpi)
        unique_samples = unique(df.iloc[:, column])
        unique_names = ['B_92','B_94','B_95','B_97','B_120','B_121','B_122','B_123','B_124','B_125','B_126','B_127','B_128']
        test_names = []
        train_names = []
        df_test = pd.DataFrame({'A' : []})
        df_train = pd.DataFrame({'A' : []})
        
        for index in tpi: #index of patient
            for index_sample in range(len(unique_samples)):
                #print(unique_names[index] + ":sample:" + unique_samples[index_sample])
                if unique_samples[index_sample].find(unique_names[index]) != -1:
                    #print(unique_names[index] + ":sample:" + unique_samples[index_sample])
                    test_names.append(unique_samples[index_sample])
                    if df_test.empty:
                        df_test = df[df.iloc[:, column] == unique_samples[index_sample]]
                    else:
                        df_test = df_test.append(df[df.iloc[:, column] == unique_samples[index_sample]])
                
        for index_sample in range(len(unique_samples)):
            if unique_samples[index_sample] not in test_names:
                train_names.append(unique_samples[index_sample])
                if df_train.empty:
                    df_train = df[df.iloc[:, column] == unique_samples[index_sample]]
                else:
                    df_train = df_train.append(df[df.iloc[:, column] == unique_samples[index_sample]])
        #print("test samples: " + str(test_names))
        #print("train samples" + str(train_names))
        return df_train, df_test

    def data_balancing(df, class0_ratio, class1_ratio, class2_ratio):
        
        class0 = df[df.iloc[:,0] == 1]
        class0_sample = class0.sample(int(class0.shape[0]*class0_ratio))
        
        class1 = df[df.iloc[:,0] == 2]
        class1_sample = class1.sample(int(class1.shape[0]*class1_ratio))
        
        class2 = df[df.iloc[:,0] == 3]
        class2_sample = class2.sample(int(class2.shape[0]*class2_ratio))

        df_2 = pd.concat(
                        [class0_sample,
                        class1_sample,
                        class2_sample,]
                        )

        return df_2

    # ----------------------------------------------------------------------------------
    # load data
    # ----------------------------------------------------------------------------------

    def GBM_data(project_dir):

        df = pd.DataFrame()

        data = pd.read_csv(project_dir)
        #data = data_balancing(data, 0.5, 1, 1)

        df = df.append(data)

        train, test = patient_strat_loop(df, select_iter[k])

        x_train = train.iloc[:, x_range]
        y_train = train.iloc[:, y_range]

        x_test  = test.iloc[:, x_range]
        y_test  = test.iloc[:, y_range]

        return x_train, y_train, x_test, y_test 

    # ----------------------------------------------------------------------------------
    # SVM model
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
                        random_state=random_state,
                        shrinking=True,
                        tol=0.001,
                        verbose=False
                        ).fit(x_train, y_train)

        svm_pred = svm_model.predict(x_test)
        accuracy = svm_model.score(x_test, y_test)
        cm       = confusion_matrix(y_test, svm_pred)  
        cm_norm  = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm  = np.around(cm_norm, 2)

        return cm, cm_norm, accuracy

    # ----------------------------------------------------------------------------------
    # calculate statistics
    # ----------------------------------------------------------------------------------

    def statistics(cm):

        ACC_1   = cm[0, 0] / (cm[0, 0] + cm[0, 1] + cm[0, 2])
        ACC_2   = cm[1, 1] / (cm[1, 0] + cm[1, 1] + cm[1, 2])
        ACC_3   = cm[2, 2] / (cm[2, 0] + cm[2, 1] + cm[2, 2])
        ACC_all = accuracy
        FP      = cm[:].sum(axis=0) - np.diag(cm[:])
        FN      = cm[:].sum(axis=1) - np.diag(cm[:])
        TP      = np.diag(cm[:])
        TN      = cm[:].sum() - (FP + FN + TP)
        ACC     = (TP + TN) / (TP + FP + FN + TN)       
        TPR     = TP / (TP + FN)
        TNR     = TN / (TN + FP)       
        PPV     = TP / (TP + FP)    
        NPV     = TN / (TN + FN)
        FPR     = FP / (FP + TN)
        FNR     = FN / (TP + FN)
        FDR     = FP / (TP + FP)
        F1      = 2 * (PPV * TPR) / (PPV + TPR)

        stat_list = [
                    ACC_1,
                    ACC_2,
                    ACC_3,
                    ACC_all,
                    TPR[0],
                    TPR[1],
                    TPR[2],
                    TNR[0],
                    TNR[1],
                    TNR[2],
                    PPV[0],
                    PPV[1],
                    PPV[2],
                    F1[0],
                    F1[1],
                    F1[2]
                    ]

        return stat_list

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
                                            random_state=random_state,
                                            shrinking=True,
                                            tol=0.001,
                                            verbose=False
                                            )
                                        ).fit(x_train, y_train)

        y_score  = SVM_model.decision_function(x_test)
        y_pred   = SVM_model.predict(x_test)
        accuracy = SVM_model.score(x_test, y_test)

        print(len(y_score.shape))
        print(y_pred.shape)
        print(accuracy.shape)
        if len(y_score.shape) < 2:
            return 'NAN'

        # Compute ROC curve and AUC for each class
        fpr       = dict()
        tpr       = dict() 
        threshold = dict()
        roc_auc   = []

        for i in range(3):
                    
            fpr[i], tpr[i], threshold[i] = roc_curve(y_test, y_score[:, i], pos_label = i+1)
            
            roc_auc.append(auc(fpr[i], tpr[i]))

        stat_list2 = [roc_auc[0], roc_auc[1], roc_auc[2]]

        return stat_list2

    # ----------------------------------------------------------------------------------
    # calculate mean and 95% CI
    # ----------------------------------------------------------------------------------

    def mean_CI(data, confidence=0.95):
        
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence)/2.0, n-1)
        
        return m, m-h, m+h

    def stat_summary():
        
        ACC_1_stat   = mean_CI(data.iloc[:, 0])
        ACC_2_stat   = mean_CI(data.iloc[:, 1])
        ACC_3_stat   = mean_CI(data.iloc[:, 2])
        ACC_all_stat = mean_CI(data.iloc[:, 3])

        TPR_1_stat   = mean_CI(data.iloc[:, 4])
        TPR_2_stat   = mean_CI(data.iloc[:, 5])
        TPR_3_stat   = mean_CI(data.iloc[:, 6])

        TNR_1_stat   = mean_CI(data.iloc[:, 7])
        TNR_2_stat   = mean_CI(data.iloc[:, 8])
        TNR_3_stat   = mean_CI(data.iloc[:, 9])

        PPV_1_stat   = mean_CI(data.iloc[:, 10])
        PPV_2_stat   = mean_CI(data.iloc[:, 11])
        PPV_3_stat   = mean_CI(data.iloc[:, 12])

        F1_1_stat    = mean_CI(data.iloc[:, 13])
        F1_2_stat    = mean_CI(data.iloc[:, 14])
        F1_3_stat    = mean_CI(data.iloc[:, 15])

        AUC_1_stat   = mean_CI(data.iloc[:, 16])
        AUC_2_stat   = mean_CI(data.iloc[:, 17])
        AUC_3_stat   = mean_CI(data.iloc[:, 18])

        for i in range(len(ACC_1_stat)):
            sum_stat.append(ACC_1_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(ACC_2_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(ACC_3_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(ACC_all_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(TPR_1_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(TPR_2_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(TPR_3_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(TNR_1_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(TNR_2_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(TNR_3_stat[i])

        for i in range(len(ACC_1_stat)):
            sum_stat.append(PPV_1_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(PPV_2_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(PPV_3_stat[i])

        for i in range(len(ACC_1_stat)):
            sum_stat.append(F1_1_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(F1_2_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(F1_3_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(AUC_1_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(AUC_2_stat[i])
        
        for i in range(len(ACC_1_stat)):
            sum_stat.append(AUC_3_stat[i])

        return sum_stat

    # ----------------------------------------------------------------------------------
    # stat report
    # ----------------------------------------------------------------------------------

    def stat_report():

        sum_stat = stat_summary()

        sum_stat = np.around(sum_stat, 3)

        sum_stat = np.reshape(sum_stat, (19, 3))

        sum_stat_df = pd.DataFrame(
                                sum_stat,
                                index=[
                                        'ACC_1',
                                        'ACC_2',
                                        'ACC_3',
                                        'ACC_ALL',
                                        'TPR_1',
                                        'TPR_2',
                                        'TPR_3',
                                        'TNR_1',
                                        'TNR_2',
                                        'TNR_3',
                                        'PPV_1',
                                        'PPV_2',
                                        'PPV_3',
                                        'F1_1',
                                        'F1_2',
                                        'F1_3',
                                        'AUC_1',
                                        'AUC_2',
                                        'AUC_3'
                                        ],                            
                                columns=[
                                            'mean',
                                            '95% CI -',
                                            '95% CI +'
                                            ]
                                )

        stat_data = 'stat_data' + str(strftime("%d-%b-%Y-%H-%M-%S", gmtime())) + '.csv'

        sum_stat_df.to_csv(os.path.join(result_dir, stat_data))

        return sum_stat_df

    # ----------------------------------------------------------------------------------
    # model hyper parameters
    # ----------------------------------------------------------------------------------

    if __name__ == '__main__':

        loop_count   = i
        c            = j
        Degree       = deg
        Gamma        = m
        random_state = 42
        sum_stat     = []
        all_stat     = []
        all_stat_df  = []
        good_samples = 0
        column_DTI   = 3
        column_DBSI  = 4
        x_range_DTI  = [0, 1]
        y_range_DTI  = 2
        x_range_DBSI = [1, 2, 3]
        y_range_DBSI = [0]

        #total_comb = list(combinations(np.arange(21), 4)) #this is total comb of samples
        total_comb = list(combinations(np.arange(13), 4)) #this is total comb of patients
        #print(total_comb[0])

        kernel_function = l

        DBSI_dir    = r'Y:\JAAH\all_data_brain.csv'
        DTI_dir     = r'Y:\JAAH\DTI_data_labeled.csv'  
        result_dir  = r'X:\Zezhong_Ye\2017_Science_Translational_Medicine\2019_yzz'
        
        project_dir = DTI_dir
        column       = column_DTI
        x_range      = x_range_DTI
        y_range      = y_range_DTI

        # ----------------------------------------------------------------------------------
        # run the model
        # ----------------------------------------------------------------------------------

        print("SVM classification: start...")

        warnings.filterwarnings("ignore")

        start = timeit.default_timer()

        #total_comb = list(combinations(np.arange(21), 4)) #this is total comb of samples
        total_comb = list(combinations(np.arange(13), 4)) #this is total comb of patients

        select_iter = tpi = np.random.choice(715, size=len(total_comb), replace=False)
        #print(len(total_comb))
        while good_samples < loop_count and k < 715:
   
            x_train, y_train, x_test, y_test = GBM_data(project_dir)

            cm, cm_norm, accuracy = SVM_prediction(c, Degree, Gamma, kernel_function)

            if np.any(np.isnan(np.array(cm))):
                #print('true')
                k = k + 1
                continue
        
            if cm.shape[0] != 3 or cm.shape[1] != 3:
                #print('true')
                k = k + 1
                continue

            stat_list = statistics(cm)
                    
            stat_list2 = SVM_ROC(c, Degree, Gamma, kernel_function)

            if stat_list2 == 'NAN':
                k = k + 1
                continue

            print(k)

            stat_list.extend(stat_list2)

            stat_list = [round(item, 3) for item in stat_list]

            breaker = False

            if np.any(np.isnan(np.array(stat_list))):
                k = k + 1
                print('true')
                continue
            #print("seg")

            all_stat.append(stat_list)
            
            good_samples = good_samples + 1
            k = k + 1
            
            print('Test added. Total proper tests:', good_samples)

        all_stat_df = pd.DataFrame(all_stat)

        data = all_stat_df
        
        sum_stat_df = stat_report()

        print(sum_stat_df)
        print('SVM C value:    ', c)
        print('SVM degree:     ', Degree)
        print('SVM gamma:      ', Gamma)
        print('Function Type:  ', kernel_function)
        print('epoch:          ', loop_count)
        print('effecive tests: ', good_samples)
    
        stop = timeit.default_timer()
        running_seconds = np.around(stop - start, 0)
        running_minutes = np.around(running_seconds/60.0, 2)
        print('\nSVM Running Time:', running_seconds, 'seconds')
        print('\nSVM Running Time:', running_minutes, 'minutes')
        
        report = open(r"Y:\JAAH\SVM_DTI_RESULTS.txt",'a')
        report.write('___________PARAM SET ' + str(counter) + '____________ \n')
        report.write(str(sum_stat_df))
        report.write('\n')
        report.write('SVM C value:    ' + str(c) + ' \n')
        report.write('SVM degree:     ' + str(Degree) + ' \n')
        report.write('SVM gamma:      ' + str(Gamma) + ' \n')
        report.write('Function Type:  ' + str(kernel_function) + ' \n')
        report.write('epoch:          ' + str(loop_count) + ' \n')
        report.write('effecive tests: ' + str(good_samples) + ' \n')
        report.write('')
        report.close()
        counter = counter + 1
    gc.collect()