#----------------------------------------------------------------------
# SVM for brain tumor DHI classification
# Patient stratification was used
#
# Author: Anthony_Wu
#
# Contact: atwu@wustl.edu
# Date: 08-14-2019
#-------------------------------------------------------------------------------------------
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import glob2 as glob
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn

import string
from matplotlib.pylab import *
#import plotly.express as px
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from itertools import combinations
import os


def unique(list1):
    unique_list = []
    for x in list1:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

def patient_stratification(df, test_frac):
    #print(df)
    unique_names = unique(df.iloc[:,4])
    #print(unique_names)
    #tpi = test patient indices 
    tpi_num = int(np.ceil(len(unique_names) * test_frac))
    tpi = np.random.choice(len(unique_names), size = tpi_num, replace = False)
    test_names = []
    train_names = []
    df_test = pd.DataFrame({'A' : []})
    df_train = pd.DataFrame({'A' : []})
    for index in tpi:
        test_names.append(unique_names[index])
        if df_test.empty:
            df_test = df[df.iloc[:,4] == unique_names[index]]
        else:
            df_test = df_test.append(df[df.iloc[:,4] == unique_names[index]])
    for index in range(len(unique_names)):
        if index not in tpi:
            train_names.append(unique_names[index])
            if df_train.empty:
                df_train = df[df.iloc[:,4] == unique_names[index]]
            else:
                df_train = df_train.append(df[df.iloc[:,4] == unique_names[index]])
    #print(df_test.shape)
    #print(df_train.shape)
    return df_train, df_test

def patient_strat_manual(df, test_names):
    train_names = []
    df_test = pd.DataFrame({'A' : []})
    df_train = pd.DataFrame({'A' : []})
    unique_names = unique(df.iloc[:,4])
    for name in unique_names:
        if name not in test_names:
            train_names.append(name)

    for name in test_names:
        if df_test.empty:
            df_test = df[df.iloc[:,4] == name]
        else:
            df_test = df_test.append(df[df.iloc[:,4] == name])

    for name in train_names:
        if df_train.empty:
            df_train = df[df.iloc[:,4] == name]
        else:
            df_train = df_train.append(df[df.iloc[:,4] == name])   
    #print(train_names)
    #print(test_names)
    #print(df_train.shape)
    #print(df_test.shape)
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

# load data from excel files
#path = r'C:\Users\csong01\Desktop\PCA_in_vivo_data_excel_8'
path = r'Z:\Anthony_Wu\Zezhong_assist\JAAH\all_data_brain.csv'
#filenames = glob.glob(path + "/*.xlsx")

df = pd.DataFrame()
# for f in filenames:
#     data = pd.read_excel(f,'Sheet1',header = None)
#     data.iloc[:, 1:] = (data.iloc[:, 1:]) / (data.iloc[:, 1:].max())
#     data = data.iloc[data[data.iloc[:, 0] != 0].index]
#     df = df.append(data)

data = pd.read_csv(path)
#data = data_balancing(data, 0.5, 1, 1)

# data.iloc[:, 7:] = (data.iloc[:, 7:]) / (data.iloc[:, 7:].max())
#data = data.iloc[data[data.iloc[:, 0] != 0].index]
df = df.append(data)

# Do patient stratification, with around 4:1 ratio tts

train, test = patient_strat_manual(df, ['B_122_4_5','B_95_1','B_128','B_94'])

#train, test = patient_stratification(df, 0.2)
#train, test = train_test_split(df, test_size=0.2)

# X -> DHisto and DTI features, y -> labels
X_train = train.iloc[:, [1, 2, 3]]
y_train = train.iloc[:, 0]

X_test = test.iloc[:, [1, 2, 3]]
y_test = test.iloc[:, 0]

# training a SVM classifier
svm_model = SVC(C=100.0, cache_size=1000, class_weight='balanced', coef0=0.0, decision_function_shape=None, degree=5,
                gamma=0.5, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001,
                verbose=False).fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)



class1__122_4_5 = svm_predictions[0:473]
class2__122_4_5 = svm_predictions[473:692]
class3__122_4_5 = svm_predictions[692:843]
class1__95_1 = svm_predictions[843:1349]
class2__95_1 = svm_predictions[1349:1394]
class1__128 = svm_predictions[1394:1786]
class3__94 = svm_predictions[1786:1963]

all_pred = svm_predictions[0: 1963]
all_x    = X_test[0: 1963]
all_y    = y_test[0: 1963]
all_pred = pd.DataFrame(all_pred)
all_x    = pd.DataFrame(all_x)
all_y    = pd.DataFrame(all_y)
all_pred = all_pred.reset_index(drop=True)
all_x    = all_x.reset_index(drop=True)
all_y    = all_y.reset_index(drop=True)
all_     = pd.concat([all_pred, all_x, all_y], axis=1)

B95_pred = svm_predictions[843: 1394]

B95_x    = X_test[843: 1394]
B95_y    = y_test[843: 1394]
B95_pred = pd.DataFrame(B95_pred)
B95_x    = pd.DataFrame(B95_x)
B95_y    = pd.DataFrame(B95_y)
B95_pred = B95_pred.reset_index(drop=True)
B95_x    = B95_x.reset_index(drop=True)
B95_y    = B95_y.reset_index(drop=True)
B95      = pd.concat([B95_pred, B95_x, B95_y], axis=1)

B94_pred = svm_predictions[1786: 1963]
B94_x    = X_test[1786: 1963]
B94_y    = y_test[1786: 1963]
B94_pred = pd.DataFrame(B94_pred)
B94_x    = pd.DataFrame(B94_x)
B94_y    = pd.DataFrame(B94_y)
B94_pred = B94_pred.reset_index(drop=True)
B94_x    = B94_x.reset_index(drop=True)
B94_y    = B94_y.reset_index(drop=True)
B94      = pd.concat([B94_pred, B94_x, B94_y], axis=1)

B128_pred = svm_predictions[1394: 1786]
B128_x    = X_test[1394: 1786]
B128_y    = y_test[1394: 1786]
B128_pred = pd.DataFrame(B128_pred)
B128_x    = pd.DataFrame(B128_x)
B128_y    = pd.DataFrame(B128_y)
B128_pred = B128_pred.reset_index(drop=True)
B128_x    = B128_x.reset_index(drop=True)
B128_y    = B128_y.reset_index(drop=True)
B128      = pd.concat([B128_pred, B128_x, B128_y], axis=1)

B122_pred = svm_predictions[0: 843]
B122_x    = X_test[0: 843]
B122_y    = y_test[0: 843]
B122_pred = pd.DataFrame(B122_pred)
B122_x    = pd.DataFrame(B122_x)
B122_y    = pd.DataFrame(B122_y)
B122_pred = B122_pred.reset_index(drop=True)
B122_x    = B122_x.reset_index(drop=True)
B122_y    = B122_y.reset_index(drop=True)
B122      = pd.concat([B122_pred, B122_x, B122_y], axis=1)


result_dir  = r'Z:\Zezhong_Ye\2017_Science_Translational_Medicine\Prediction_Data'

file_122 = 'B122' + '.csv'
file_128 = 'B128' + '.csv'
file_94  = 'B94' + '.csv'
file_95  = 'B95' + '.csv'
file_all = 'all' + '.csv'

B122.to_csv(os.path.join(result_dir, file_122))
B128.to_csv(os.path.join(result_dir, file_128))
B94.to_csv(os.path.join(result_dir, file_94))
B95.to_csv(os.path.join(result_dir, file_95))
all_.to_csv(os.path.join(result_dir, file_all))



print("B_122_4_5 class 1 prediction: ", class1__122_4_5[class1__122_4_5 == 1].shape[0] \
    /class1__122_4_5.shape[0])
print("B_122_4_5 class 2 prediction: ", class2__122_4_5[class2__122_4_5 == 2].shape[0] \
    /class2__122_4_5.shape[0])
print("B_122_4_5 class 3 prediction: ", class3__122_4_5[class3__122_4_5 == 3].shape[0] \
    /class3__122_4_5.shape[0])
print("B_95_1 class 1 prediction: ", class1__95_1[class1__95_1 == 1].shape[0]/class1__95_1.shape[0])
print("B_95_1 class 2 prediction: ", class2__95_1[class2__95_1 == 2].shape[0]/class2__95_1.shape[0])
print("B_128 class 1 prediction: ", class1__128[class1__128 == 1].shape[0]/class1__128.shape[0])
print("B_94 class 3 prediction: ", class3__94[class3__94 == 3].shape[0]/class3__94.shape[0])

# model accuracy for X_test
accuracy = svm_model.score(X_test, y_test)
#print(confusion_matrix(y_test, svm_predictions))

cm = confusion_matrix(y_test, svm_predictions)
cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.around(cm_norm, 2)

### calculate individual class accuracy
cm = confusion_matrix(y_test, svm_predictions)
accuracy_class_1 = cm[0,0]/(cm[0,0]+cm[0,1]+cm[0,2])
accuracy_class_2 = cm[1,1]/(cm[1,0]+cm[1,1]+cm[1,2])
accuracy_class_3 = cm[2,2]/(cm[2,0]+cm[2,1]+cm[2,2])
overall_accuracy = accuracy

print(accuracy_class_1)
print(accuracy_class_2)
print(accuracy_class_3)
print(overall_accuracy)


