
#------------------------------------------------------------------------------------------
# deep learning classifier using sklearn framework
#
# Author: Zezhong Ye;
# Date: 03.30.2019
# ROC support:
#   :scalars:
#     - accuracy
#     - wieghts and biases
#     - cost/cross entropy
#     - dropout
#   :images:
#     - reshaped input
#     - conv layers outputs
#     - conv layers weights visualisation
#   :graph:
#     - full graph of the network
#   :distributions and histograms:
#     - weights and biases
#     - activations
#   :checkpoint saving:
#     - checkpoints/saving model
#     - weights embeddings
#
#   :to be implemented:
#     - image embeddings (as in https://www.tensorflow.org/get_started/embedding_viz)
#     - ROC curve calculation (as in http://blog.csdn.net/mao_feng/article/details/54731098)
#--------------------------------------------------------------------------------------------

import os
import csv
import pandas as pd
import glob2 as glob
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# ----------------------------------------------------------------------------------
# DNN paramters
# ----------------------------------------------------------------------------------
# key model parameters
n_classes = 2
solver = 'adam'                               # optimized funciton: "adam", "SGD"
activation = 'tanh'                           # "relu", "sigmoid", "tanh"
learning_rate = 'adaptive'                    # "constant", "adaptive"
learning_rate_init = 0.001
batch_size = 'auto'                           # "auto"
momentum = 0.97
hidden_layer = (200, 200, 200, 200, 200)
# routine parameters
d = 3
test_size = 0.5
random_state = 42
val_test_size = 0.2
max_iteration = 500

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
print("PCa DNN classification ROC analysis: start...")

# data path for windows system
project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_ex_vivo\Deep_Learning'
result_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Deep_Learning\pca_exvivo_1\result'
log_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Deep_Learning\pca_exvivo_1\log'

# data path for linux or mac system
# project_dir = '/bmrp092temp/Zezhong_Ye/Prostate_Cancer_ex_vivo/Deep_Learning/'
# result_dir = '/bmrp092temp/Zezhong_Ye/Deep_Learning/pca_exvivo_grading/result/'
# log_dir = '/bmrp092temp/Zezhong_Ye/Deep_Learning/pca_exvivo_grading/log/'

if not os.path.exists(result_dir):
    print('result directory does not exist - creating...')
    os.makedirs(result_dir)
    print('log directory created.')
else:
    print('result directory already exists ...')

if not os.path.exists(log_dir):
       print('log directory does not exist - creating...')
       os.makedirs(log_dir)
       os.makedirs(log_dir + '/train')
       os.makedirs(log_dir + '/validation')
       print('log directory created.')
else:
    print('log directory already exists ...')

# ----------------------------------------------------------------------------------
# construct train, validation and test dataset
# ----------------------------------------------------------------------------------
maps_list = [
             'b0_map.nii',                       #07
             'dti_adc_map.nii',                  #08
             'dti_axial_map.nii',                #09
             'dti_fa_map.nii',                   #10
             'dti_radial_map.nii',               #11
             'fiber_ratio_map.nii',              #12
             'fiber1_axial_map.nii',             #13
             'fiber1_fa_map.nii',                #14
             'fiber1_fiber_ratio_map.nii',       #15
             'fiber1_radial_map.nii',            #16
             'fiber2_axial_map.nii',             #17
             'fiber2_fa_map.nii',                #18
             'fiber2_fiber_ratio_map.nii',       #19
             'fiber2_radial_map.nii',            #20
             'hindered_ratio_map.nii',           #21
             'hindered_adc_map.nii',             #22
             'iso_adc_map.nii',                  #23
             'restricted_adc_1_map.nii',         #24
             'restricted_adc_2_map.nii',         #25
             'restricted_ratio_1_map.nii',       #26
             'restricted_ratio_2_map.nii',       #27
             'water_adc_map.nii',                #28
             'water_ratio_map.nii',              #29
]

# load data from csv files and define x and y
df = pd.read_csv(os.path.join(project_dir, 'PCa.csv'))
# df = df[~df['Sub_ID'].str.contains("SH")]

df.loc[df['ROI_Class'] == 'PCa', 'y_cat'] = 0
df.loc[df['ROI_Class'] == 'BPZ', 'y_cat'] = 1
# df.loc[df['ROI_Class'] == 'BPZ', 'y_cat'] = 1
# df.loc[df['ROI_Class'] == 'SBPH', 'y_cat'] = 1
class1 = df[df['y_cat'] == 1]
class1_sample = class1.sample(int(class1.shape[0]))
class0 = df[df['y_cat'] == 0]
class0_sample = class0.sample(int(class0.shape[0]*0.6))
# class1 = df[df['y_cat'] == 1]
# class1_sample = class1.sample(int(class1.shape[0]*0.8))
# class1 = df[df['y_cat'] == 2]
# class1_sample = class1.sample(int(class1.shape[0]*0.849))
df_2 = pd.concat([class0_sample, class1_sample])
# df_2 = pd.concat([class0_sample, class1_sample, class2_sample])
# select train, validation and test features 
X = df_2.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 28, 29]]
# define train, validation and test labels
Y = df_2.y_cat.astype('int')
# normalize data using standard scalar 
scaler = StandardScaler()
x = scaler.fit_transform(X.astype(np.float64))

# construct train dataset with 70% slpit
x_train, x_val_test, y_train, y_val_test = train_test_split(
                                                            x,
                                                            Y,
                                                            test_size=val_test_size,
                                                            random_state=random_state
)

# construct validation and test dataset with 50% split
x_val, x_test, y_val, y_test = train_test_split(
                                                x_val_test,
                                                y_val_test,
                                                test_size=test_size,
                                                random_state=random_state
)

train_size = len(x_train)
print("data loading: complete!")
print("train set size:", len(x_train))
print("validation set size:", len(x_val))
print("test set size:", len(x_test))
print("loading data from csv file: complete!!!")
print("deep neuronetwork construction: start...")

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# ----------------------------------------------------------------------------------
DNN_model = MLPClassifier(
                          activation=activation,
                          hidden_layer_sizes=hidden_layer,
                          beta_1=0.9,
                          beta_2=0.999,
                          alpha=0.0001,
                          batch_size=batch_size,
                          early_stopping=False,
                          epsilon=1e-08,
                          learning_rate=learning_rate,
                          learning_rate_init=learning_rate_init,
                          max_iter=max_iteration,
                          momentum=momentum,
                          nesterovs_momentum=True,
                          power_t=0.5,
                          random_state=None,
                          shuffle=True,
                          solver=solver,
                          tol=0.0001,
                          validation_fraction=0.1,
                          verbose=False,
                          warm_start=False
)
                                     
DNN_model.fit(x_train, y_train)
y_pred = DNN_model.predict(x_test).ravel()
y_prob = DNN_model.predict_proba(x_test)[:, 1]

print("training a DNN classifier: complete!")

# ----------------------------------------------------------------------------------
# create and plot a ROC curve
# ----------------------------------------------------------------------------------
print("plotting ROC curves: start...")

fpr, tpr, threshold = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(np.around(roc_auc, 3))

# plot ROC curve
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#plt.title('ROC', fontsize=14, fontweight='bold')
plt.plot(fpr, tpr, color='royalblue', linewidth=3, label='AUC = %0.3f'%roc_auc)
plt.legend(loc='lower right')
plt.legend(fontsize=14)
legend_properties = {'weight': 'bold'}
plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
plt.xlim([-0.03, 1])
plt.ylim([0, 1.03])
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
ax.tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5)
ax.axhline(y=0, color='k', linewidth=3)
ax.axhline(y=1.03, color='k', linewidth=3)
ax.axvline(x=-0.03, color='k', linewidth=3)
ax.axvline(x=1, color='k', linewidth=3)
plt.grid(True)
ax.set_aspect('equal')
plt.show()
#plt.savefig(os.path.join(result_dir, 'ROC.png'), format='png', dpi=600)
plt.close()

print("plotting ROC curves: complete!")
print("DNN classification ROC analysis: complete!")

# np.savetxt(project_dir,filename,fmt='%d')

# ----------------------------------------------------------------------------------
# create and plot a precision-recall curve
# ----------------------------------------------------------------------------------

##from sklearn.metrics import precision_recall_curve
##from sklearn.metrics import average_precision_score
##import pandas as pd
##
### For each class
##precision = dict()
##recall = dict()
##average_precision = dict()
###n_classes = Y.shape[1]
##for i in range(n_classes):
##    precision[i], recall[i], _ = precision_recall_curve(Y_train[:, i], y_score[:, i])
##    average_precision[i] = average_precision_score(Y_train[:, i], y_score[:, i])

    
