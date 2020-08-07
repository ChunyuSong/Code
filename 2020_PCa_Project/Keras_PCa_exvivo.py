#------------------------------------------------------------------------------------------
# DNN model based on Keras
# batch normalization layers were used
# dropout layers were used
# Author: Zezhong Ye;
# Date: 03.29.2019
#
# :to be implemented:
#     - image embeddings (as in https://www.tensorflow.org/get_started/embedding_viz)
#     - ROC curve calculation (as in http://blog.csdn.net/mao_feng/article/details/54731098)
#--------------------------------------------------------------------------------------------
import os
import seaborn as sn
import pandas as pd
import glob2 as glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import timeit

import keras
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import ELU, LeakyReLU

import tensorflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix

start = timeit.default_timer()
print("Deep Neural Network for PCa grade classification: start...")
# ----------------------------------------------------------------------------------
# DNN paramters
# ----------------------------------------------------------------------------------
# key model parameters
learning_rate = 0.0001
batch_momentum = 0.97
batch_size = 200
dropout_rate = 0.2
n_neurons = 100
epochs = 1
# DNN hidden layerss
n_hidden1 = n_neurons
n_hidden2 = n_neurons
n_hidden3 = n_neurons
n_hidden4 = n_neurons
n_hidden5 = n_neurons
n_hidden6 = n_neurons
n_hidden7 = n_neurons
n_hidden8 = n_neurons
n_hidden9 = n_neurons
n_hidden10 = n_neurons
# routine parameters
d = 3
n_inputs = 18
n_outputs = 2
optimizer = 'adam'
test_size = 0.5
random_state = 42
val_test_size = 0.2
display_step = 10

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

df = pd.read_csv(os.path.join(project_dir, 'PCa.csv'))
# df = df[~df['Sub_ID'].str.contains("SH")]

df.loc[df['ROI_Class'] == 'PCa', 'y_cat'] = 1
df.loc[df['ROI_Class'] == 'BPZ', 'y_cat'] = 0  # BPH, BPZ, SBPH

# balance data from each class
class1 = df[df['y_cat'] == 1]
class1_sample = class1.sample(int(class1.shape[0]))
class0 = df[df['y_cat'] == 0]
class0_sample = class0.sample(int(class0.shape[0]))

# cnostruct new dataset with selecting class
df_2 = pd.concat([class0_sample, class1_sample])
# select train, validation and test features 
x = df_2.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 28, 29]]
# define train, validation and test labels
y = df_2.y_cat.astype('int')

# construct train dataset with 80% slpit
x_train, x_val_test, y_train, y_val_test = train_test_split(
                                                            x,
                                                            y,
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

### convert class vectors to binary class matrices
##y_train = keras.utils.to_categorical(y_train, n_outputs)
##y_test = keras.utils.to_categorical(y_test, n_outputs)

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# ----------------------------------------------------------------------------------
model = Sequential()

dense_layer = partial(
                      keras.layers.Dense,
                      init="he_normal", #    "he_uniform", "lecun_normal", "lecun_uniform", "glorot_uniform", "glorot_normal"
                      use_bias=False,
                      activation=None,
                      )

batch_normalization = partial(
                              keras.layers.BatchNormalization,
                              axis=-1,
                              momentum=batch_momentum,
                              epsilon=0.001,
                              beta_initializer='zeros',
                              gamma_initializer='ones',
                              beta_regularizer=None,
                              gamma_regularizer=None                             
)     
                                    
                                    
# input layer                              
model.add(dense_layer(18, input_dim=n_inputs))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(dropout_rate))

# hidden layer 1
model.add(dense_layer(n_hidden1))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
#model.add(Dropout(dropout_rate))

# hidden layer 2
model.add(dense_layer(n_hidden2))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(dropout_rate))

# hidden layer 3
model.add(dense_layer(n_hidden3))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
#model.add(Dropout(dropout_rate))

# hidden layer 4
model.add(dense_layer(n_hidden4))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(dropout_rate))

# hidden layer 5
model.add(dense_layer(n_hidden5))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
#model.add(Dropout(dropout_rate))

# hidden layer 6
model.add(dense_layer(n_hidden6))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(dropout_rate))

# hidden layer 7
model.add(dense_layer(n_hidden7))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
#model.add(Dropout(dropout_rate))

# hidden layer 8
model.add(dense_layer(n_hidden8))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(dropout_rate))

# hidden layer 9
model.add(dense_layer(n_hidden9))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
#model.add(Dropout(dropout_rate))
          
# hidden layer 10
model.add(dense_layer(n_hidden10))
model.add(batch_normalization())
model.add(ELU(alpha=1.0))
model.add(Dropout(dropout_rate))

# output layer
model.add(dense_layer(n_outputs))
model.add(batch_normalization())
model.add(Activation('softmax'))

model.summary()

model.compile(
              loss='sparse_categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy']
)

history = model.fit(
                    x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2,
                    callbacks=None,
                    validation_split=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None,
                    initial_epoch=0,
                    steps_per_epoch=None,
                    validation_steps=None,            
                    validation_data=(x_val, y_val),
)

score = model.evaluate(x_test, y_test, verbose=0)
print('\noverall test loss:', np.around(score[0], d))
print('overall test accuracy:', np.around(score[1], d))

print("\ntrain set size:", len(x_train))
print("validation set size:", len(x_val))
print("test set size:", len(x_test))

# ----------------------------------------------------------------------------------
# create and plot confusion matrix
# ----------------------------------------------------------------------------------
y_pred = model.predict(x_test)
y_pred_label = np.argmax(y_pred, axis=1)

# calculate confusion matrix
cm = confusion_matrix(y_test, y_pred_label)
print("\nconfusion matrix:")
print(np.around(cm, 2))
# calculate normalized confusion matrix
print("\nnormalized confusion matrix:")
cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
print(np.around(cm_norm, 2))

# ----------------------------------------------------------------------------------
# sensitivity, specificity, precision
# ----------------------------------------------------------------------------------
FP = cm[:].sum(axis=0) - np.diag(cm[:])  
FN = cm[:].sum(axis=1) - np.diag(cm[:])
TP = np.diag(cm[:])
TN = cm[:].sum() - (FP+FN+TP)

# Sensitivity, hit rate, recall, or true positive rate
TPR = TP/(TP+FN)
# Specificity or true negative rate
TNR = TN/(TN+FP) 
# Precision or positive predictive value
PPV = TP/(TP+FP)
# Negative predictive value
NPV = TN/(TN+FN)
# Fall out or false positive rate
FPR = FP/(FP+TN)
# False negative rate
FNR = FN/(TP+FN)
# False discovery rate
FDR = FP/(TP+FP)
# Overall accuracy
ACC = (TP+TN)/(TP+FP+FN+TN)

print("\nsensitivity:", np.around(TPR, 3))
print("specificity:", np.around(TNR, 3))
print("precision:", np.around(PPV, 3))
print("negative predictive value:", np.around(NPV, 3))
print("false positive rate:", np.around(FPR, 3))
print("false negative rate:", np.around(FNR, 3))
print("false discovery rate:", np.around(FDR, 3))

# ----------------------------------------------------------------------------------
# classification report: presicion, recall, f-score
# ----------------------------------------------------------------------------------
print("\nprecision, recall, f1-score, support:")
print(classification_report(y_test, y_pred_label, digits=d))
# calcualte running time
stop = timeit.default_timer()
print('Running Time:', np.around(stop-start, 0), 'seconds')

# ----------------------------------------------------------------------------------
# ROC curve
# ----------------------------------------------------------------------------------
y_prob = model.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print("AUC:", np.around(roc_auc, 3))
# plot ROC curve
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(fpr, tpr, color='royalblue', linewidth=3, label='AUC = %0.3f'%roc_auc)
plt.legend(loc='lower right')
plt.legend(fontsize=16)
legend_properties = {'weight': 'bold'}
plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
plt.xlim([-0.03, 1.0])
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
plt.savefig(os.path.join(result_dir, 'ROC.png'), format='png', dpi=600)
plt.close()

# ----------------------------------------------------------------------------------
# Precision-Recall Curve
# ----------------------------------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
#auc = auc(precision, recall)
#print("AUC:", np.around(auc, 3))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(precision, recall, color='royalblue', linewidth=3, label='AUC = %0.3f'%roc_auc)
#plt.legend(loc='lower right')
#plt.legend(fontsize=16)
#legend_properties = {'weight': 'bold'}
plt.plot([0, 1], [1, 0], 'r--', linewidth=2)
plt.xlim([0, 1.03])
plt.ylim([0, 1.03])
plt.ylabel('Precision', fontweight='bold', fontsize=14)
plt.xlabel('Recall', fontweight='bold', fontsize=14)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
ax.tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5)
ax.axhline(y=0, color='k', linewidth=3)
ax.axhline(y=1.03, color='k', linewidth=3)
ax.axvline(x=0, color='k', linewidth=3)
ax.axvline(x=1.03, color='k', linewidth=3)
plt.grid(True)
ax.set_aspect('equal')
plt.show()
plt.savefig(os.path.join(result_dir, 'precision-racall.png'), format='png', dpi=600)
plt.close()

print("DNN classification ROC analysis: complete!")
  








