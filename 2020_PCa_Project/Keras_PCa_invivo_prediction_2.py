# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#------------------------------------------------------------------------------------------
# DNN model based on Keras
# batch normalization layers were used
# dropout layers were used
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
import nibabel as nib
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
learning_rate = 0.001
batch_momentum = 0.97
batch_size = 200
dropout_rate = 0
n_neurons = 100
epochs = 5

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
val_test_size = 0.5
display_step = 10

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
print("PCa DNN classification ROC analysis: start...")

# data and results path for windows system
project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\data'
result_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\result'
log_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\log'
pred_dir = r'\\10.39.42.102\temp\PCa_prediction\test_1'

# # data path for linux system
# project_dir = '/bmrp092temp/Prostate_Cancer_Project_Shanghai/PCa_Machine_Learning/PCa_Benign_Classification/data/'
# result_dir = '/bmrp092temp/Prostate_Cancer_Project_Shanghai/PCa_Machine_Learning/PCa_Benign_Classification/data/'
# log_dir = '/bmrp092temp/Prostate_Cancer_Project_Shanghai/PCa_Machine_Learning/PCa_Benign_Classification/data/'

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

# construct training dataset dataset
df_1 = pd.read_csv(os.path.join(project_dir, 'benign_mpMRI.csv'))
df_2 = pd.read_csv(os.path.join(project_dir, 'PCa_train.csv'))
df_3 = df_1.append(df_2)

df_3.loc[df_3['ROI_Class'] == 't', 'y_cat'] = 1
df_3.loc[df_3['ROI_Class'].isin(['p', 'c']), 'y_cat'] = 0

class0_train = df_3[df_3['y_cat'] == 0]
class0_train_sample = class0_train.sample(int(class0_train.shape[0]))


class1_train = df_3[df_3['y_cat'] == 1]
class1_train_sample = class1_train.sample(int(class1_train.shape[0]))

df_4 = pd.concat([class0_train_sample, class1_train_sample])
df_4 = df_4.sample(frac=1).reset_index(drop=True)

x_train = df_4.iloc[:, [8, 9, 10, 11, 12, 13, 14, 15, 16,
                        21, 22, 23, 24, 25, 26, 27, 28, 29]]

y_train = df_4.y_cat.astype('int')

# construct validation and testing dataset
df_5 = pd.read_csv(os.path.join(project_dir, 'benign_biopsy.csv'))
df_6 = pd.read_csv(os.path.join(project_dir, 'PCa_test.csv'))
df_7 = df_5.append(df_6)

df_7.loc[df_7['ROI_Class'] == 't', 'y_cat'] = 1
df_7.loc[df_7['ROI_Class'].isin(['p', 'c']), 'y_cat'] = 0
# c = transition zone, p = peripheral zone

class0_test = df_7[df_7['y_cat'] == 0]
class0_test_sample = class0_test.sample(int(class0_test.shape[0]))

class1_test = df_7[df_7['y_cat'] == 1]
class1_test_sample = class1_test.sample(int(class1_test.shape[0]))

df_8 = pd.concat([class0_test_sample, class1_test_sample])
df_8 = df_8.sample(frac=1).reset_index(drop=True)

x_val_test = df_8.iloc[:, [8, 9, 10, 11, 12, 13, 14, 15, 16,
                           21, 22, 23, 24, 25, 26, 27, 28, 29]]

y_val_test = df_8.y_cat.astype('int')

# construct validation set and test set with 1:1 ratio
x_val, x_test, y_val, y_test = train_test_split(
                                                x_val_test,
                                                y_val_test,
                                                test_size=test_size,
                                                random_state=random_state
                                                )

# construct prediction dataset
df_pred = pd.read_csv(os.path.join(pred_dir, 'PCa.csv'))

x_pred = df_pred.iloc[:, [8, 9, 10, 11, 12, 13, 14, 15, 16,
                          21, 22, 23, 24, 25, 26, 27, 28, 29]]

print("data loading: complete!")

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# "he_uniform", "lecun_normal", "lecun_uniform", "glorot_uniform", "glorot_normal"
# ----------------------------------------------------------------------------------
print("deep neuronetwork construction: start...")

model = Sequential()

dense_layer = partial(
                      Dense,
                      kernel_initializer="he_normal", 
                      use_bias=False,
                      activation=None,
                      )

batch_normalization = partial(
                              BatchNormalization,
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
                    validation_data=(x_val, y_val)
                    )

score = model.evaluate(x_test, y_test, verbose=0)
print('\noverall test loss:', np.around(score[0], d))
print('overall test accuracy:', np.around(score[1], d))

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

# ----------------------------------------------------------------------------------
# create prediction
# ----------------------------------------------------------------------------------
y_prediction = model.predict(x_pred)
y_prediction_label = np.argmax(y_prediction, axis=1)

x_index = np.asarray(df_pred.iloc[:, [1]])[:,0]
y_index = np.asarray(df_pred.iloc[:, [2]])[:,0]
z_index = np.asarray(df_pred.iloc[:, [3]])[:,0]

img = np.zeros(shape=(280, 224, 24))
#img = np.zeros(shape=(200, 200, 20))


for i in range(x_index.shape[0]):
    img[x_index[i], y_index[i], z_index[i]] = y_prediction_label[i]
	
aff = nib.load(os.path.join(pred_dir, 'DBSI_results_0.1_0.1_0.8_0.8_2.3_2.3', 'b0_map.nii')).get_affine()
data = nib.Nifti1Image(img, aff)
nib.save(data, os.path.join(pred_dir, 'prediction_map.nii'))

# ----------------------------------------------------------------------------------
# print key DNN model parameters
# ----------------------------------------------------------------------------------
print("train dataset number:", len(x_train))
print("validation dataset number:", len(x_val))
print("test dataset number:", len(x_test))
print("prediction dataset number:", len(x_pred))

print("epochs:", epochs)
print("batch size:", batch_size)
print("dropout rate:", dropout_rate)
print("batch momentum:", batch_momentum)
print("initial learning rate:", learning_rate)
print("hidden layer neuron numbers:", n_neurons )
print("Deep learning model for PCa Gleason grades classificaiton: complete!!!")

# calcualte running time
stop = timeit.default_timer()
print('Running Time:', np.around(stop - start, 0), 'seconds')



