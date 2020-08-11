
#------------------------------------------------------------------------------------------
# deep learning classifier for ex classification on ex vivo PCa Gleason grades 
# individual ROC for each class was plotted
# 
# Keras support:
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
import timeit
import seaborn as sn
import pandas as pd
import glob2 as glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
from scipy import interp
from itertools import cycle

import keras
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.wrappers.scikit_learn import KerasClassifier

import tensorflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

start = timeit.default_timer()
print("Deep Neural Network for PCa grade classification: start...")
# ----------------------------------------------------------------------------------
# DNN paramters
# ----------------------------------------------------------------------------------
# key model parameters
learning_rate = 0.00001
batch_momentum = 0.97
batch_size = 100
dropout_rate = 0
n_neurons = 100
epochs = 20
alpha = 0.3
# routine parameters
d = 3
n_inputs = 19
n_outputs = 4
n_classes = 4
test_size = 0.5
random_state = 42
val_test_size = 0.3
# DNN hidden layerss
n_neurons = 100
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

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
print("PCa DNN classification ROC analysis: start...")

# data path for windows system
project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning'
result_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\grading\result'
log_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\grading\log'

# # data path for linux or mac system
# project_dir = '/bmrp092temp/Prostate_Cancer_ex_vivo/Deep_Learning/'
# result_dir = '/bmrp092temp/Prostate_Cancer_ex_vivo/Deep_Learning/result/'
# log_dir = '/bmrp092temp/Prostate_Cancer_ex_vivo/Deep_Learning/log/'

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
def data_loading(filename):
       
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
    df = pd.read_csv(os.path.join(project_dir, filename))
    # df = df[~df['Sub_ID'].str.contains("SH")]

    # define label class
    df.loc[df['ROI_Class'] == 'G1', 'y_cat'] = 0
    df.loc[df['ROI_Class'] == 'G2', 'y_cat'] = 1
    df.loc[df['ROI_Class'] == 'G3', 'y_cat'] = 2
    df.loc[df['ROI_Class'] == 'G5', 'y_cat'] = 3
    df.loc[df['ROI_Class'] == 'G4', 'y_cat'] = 4
    
    # balance train and validation data from each class
    class0 = df[df['y_cat'] == 0]
    class0_sample = class0.sample(int(class0.shape[0]))
    class1 = df[df['y_cat'] == 1]
    class1_sample = class1.sample(int(class1.shape[0]))
    class2 = df[df['y_cat'] == 2]
    class2_sample = class2.sample(int(class2.shape[0]*0.7))
    class3 = df[df['y_cat'] == 3]
    class3_sample = class3.sample(int(class3.shape[0]*0.12))

    # reconstruct dataset from balanced data
    df_2 = pd.concat([class0_sample, class1_sample, class2_sample, class3_sample])
    # select train, validation and test features 
    X = df_2.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15,
                   16, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    # define train, validation and test labels
    Y = df_2.y_cat.astype('int')
    # binarize the output
    Y_2 = label_binarize(Y, classes=[0, 1, 2, 3])
    # construct train dataset with 70% slpit
    x_train, x_test, y_train, y_test = train_test_split(
                                                        X,
                                                        Y_2,
                                                        test_size=test_size,
                                                        random_state=random_state
                                                        )
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = data_loading('Gleason.csv')

print("loading data from csv file: complete!!!")
print("deep neuronetwork construction: start...")

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# kernal initializer function: "he_uniform", "lecun_normal", "lecun_uniform",
#                              "glorot_uniform", "glorot_normal";
# ----------------------------------------------------------------------------------
def Keras_model():
    
    model = Sequential()

    dense_layer = partial(
                          Dense,
                          init='he_normal', 
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
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 1
    model.add(dense_layer(n_hidden1))
    model.add(batch_normalization())
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 2
    model.add(dense_layer(n_hidden2))
    model.add(batch_normalization())
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 3
    model.add(dense_layer(n_hidden3))
    model.add(batch_normalization())
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 4
    model.add(dense_layer(n_hidden4))
    model.add(batch_normalization())
    #model.add(ELU(alpha=1.0))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 5
    model.add(dense_layer(n_hidden5))
    model.add(batch_normalization())
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 6
    model.add(dense_layer(n_hidden6))
    model.add(batch_normalization())
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 7
    model.add(dense_layer(n_hidden7))
    model.add(batch_normalization())
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 8
    model.add(dense_layer(n_hidden8))
    model.add(batch_normalization())
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # hidden layer 9
    model.add(dense_layer(n_hidden9))
    model.add(batch_normalization())
    #model.add(ELU(alpha=1.0))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))
              
    # hidden layer 10
    model.add(dense_layer(n_hidden10))
    model.add(batch_normalization())
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(dropout_rate))

    # output layer
    model.add(dense_layer(n_outputs))
    model.add(batch_normalization())
    model.add(Activation('softmax'))

    model.summary()

    model.compile(
                  loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    
    return model

# ----------------------------------------------------------------------------------
# trainning DNN model
# ----------------------------------------------------------------------------------
def DNN_model_training():
    
    DNN_model = Keras_model()
    
    history = DNN_model.fit(
                            x=x_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            callbacks=None,
                            validation_split=0.2,
                            shuffle=True,
                            class_weight=None,
                            sample_weight=None,
                            initial_epoch=0,
                            steps_per_epoch=None,
                            validation_steps=None,            
                            )

    score = DNN_model.evaluate(
                               x_test,
                               y_test,
                               verbose=0
                               )
    
    y_pred = DNN_model.predict(x_test)
    y_pred_label = np.argmax(y_pred, axis=1)

    return score, y_pred, y_pred_label

score, y_pred, y_pred_label = DNN_model_training()
print('\noverall test loss:', np.around(score[0], d))
print('overall test accuracy:', np.around(score[1], d))

# ----------------------------------------------------------------------------------
# plot summed ROC curves in one figure
# ----------------------------------------------------------------------------------
fpr = dict()
tpr = dict()
roc_auc = dict()
threshold = dict()

for i in range(n_classes):
    fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())                                                                          
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
# Finally average it and compute AUC
mean_tpr /= n_classes
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
#plt.title('ROC', fontweight='bold', fontsize=14)
##plt.plot(
##         fpr["micro"],
##         tpr["micro"],
##         label='micro-average ROC curve (area = {0:0.2f})'
##               ''.format(roc_auc["micro"]),
##         color='deeppink',
##         linestyle=':',
##         linewidth=3
##)
##plt.plot(
##         fpr["macro"],
##         tpr["macro"],
##         label='macro-average ROC curve (area = {0:0.2f})'
##               ''.format(roc_auc["macro"]),
##         color='navy',
##         linestyle=':',
##         linewidth=3
##)
colors = cycle(['aqua', 'red', 'purple', 'royalblue'])

for i, color in zip(range(n_classes), colors):
    plt.plot(
             fpr[i],
             tpr[i],
             color=color,
             linewidth=3,
             label='Class {0} (AUC {1:0.2f})'
             ''.format(i+1, roc_auc[i])
             )

#plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
plt.xlim([-0.03, 1])
plt.ylim([0, 1.03])
ax.axhline(y=0, color='k', linewidth=4)
ax.axhline(y=1.03, color='k', linewidth=4)
ax.axvline(x=-0.03, color='k', linewidth=4)
ax.axvline(x=1, color='k', linewidth=4) 
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=15)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=15)
plt.legend(loc='lower right', prop={'size': 13, 'weight': 'bold'}) 
plt.grid(True)
plt.show()
plt.savefig(os.path.join(result_dir, 'ROC_sum_1' + '.png'), format='png', dpi=600)
plt.close()

# ----------------------------------------------------------------------------------
# plot summed ROC curves zoom in view
# ----------------------------------------------------------------------------------

# Zoom in view of the upper left corner.
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')
plt.grid(True)
plt.plot(
         fpr["micro"],
         tpr["micro"],
         label='Micro Avg (AUC {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='orange',
         linestyle=':',
         linewidth=3
         )

plt.plot(
         fpr["macro"],
         tpr["macro"],
         label='Macro Avg (AUC {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy',
         linestyle=':',
         linewidth=3
         )

colors = cycle(['aqua', 'red', 'purple', 'royalblue'])

for i, color in zip(range(n_classes), colors):
    plt.plot(
             fpr[i],
             tpr[i],
             color=color,
             linewidth=3,
             #label='Class {0} (area = {1:0.2f})'
             #''.format(i+1, roc_auc[i])
             )
    
#plt.title('ROC', fontweight='bold', fontsize=14)
#plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
plt.xlim(0, 0.41)
plt.ylim(0.6, 1.01)
ax.axhline(y=0.6, color='k', linewidth=4)
ax.axhline(y=1.01, color='k', linewidth=4)
ax.axvline(x=0, color='k', linewidth=4)
ax.axvline(x=0.41, color='k', linewidth=4) 
plt.xticks([0, 0.1, 0.2, 0.3, 0.4], fontsize=14, fontweight='bold')
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate',  fontweight='bold', fontsize=15)
plt.ylabel('True Positive Rate',  fontweight='bold', fontsize=15)
plt.legend(loc='lower right', prop={'size': 13, 'weight':'bold'})
plt.show()
plt.savefig(os.path.join(result_dir, 'ROC_sum_2' + '.png'), format='png', dpi=600)
plt.close()

# ----------------------------------------------------------------------------------
# plot individual ROC curve
# ----------------------------------------------------------------------------------
#y_prob = _keras_model.predict_proba(x_test)[:, 1]

fpr = dict()
tpr = dict()
roc_auc = dict()
threshold = dict()

for i in range(n_classes):
    fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i],
                                             y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    #print("AUC:", np.around(roc_auc[i], 3))
    print('Class {0}: AUC = {1:0.2f}'
             ''.format(i+1, roc_auc[i]))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.plot(
             fpr[i],
             tpr[i],
             color='royalblue',
             linewidth=3,
             label='AUC = %0.3f'%roc_auc[i]
             )
    plt.legend(loc='lower right', prop={'size': 13, 'weight':'bold'})
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlim([-0.03, 1.0])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=3)
    ax.axhline(y=1.03, color='k', linewidth=3)
    ax.axvline(x=-0.03, color='k', linewidth=3)
    ax.axvline(x=1, color='k', linewidth=3)
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
    ax.tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5)                                                                                     
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)      
    plt.grid(True)
    #plt.show()
    #plt.savefig(os.path.join(result_dir, lambda x: 'ROC_'+str(x+1)+'.png', format='png', dpi=600))
    plt.savefig(os.path.join(result_dir, 'ROC_' + str(i+1) + '.png'), format='png', dpi=600)
    plt.close()

# ----------------------------------------------------------------------------------
# print key DNN model parameters
# ----------------------------------------------------------------------------------
print("key model parameters:")
print("train dataset number:", len(x_train))
print("validation dataset number:", len(x_val))
print("test dataset number:", len(x_test))
print("epochs:", epochs)
print("batch size:", batch_size)
print("dropout rate:", dropout_rate)
print("batch momentum:", batch_momentum)
print("initial learning rate:", learning_rate)
print("hidden layer neuron numbers:", n_neurons )
# calcualte running time
stop = timeit.default_timer()
print('Running Time:', np.around(stop - start, 0), 'seconds')
print("Deep learning model for PCa Gleason grades classificaiton: complete!!!")







































