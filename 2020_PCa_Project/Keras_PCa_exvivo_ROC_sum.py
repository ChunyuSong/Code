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
import timeit
import seaborn as sn
import pandas as pd
import glob2 as glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial

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
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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
epochs = 10
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
n_inputs = 14
n_outputs = 2
optimizer = 'adam'
test_size = 0.2
random_state = 42
display_step = 10

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
print("PCa DNN classification ROC analysis: start...")

# data path for windows system
project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning'
result_dir = r'\\10.39.42.102\temp\2019_Nature_Medicine\result'
log_dir = r'\\10.39.42.102\temp\2019_Nature_Medicine\log'

# data path for linux or mac system
# project_dir = '/bmrp092temp/Prostate_Cancer_ex_vivo/Deep_Learning/'
# result_dir = '/bmrp092temp/Deep_Learning/pca_exvivo_grading/result/'
# log_dir = '/bmrp092temp/Deep_Learning/pca_exvivo_grading/log/'

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

# ----------------------------------------------------------------------------------
# function to replace y_cat on list ['SBPH','BPZ','BPH','Benign']
# ----------------------------------------------------------------------------------
def replace_model(option):
    df['y_cat'] = 2
    df.loc[df['ROI_Class'] == 'PCa', 'y_cat'] = 1
    if option == 'Benign':
       df.loc[df['ROI_Class'].isin(['SBPH','BPZ','BPH']), 'y_cat'] = 0
    else:
         df.loc[df['ROI_Class'] == option, 'y_cat'] = 0
         
    # balance data from each class
    class1 = df[df['y_cat']==1]
    class1_sample = class1.sample(int(class1.shape[0]))
    class0 = df[df['y_cat']==0]
    class0_sample = class0.sample(int(class0.shape[0]))
    df_2 = pd.concat([class0_sample, class1_sample])
    
    #x = df_2.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 28, 29]]

    x = df_2.iloc[:, [7, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 28, 29]]

    #x = df_2.iloc[:, [7, 8]]

    #x = df_2.iloc[:, [8]]
    
    Y = df_2.y_cat.astype('int')
    
    x_train, x_test, y_train, y_test = train_test_split(
                                                        x,
                                                        Y,
                                                        test_size=test_size,
                                                        random_state=random_state
                                                        )

    return x_train, x_test, y_train, y_test

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# kernal intitializer: "he_uniform", "lecun_normal", "lecun_uniform", "glorot_uniform",
#                      "glorot_normal"
# ----------------------------------------------------------------------------------
print("deep neuronetwork construction: start...")
def Keras_model(init, loss):
    model = Sequential()

    dense_layer = partial(
                          Dense,
                          kernel_initializer=init, 
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
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
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
    model.add(ELU(alpha=1.0))
    #model.add(LeakyReLU(alpha=alpha))
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
                  loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy']
                  )
    
    return model

model = Keras_model(
                    'he_normal',
                    'sparse_categorical_crossentropy'
                    )
# ----------------------------------------------------------------------------------
# function to train DNN on list ['SBPH','BPZ','BPH','Benign']
# ----------------------------------------------------------------------------------
def train_model(x_train, x_test, y_train, y_test):
       
    history = model.fit(
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
                        validation_steps=None            
                        )
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\noverall test loss:', np.around(score[0], d))
    print('overall test accuracy:', np.around(score[1], d))
    return score

# ----------------------------------------------------------------------------------
# 
# ----------------------------------------------------------------------------------

my_list = ['SBPH','BPZ','BPH','Benign']
x_test_list = []
y_test_list = []
score_list = []

for option in my_list:
    x_train, x_test, y_train, y_test = replace_model(option)
    x_test_list.append(x_test)
    y_test_list.append(y_test)
    score_list.append(train_model(x_train, x_test, y_train, y_test))
                   
y_prob_list = [model.predict_proba(test)[:,1] for test in x_test_list ]

# calcualte running time
stop = timeit.default_timer()
print('Running Time:', np.around(stop - start, 0), 'seconds')

# ----------------------------------------------------------------------------------
# draw multiple ROC
# ----------------------------------------------------------------------------------
fpr = dict()
tpr = dict()
threshold = dict()
roc_auc = dict()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')

colors = ['aqua', 'red', 'purple', 'royalblue']
for i, color in zip(range(len(y_prob_list)), colors):
    fpr[i], tpr[i], threshold[i] = roc_curve(y_test_list[i], y_prob_list[i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print("AUC:", np.around(roc_auc[i], 3)) 
    # plot ROC curve
    plt.plot(
             fpr[i],
             tpr[i],
             color=color,
             linewidth=3,
             label='AUC %0.2f'%roc_auc[i]
             )

plt.legend(loc='lower right', prop={'size': 13, 'weight': 'bold'}) 
#plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
plt.xlim([-0.03, 1.0])
plt.ylim([0, 1.03])
plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
ax.axhline(y=0, color='k', linewidth=3)
ax.axhline(y=1.03, color='k', linewidth=3)
ax.axvline(x=-0.03, color='k', linewidth=3)
ax.axvline(x=1, color='k', linewidth=3)

plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
ax.tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5)

plt.grid(True)
plt.show()
plt.savefig(os.path.join(result_dir, 'ROC_exvivo_1.png'), format='png', dpi=600)
plt.close()  

# ----------------------------------------------------------------------------------
# draw multiple ROC zoom in view
# ----------------------------------------------------------------------------------
fpr = dict()
tpr = dict()
threshold = dict()
roc_auc = dict()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_aspect('equal')

colors = ['aqua', 'red', 'purple', 'royalblue']
for i, color in zip(range(len(y_prob_list)), colors):
       fpr[i], tpr[i], threshold[i] = roc_curve(y_test_list[i], y_prob_list[i])
       roc_auc[i] = auc(fpr[i], tpr[i])
       # plot ROC curve
       plt.plot(fpr[i], tpr[i], color=color, linewidth=3, label='AUC %0.2f'%roc_auc[i])

plt.legend(loc='lower right', prop={'size': 13, 'weight': 'bold'})
#plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
plt.xlim([0, 0.41])
plt.ylim([0.6, 1.01])
plt.xticks([0, 0.1, 0.2, 0.3, 0.4], fontsize=14, fontweight='bold')
plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=14, fontweight='bold')
ax.axhline(y=0.6, color='k', linewidth=3)
ax.axhline(y=1.01, color='k', linewidth=3)
ax.axvline(x=0, color='k', linewidth=3)
ax.axvline(x=0.41, color='k', linewidth=3)

plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
plt.xticks(fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
ax.tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5)

plt.grid(True)
plt.show()
plt.savefig(os.path.join(result_dir, 'ROC_exvivo_2.png'), format='png', dpi=600)
plt.close()

# ----------------------------------------------------------------------------------
# print key DNN model parameters
# ----------------------------------------------------------------------------------
print("train dataset number:", len(x_train)*0.8)
print("validation dataset number:", len(x_train)*0.2)
print("test dataset number:", len(x_test))
print("epochs:", epochs)
print("batch size:", batch_size)
print("dropout rate:", dropout_rate)
print("batch momentum:", batch_momentum)
print("initial learning rate:", learning_rate)
print("hidden layer neuron numbers:", n_neurons)
print("Deep learning model for PCa Gleason grades classificaiton: complete!!!")

# calcualte running time
stop = timeit.default_timer()
print('Running Time:', np.around(stop - start, 0), 'seconds')









