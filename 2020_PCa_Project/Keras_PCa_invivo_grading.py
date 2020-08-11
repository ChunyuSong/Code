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
dropout_rate = 0
n_neurons = 100
epochs = 200
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
alpha = 0.3
n_inputs = 2
n_outputs = 5
optimizer = 'adam'
test_size = 0.2
random_state = 42

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
print("PCa DNN classification ROC analysis: start...")

# data path for windows system
project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCA_in_vivo_data_excel'
result_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\invivo_grading\result'
log_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\invivo_grading\log'

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
def data_loading():
       
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

    files = glob.glob(project_dir + "/*.xlsx")
    df = pd.DataFrame()
    for f in files:
        data = pd.read_excel(f,'Sheet1',header = None)
        data.iloc[:, 1:] = (data.iloc[:, 1:])/(data.iloc[:, 1:].max())
        data = data.iloc[data[data.iloc[:, 0] != 0].index]
        df = df.append(data)

    ### define label class
    df.loc[df.iloc[:, 0] == 1, 'y_cat'] = 0
    df.loc[df.iloc[:, 0] == 2, 'y_cat'] = 1
    df.loc[df.iloc[:, 0] == 3, 'y_cat'] = 2
    df.loc[df.iloc[:, 0] == 4, 'y_cat'] = 3
    df.loc[df.iloc[:, 0] == 5, 'y_cat'] = 4

    # balance train and validation data from each class
    class0 = df[df['y_cat'] == 0]
    class0_sample = class0.sample(int(class0.shape[0]))
    class1 = df[df['y_cat'] == 1]
    class1_sample = class1.sample(int(class1.shape[0]))
    class2 = df[df['y_cat'] == 2]
    class2_sample = class2.sample(int(class2.shape[0]))
    class3 = df[df['y_cat'] == 3]
    class3_sample = class3.sample(int(class3.shape[0]))
    class4 = df[df['y_cat'] == 4]
    class4_sample = class4.sample(int(class4.shape[0]))
    print(len(class0_sample),len(class1_sample),len(class2_sample),len(class3_sample),len(class4_sample))
    # reconstruct dataset from balanced data
    df_2 = pd.concat(
                     [class0_sample,
                      class1_sample,
                      class2_sample,
                      class3_sample,
                      class4_sample]
                     )

    # DBSI
    #x = df.iloc[:, [1,13,14,18,19,20,22,26,27,28,29,31,32,33,35,36,37,38,39,40]]
    
    # DBSI + DTI
    #x = df.iloc[:, [1,2,3,8,11,13,14,18,19,20,22,26,27,28,29,31,32,33,35,36,37,38,39,40]]
    
    # DTI
    #x = df.iloc[:, [2, 3, 8, 11]]

    # ADC + b0
    x = df.iloc[:, [1, 2]]
    
    y = df_2['y_cat'].astype('int')

    # construct train dataset with 80% slpit
    x_train, x_test, y_train, y_test = train_test_split(
                                                        x,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state
                                                        )

    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = data_loading()
print("data loading: complete!")

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# kernal initializer function: "he_uniform", "lecun_normal", "lecun_uniform",
#                              "glorot_uniform", "glorot_normal";
# ----------------------------------------------------------------------------------
def Keras_model(init, loss):
    model = Sequential()

    dense_layer = partial(
                          keras.layers.Dense,
                          init=init, 
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
                  loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy']
                  )
    
    return model
# ----------------------------------------------------------------------------------
# trainning DNN model
# ----------------------------------------------------------------------------------
def DNN_model_training(verbose, validation_split):
    DNN_model = Keras_model('he_normal', 'sparse_categorical_crossentropy')
    history = DNN_model.fit(
                            x=x_train,
                            y=y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=verbose,
                            callbacks=None,
                            validation_split=0.2,
                            shuffle=True,
                            class_weight=None,
                            sample_weight=None,
                            initial_epoch=0,
                            steps_per_epoch=None,
                            validation_steps=None,            
                            )

    score = DNN_model.evaluate(x_test, y_test, verbose=0)
    y_pred = DNN_model.predict(x_test)
    y_pred_label = np.argmax(y_pred, axis=1)

    return score, y_pred, y_pred_label

score, y_pred, y_pred_label = DNN_model_training(2, 0.5)
print('\noverall test loss:', np.around(score[0], d))
print('overall test accuracy:', np.around(score[1], d))

# ----------------------------------------------------------------------------------
# calculate confusion matrix and nornalized confusion matrix
# ----------------------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_label)
cm = np.around(cm, 2)
print("\nconfusion matrix:")
print(cm)

print("\nnormalized confusion matrix:")
cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
cm_norm = np.around(cm_norm, 2)
print(cm_norm)

# ----------------------------------------------------------------------------------
# sensitivity, specificity, precision
# ----------------------------------------------------------------------------------
def statistical_results(cm):
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

    return ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR

ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR = statistical_results(cm)

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
print('Running Time:', np.around(stop - start, 0), 'seconds')

# ----------------------------------------------------------------------------------
# plot confusion matrix
# ----------------------------------------------------------------------------------
def plot_CM(cm, fmt):
    ax = sn.heatmap(
                    cm,
                    annot=True,
                    cbar=True,
                    cbar_kws={'ticks': [0]},
                    annot_kws={'size': 15, 'fontweight': 'bold'},
                    cmap="Blues",
                    fmt=fmt,
                    linewidths=0.5
                    )

    #cbar.tick_params(labelsize=12)
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=5, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=5, color='k', linewidth=4)

    ax.tick_params(direction='out', length=4, width=2, colors='k')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'confusion_matrix_1.png'),
                format='png',
                dpi=600)

    plt.show()
    plt.close()
    return plt

plot_CM(cm_norm, '')
plot_CM(cm, 'd')
  
# ----------------------------------------------------------------------------------
# print key DNN model parameters
# ----------------------------------------------------------------------------------
print("train dataset number:", len(x_train)/2)
print("validation dataset number:", len(x_train)/2)
print("test dataset number:", len(x_test))
print("epochs:", epochs)
print("batch size:", batch_size)
print("dropout rate:", dropout_rate)
print("batch momentum:", batch_momentum)
print("initial learning rate:", learning_rate)
print("hidden layer neuron numbers:", n_neurons )
print("Deep learning model for PCa Gleason grades classificaiton: complete!!!")




























