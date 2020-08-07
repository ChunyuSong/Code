#----------------------------------------------------------------------
# deep learning classifier using a multiple layer perceptron (MLP)
# batch normalization was used
#
# Author: Zezhong Ye;
# Date: 03.14.2019
# TensorBoard support:
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
#----------------------------------------------------------------------

import os
import random
import seaborn as sn
import pandas as pd
import glob2 as glob
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import timeit
import scipy.stats
from time import gmtime, strftime

import numpy as np
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax

import keras
from keras import backend as K

from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Activation, Dropout
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.backend import tensorflow_backend as K

import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
def data_path():

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
class data(object):
    
    '''
    calculate DNN statisical results
    '''
    
    def __init__(
                 self,
                 file,
                 project_dir,
                 random_state,
                 train_split,
                 class0_ratio, 
                 class1_ratio,
                 class2_ratio,
                 class3_ratio,
                 class4_ratio,
                 x_range
                 ):
                        
        self.file         = file
        self.project_dir  = project_dir
        self.random_state = random_state
        self.train_split  = train_split
        self.class0_ratio = class0_ratio
        self.class1_ratio = class1_ratio
        self.class2_ratio = class2_ratio
        self.ratio3_ratio = class3_ratio
        self.ratio4_ratio = class4_ratio
        self.x_range      = x_range
       
    def data_loading(self):
        
        maps_list = [
                     'dti_fa_map.nii',                      #15
                     'dti_adc_map.nii',                     #16
                     'dti_axial_map.nii',                   #17
                     'dti_radial_map.nii',                  #18
                     'fiber_ratio_map.nii',                 #19
                     'fiber1_fa_map.nii',                   #20
                     'fiber1_axial_map.nii',                #21
                     'fiber1_radial_map.nii',               #22
                     'restricted_ratio_map.nii',            #23
                     'hindered_ratio_map.nii',              #24
                     'water_ratio_map.nii',                 #25
                     'b0_map.nii',                          #26
                     'T2W',                                 #27
                     'FLAIR',                               #28
                     'MPRAGE',                              #29
                     'MTC'                                  #30
                     ]

        df = pd.read_csv(os.path.join(self.project_dir, self.file))
        
        return df
    
    def data_structuring(self):

        df = self.data_loading()

        df.loc[df['ROIClass'] == 1, 'y_cat'] = 0
        df.loc[df['ROIClass'] == 2, 'y_cat'] = 1
        df.loc[df['ROIClass'] == 4, 'y_cat'] = 2
        df.loc[df['ROIClass'] == 5, 'y_cat'] = 3
        df.loc[df['ROIClass'] == 6, 'y_cat'] = 4

        class0 = df[df['y_cat'] == 0]
        class0_sample = class0.sample(int(class0.shape[0]*class0_ratio))
        class1 = df[df['y_cat'] == 1]
        class1_sample = class1.sample(int(class1.shape[0]*class1_ratio))
        class2 = df[df['y_cat'] == 2]
        class2_sample = class2.sample(int(class2.shape[0]*class2_ratio))
        class3 = df[df['y_cat'] == 3]
        class3_sample = class3.sample(int(class3.shape[0]*class3_ratio))
        class4 = df[df['y_cat'] == 4]
        class4_sample = class4.sample(int(class4.shape[0]*class4_ratio))

        df_2 = pd.concat([
                          class0_sample,
                          class1_sample,
                          class2_sample,
                          class3_sample,
                          class4_sample
                          ])

        return df_2

    def dataset_construction(self):
        
        df_2 = self.data_structuring()

        y = df_2.y_cat.astype('int')

        x = df_2.iloc[:, self.x_range]         
        
        x_train, x_test, y_train, y_test = train_test_split(
                                                            x,
                                                            y,
                                                            test_size=self.train_split,
                                                            random_state=self.random_state,
                                                            stratify=y
                                                            )

        return x_train, x_test, y_train, y_test

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# ----------------------------------------------------------------------------------
class Keras_model(object):
    def __init__(
                 self,
                 init,
                 optimizer,
                 loss,
                 activation,
                 dropout_rate,
                 batch_momentum, 
                 n_inputs,
                 n_outputs
                 ):
        
        self.init           = init
        self.optimizer      = optimizer
        self.loss           = loss
        self.dropout_rate   = dropout_rate
        self.batch_momentum = batch_momentum
        self.n_inputs       = n_inputs
        self.n_outputs      = n_outputs
        self.activation     = activation
           
    def build_model(self):
    
        model = Sequential()

        dense_layer = partial(
                              Dense,
                              init=self.init, 
                              use_bias=False,
                              activation=None,
                              )

        batch_normalization = partial(
                                      BatchNormalization,
                                      axis=-1,
                                      momentum=self.batch_momentum,
                                      epsilon=0.001,
                                      beta_initializer='zeros',
                                      gamma_initializer='ones',
                                      beta_regularizer=None,
                                      gamma_regularizer=None                             
                                      )

                                                                  
        # input layer                              
        model.add(dense_layer(self.n_inputs, input_dim=self.n_inputs))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 1
        model.add(dense_layer(n_hidden1))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 2
        model.add(dense_layer(n_hidden2))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 3
        model.add(dense_layer(n_hidden3))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 4
        model.add(dense_layer(n_hidden4))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 5
        model.add(dense_layer(n_hidden5))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 6
        model.add(dense_layer(n_hidden6))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 7
        model.add(dense_layer(n_hidden7))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 8
        model.add(dense_layer(n_hidden8))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # hidden layer 9
        model.add(dense_layer(n_hidden9))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))
                  
        # hidden layer 10
        model.add(dense_layer(n_hidden10))
        model.add(batch_normalization())
        model.add(self.activation)
        model.add(Dropout(self.dropout_rate))

        # output layer
        model.add(dense_layer(self.n_outputs))
        model.add(batch_normalization())
        model.add(Activation(output_activation))

        #model.summary()

        model.compile(
                      loss=self.loss,
                      optimizer=self.optimizer,
                      metrics=['accuracy']
                      )
        
        return model
# ----------------------------------------------------------------------------------
# trainning DNN model
# ----------------------------------------------------------------------------------
def model_training(x_train, x_test, y_train, y_test):
    config = tf.ConfigProto(
                           intra_op_parallelism_threads=32,
                           inter_op_parallelism_threads=32,
                           allow_soft_placement=True,
                           log_device_placement=True,
                           device_count={"CPU": 32}
                           )
    tf.set_random_seed(1)
    session = tf.Session(graph=tf.get_default_graph(), config=config)
    K.set_session(session)
    history = model.fit(
                        x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        callbacks=None,
                        validation_split=val_split,
                        validation_data=None,
                        shuffle=True,
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0,
                        steps_per_epoch=None,
                        validation_steps=None,            
                        )

    score = model.evaluate(
                           x_test,
                           y_test,
                           verbose=0
                           )
    
    y_pred = model.predict(x_test)
    
    y_pred_label = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_label)
    cm = np.around(cm, 3)
    
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 2)
    
    return score, cm, cm_norm

# ----------------------------------------------------------------------------------
# plot confusion matrix
# ----------------------------------------------------------------------------------
def plot_CM(CM, fmt):
    
    ax = sn.heatmap(
                    CM,
                    annot=True,
                    cbar=True,
                    cbar_kws={'ticks': [-0.1]},
                    annot_kws={'size': 15, 'fontweight': 'bold'},
                    cmap="Blues",
                    fmt=fmt,
                    linewidths=0.5
                    )

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
    
    plt.savefig(
                os.path.join(result_dir, 'CM_1.png'),
                format='png',
                dpi=600
                )

    plt.show()

# ----------------------------------------------------------------------------------
# model evaluation
# ----------------------------------------------------------------------------------
def model_eval(cm):
    
    ACC= [] #accuracy
    TPR= [] #True positive rate (Sensitivity, Recall)
    TNR= [] #True negative rate (Specificity, Selectivity)
    PPV= [] #Positive Predictive Value
    NPV= [] #Negative Predictive Value
    FPR= [] #False Positive Rate
    FNR= [] #False Negative Rate
    FDR= [] #False Discovery Rate
    PR = [] #Precision
    F1 = [] #F1 score

    FP  = cm[:].sum(axis=0) - np.diag(cm[:])
    FN  = cm[:].sum(axis=1) - np.diag(cm[:])
    TP  = np.diag(cm[:])
    TN  = cm[:].sum() - (FP+FN+TP)

    ACC = (TP + TN) / (TP + FP + FN + TN)       
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)       
    PPV = TP / (TP + FP)    
    NPV = TN / (TN + FN)
    FPR = FP / (FP + TN)
    FNR = FN / (TP + FN)
    FDR = FP / (TP + FP)
    F_1  = 2 * (PPV * TPR) / (PPV + TPR)

    stat_list = [
                 TPR[0], TPR[1], TPR[2], TPR[3], TPR[4],
                 TNR[0], TNR[1], TNR[2], TNR[3], TNR[4],
                 PPV[0], PPV[1], PPV[2], PPV[3], PPV[4],
                 NPV[0], NPV[1], NPV[2], NPV[3], NPV[4],
                 F_1[0], F_1[1], F_1[2], F_1[3], F_1[4],
                 ACC[0], ACC[1], ACC[2], ACC[3], ACC[4]
                 ]
    
    stat_list = np.array(stat_list)
    stat_list = np.around(stat_list.astype(np.double), decimals=3)

    return stat_list

def stat_report(stat_list):

    stat_df = pd.DataFrame(
                           stat_list,
                           index=[
                                  'TPR_1', 'TPR_2', 'TPR_3', 'TPR_4', 'TPR_5',
                                  'TNR_1', 'TNR_2', 'TNR_3', 'TNR_4', 'TNR_5',
                                  'PPV_1', 'PPV_2', 'PPV_3', 'PPV_4', 'PPV_5',
                                  'NPV_1', 'NPV_2', 'NPV_3', 'NPV_4', 'NPV_5',
                                  'F1_1',  'F1_2',  'F1_3',  'F1_4',  'F1_5',
                                  'ACC_1', 'ACC_2', 'ACC_3', 'ACC_4', 'ACC_5'
                                  ],                            
                           columns=[
                                    'value',
                                   ]
                           )

    filename = str(DNN_model) + '_' + \
               str(epochs) + '_' + \
               str(strftime("%d-%b-%Y-%H-%M-%S", gmtime())) + \
               '.csv'

    stat_df.to_csv(os.path.join(result_dir, filename)) 

    return stat_df
    
# ----------------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    epochs         = 100
    learning_rate  = 0.001
    batch_momentum = 0.97
    batch_size     = 200
    dropout_rate   = 0    
    alpha          = 0.3
    random_state   = 42
    ELU_alpha      = 1.0
    digit          = 3
    train_split    = 0.1
    val_split      = 0.1
    class0_ratio   = 1.0
    class1_ratio   = 1.0
    class2_ratio   = 1.0
    class3_ratio   = 1.0
    class4_ratio   = 1.0

    x_DBSI         = [16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]     
    x_T1W_T2W      = [28, 29]                                        
    x_MTC          = [28, 29, 30]                                    
    x_DTI          = [15, 16, 17, 18, 28, 29]
    DNN_model      = 'DBSI'
    x_range        = x_DBSI
    n_inputs       = len(x_range)
    n_outputs      = 5
    
    n_neurons      = 100
    n_hidden1      = n_neurons
    n_hidden2      = n_neurons
    n_hidden3      = n_neurons
    n_hidden4      = n_neurons
    n_hidden5      = n_neurons
    n_hidden6      = n_neurons
    n_hidden7      = n_neurons
    n_hidden8      = n_neurons
    n_hidden9      = n_neurons
    n_hidden10     = n_neurons

    # model functions
    init              = 'he_normal'     
    optimizer         = 'adam'         
    loss              = 'sparse_categorical_crossentropy'
    output_activation = 'softmax'
    activation        = ELU(alpha=ELU_alpha)

    '''
    keranl initializer: 'he_uniform', 'lecun_normal', 'lecun_uniform'
    optimizer function: 'adam', 'adamax', 'nadam', 'sgd'
    loss function: 'categorical_crossentropy'
    activation function: LeakyReLU(alpha=alpha)
    '''

    # data and results path for windows system
    #project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI'
    #result_dir  = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI\result'
    #log_dir     = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI\log'

    # data and results path for linux system
    project_dir = '/bmrp092temp/Zezhong_Ye/2019_MS/AI'
    result_dir  = '/bmrp092temp/Zezhong_Ye/2019_MS/AI/result'
    log_dir     = '/bmrp092temp/Zezhong_Ye/2019_MS/AI/log'
    file        = '20190302.csv'

    # ----------------------------------------------------------------------------------
    # run functions
    # ----------------------------------------------------------------------------------

    # os.environ["OMP_NUM_THREADS"] = "32"
    # os.environ["KMP_BLOCKTIME"] = "30"
    # os.environ["KMP_SETTINGS"] = "1"
    # os.environ["KMP_AFFINITY"] = "granularity=fine, verbose, compact, 1, 0"

    print("Deep Neural Network for PCa grade classification: start...")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    start = timeit.default_timer()

    data_path()

    x_train, x_test, y_train, y_test = data(
                                            file,
                                            project_dir,
                                            random_state,
                                            train_split,
                                            class0_ratio,
                                            class1_ratio,
                                            class2_ratio,
                                            class3_ratio,
                                            class4_ratio,
                                            x_range
                                            ).dataset_construction()

    model = Keras_model(
                        init,
                        optimizer,
                        loss,
                        activation,
                        dropout_rate,
                        batch_momentum,
                        n_inputs,
                        n_outputs
                        ).build_model()

    score, cm, cm_norm = model_training(x_train, x_test, y_train, y_test)

    stat_list = model_eval(cm)

    stat_df = stat_report(stat_list)


    print('\noverall test loss:', np.around(score[0], digit))
    print('overall test accuracy:', np.around(score[1], digit))
    print('confusion matrix:\n', cm)
    print('confusion matrix:\n', cm_norm)
    print(stat_df)
    #print(classification_report(y_test, y_pred_label, digits=3))
    # plot_CM(cm, 'd')
    # plot_CM(cm_norm, '')

    print('key model parameters:')
    print('train dataset size:  ', len(x_train))
    print('val dataset size:    ', np.around(len(x_train) * val_split, 0))
    print('test dataset size:   ', len(x_test))
    print('epochs:              ', epochs)
    print('batch size:          ', batch_size)
    print('dropout rate:        ', dropout_rate)
    print('batch momentum:      ', batch_momentum)
    print('learning rate:       ', learning_rate)
    print('neuron numbers:      ', n_neurons)

    stop = timeit.default_timer()
    running_seconds = np.around(stop - start, 0)
    running_minutes = np.around(running_seconds/60, 0)
    print('DNN running time:    ', running_seconds, 'seconds')
    print('DNN running time:    ', running_minutes, 'minutes')





















