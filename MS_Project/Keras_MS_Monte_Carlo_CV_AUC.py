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
import numpy as np
import itertools
import matplotlib.pyplot as plt
from datetime import datetime
from functools import partial
import timeit
import scipy.stats
from time import gmtime, strftime

import numpy
from numpy import mean
from numpy import std
from numpy import array
from numpy import argmax

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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
from sklearn.preprocessing import label_binarize


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
        self.class0_ratio = class0_ratio
        self.class1_ratio = class1_ratio
        self.class2_ratio = class2_ratio
        self.ratio3_ratio = class3_ratio
        self.ratio4_ratio = class4_ratio
        self.x_range     = x_range

        
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

    def dataset_construct(self):
        
        df_2 = self.data_structuring()
        #print(df_2.shape)

        y = df_2.y_cat.astype('int')
        #print(y.shape)

        x = df_2.iloc[:, self.x_range]
        #print(x.shape)
        
        return df_2, x, y

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

        # optimizer functions

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
def model_training(model, x_tarin, y_train, x_test, y_test):
    
    history = model.fit(
                        x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        callbacks=None,
                        validation_split=val_split,
                        validation_data=None,
                        shuffle=True,
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0,
                        steps_per_epoch=None,
                        validation_steps=None            
                        )

    score = model.evaluate(
                           x_test,
                           y_test,
                           verbose=0
                           )
    
    y_pred = model.predict(x_test)
    
    y_pred_label = np.argmax(y_pred, axis=1)
    
    return y_pred

# ----------------------------------------------------------------------------------
# ROC and AUC
# ----------------------------------------------------------------------------------
def ROC_AUC():

    fpr       = dict()
    tpr       = dict()
    roc_auc   = dict()
    threshold = dict()
    AUC       = dict()
    
    for i in range(n_outputs):
        
        fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_pred[:, i])
        
        roc_auc[i] = auc(fpr[i], tpr[i])

    #print(roc_auc)

    AUC = np.array([item for item in roc_auc.values()])
    AUC = np.round(AUC, decimals=3)
    print(AUC)

    return AUC
      
# ----------------------------------------------------------------------------------
# mean, 95% CI
# ----------------------------------------------------------------------------------
def mean_CI(stat, confidence=0.95):
    
    alpha  = 0.95
    mean   = np.mean(np.array(stat))
    
    p_up   = (1.0 - alpha)/2.0*100
    lower  = max(0.0, np.percentile(stat, p_up))
    
    p_down = ((alpha + (1.0 - alpha)/2.0)*100)
    upper  = min(1.0, np.percentile(stat, p_down))
    
    return mean, lower, upper

def stat_summary():

    stat_sum = []
    
    for i in range(all_stat.shape[1]):
        
        stat  = all_stat[:, i]
        stat_ = mean_CI(stat)
        stat_sum.append(stat_)
        
    stat_sum = np.array(stat_sum)
    stat_sum = np.round(stat_sum, decimals=3)
    
    return stat_sum

def stat_report():

    stat_sum = stat_summary()

    stats = pd.DataFrame(
                         stat_sum,
                         index=[                           
                                'AUC_1', 'AUC_2', 'AUC_3', 'AUC_4', 'AUC_5'
                               ],                            
                         columns=[
                                  'mean',
                                  '95% CI -',
                                  '95% CI +'
                                  ]
                         )

    filename = str(DNN_model) + '_' + \
               str(iteration) + '_' + \
               str(epochs) + '_' + \
               str(strftime("%d-%b-%Y-%H-%M-%S", gmtime())) + \
               '.csv'

    stats.to_csv(os.path.join(result_dir, filename))

    return stats
    
# ----------------------------------------------------------------------------------
# main function
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    epochs         = 100
    iteration      = 10
    learning_rate  = 0.001
    batch_momentum = 0.97
    batch_size     = 100
    dropout_rate   = 0    
    alpha          = 0.3
    random_state   = 42
    ELU_alpha      = 1.0
    digit          = 3
    count          = 0
    val_split      = 0.1
    train_size     = 0.95
    class0_ratio   = 1.0
    class1_ratio   = 1.0
    class2_ratio   = 1.0
    class3_ratio   = 1.0
    class4_ratio   = 1.0

    DNN_model      = 'cMRI'
    x_DBSI         = [16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]     
    x_cMRI         = [28, 29]                                        
    x_MTC          = [28, 29, 30]                                    
    x_DTI          = [15, 16, 17, 18, 28, 29]
    x_range        = x_cMRI
    n_inputs       = len(x_range)
    n_outputs      = 5
    all_stat       = []
    
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
    init              = 'he_uniform'     
    optimizer         = 'adam'         
    loss              = 'categorical_crossentropy'
    output_activation = 'softmax'
    activation        = ELU(alpha=ELU_alpha)

    '''
    keranl initializer: 'he_uniform', 'lecun_normal', 'lecun_uniform'
    optimizer function: 'adam', 'adamax', 'nadam', 'sgd'
    loss function: 'categorical_crossentropy'
    activation function: LeakyReLU(alpha=alpha)
    '''

    # data and results path for windows system
    project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI'
    result_dir  = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI\result'
    log_dir     = r'\\10.39.42.102\temp\Zezhong_Ye\2019_MS\AI\log'
    file        = '20190302.csv'


    # ----------------------------------------------------------------------------------
    # run functions
    # ---------------------------------------------------------------------------------- 
    print("Deep Neural Network for PCa grade classification: start...")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

    start = timeit.default_timer()
    
    data_path()
    
    df_tot, x, y = data(
                        file,
                        project_dir,
                        random_state,
                        class0_ratio, 
                        class1_ratio,
                        class2_ratio,
                        class3_ratio,
                        class4_ratio,
                        x_range
                        ).dataset_construct()

    y = label_binarize(y, classes=[0, 1, 2, 3, 4])

    sss = StratifiedShuffleSplit(
                                 n_splits=iteration,
                                 test_size=0.1,
                                 random_state=42
                                 )
        
    for train_index, test_index in sss.split(x, y):

        count += 1
        print('\niteration: ' + str(count) + '/' + str(iteration))

        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
          
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
            
        y_pred = model_training(
                                model,
                                x_train,
                                y_train,
                                x_test,
                                y_test
                                )
        
        AUC = ROC_AUC()
        
        all_stat.append(AUC)

    all_stat = np.array(all_stat)
    
    stats = stat_report()
    
    print(stats)
    
    print('key model parameters:')
    print('DNN model:           ', str(DNN_model))   
    print('bagging iteration:   ', iteration)
    print('train dataset size:  ', len(x_train))
    print('val dataset size:    ', np.around(len(x_train)*val_split, 0))  
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



















