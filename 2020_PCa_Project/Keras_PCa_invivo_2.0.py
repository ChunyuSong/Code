#----------------------------------------------------------------------
# deep learning classifier for prostate cancer classification
# batch normalization was used
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
#-------------------------------------------------------------------------------------------

import os
import timeit
import itertools
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
import nibabel as nib
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
from time import gmtime, strftime

import keras
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Activation, Dropout
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
class PCa_data(object):
    
    '''
    calculate DNN statisical results
    '''

    def __init__(
                 self,
                 project_dir,
                 random_state,
                 val_test_split,
                 train_file_1,
                 train_file_2,
                 test_file_1,
                 test_file_2,
                 pred_file,
                 ratio_1, 
                 ratio_2,
                 ratio_3,
                 ratio_4,
                 ratio_5,
                 ratio_6,       
                 x_inputs
                 ):
                    
        self.project_dir    = project_dir
        self.random_state   = random_state
        self.val_test_split = val_test_split
        self.train_file_1   = train_file_1
        self.train_file_2   = train_file_2
        self.test_file_1    = test_file_1
        self.test_file_2    = test_file_2
        self.pred_file      = pred_file
        self.ratio_1        = ratio_1
        self.ratio_2        = ratio_2
        self.ratio_3        = ratio_3
        self.ratio_4        = ratio_4
        self.ratio_5        = ratio_5
        self.ratio_6        = ratio_6 
        self.x_inputs       = x_inputs

    def map_list():
    
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
        
        return map_list

    def data_train_loading(self):
        
        df_1     = pd.read_csv(os.path.join(project_dir, self.train_file_1))
        df_2     = pd.read_csv(os.path.join(project_dir, self.train_file_2))
        df_train = pd.concat([df_1, df_2])
    
        return df_train

    def data_test_loading(self):
        
        df_5    = pd.read_csv(os.path.join(project_dir, self.test_file_1))
        df_6    = pd.read_csv(os.path.join(project_dir, self.test_file_2))
        df_test = pd.concat([df_5, df_6])
        
        return df_test

    def data_train(self):
        '''
        construct train dataset
        '''

        df_train = self.data_train_loading()

        df_train.loc[df_train['ROI_Class'] == 'p', 'y_cat'] = 0
        df_train.loc[df_train['ROI_Class'] == 'c', 'y_cat'] = 1
        df_train.loc[df_train['ROI_Class'] == 't', 'y_cat'] = 2
        
        class1_train        = df_train[df_train['y_cat'] == 0]
        class1_train_sample = class1_train.sample(int(class1_train.shape[0]*self.ratio_1))
        
        class2_train        = df_train[df_train['y_cat'] == 1]
        class2_train_sample = class2_train.sample(int(class2_train.shape[0]*self.ratio_2))

        class3_train        = df_train[df_train['y_cat'] == 2]
        class3_train_sample = class3_train.sample(int(class3_train.shape[0]*self.ratio_3))

        df_train_sum = pd.concat([class1_train_sample, class2_train_sample, class3_train_sample])
        
        x_train      = df_train_sum.iloc[:, self.x_inputs]
        
        y_train      = df_train_sum.y_cat.astype('int')

        return x_train, y_train

    def data_val_test(self):
        
        '''
        construct test dataset
        '''

        df_test = self.data_test_loading()

        df_test.loc[df_test['ROI_Class'] == 'p', 'y_cat'] = 0
        df_test.loc[df_test['ROI_Class'] == 'c', 'y_cat'] = 1
        df_test.loc[df_test['ROI_Class'] == 't', 'y_cat'] = 2
        
        class1_test        = df_test[df_test['y_cat'] == 0]
        class1_test_sample = class1_test.sample(int(class1_test.shape[0]*self.ratio_4))
        
        class2_test        = df_test[df_test['y_cat'] == 1]
        class2_test_sample = class2_test.sample(int(class2_test.shape[0]*self.ratio_5))

        class3_test        = df_test[df_test['y_cat'] == 2]
        class3_test_sample = class3_test.sample(int(class3_test.shape[0]*self.ratio_6))

        df_test_sum = pd.concat([class1_test_sample, class2_test_sample, class3_test_sample])
        
        x_val_test  = df_test_sum.iloc[:, self.x_inputs]
        
        y_val_test  = df_test_sum.y_cat.astype('int')

        x_val, x_test, y_val, y_test = train_test_split(
                                                        x_val_test,
                                                        y_val_test,
                                                        test_size=self.val_test_split,
                                                        random_state=self.random_state
                                                        )
        return x_val, x_test, y_val, y_test

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
def model_training():
    
    history = model.fit(
                        x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        callbacks=None,
                        validation_split=None,
                        validation_data=(x_val, y_val),
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

    y_pred        = model.predict(x_test)  
    y_pred_label  = np.argmax(y_pred, axis=1)

    test_loss     = score[0]
    test_accuracy = score[1]

    test_loss     = np.around(test_loss, 3)
    test_accuracy = np.around(test_accuracy, 3)
    
    return score, y_pred, y_pred_label, test_loss, test_accuracy

# ----------------------------------------------------------------------------------
# calculate confusion matrix and nornalized confusion matrix
# ----------------------------------------------------------------------------------
def con_mat(y_pred_label):
    
    cm      = confusion_matrix(y_test, y_pred_label)
    cm      = np.around(cm, 2)
    
    cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.around(cm_norm, 2)
    
    return cm, cm_norm

# ----------------------------------------------------------------------------------
# plot confusion matrix
# ----------------------------------------------------------------------------------
def plot_CM(CM, fmt):
    
    ax = sn.heatmap(
                    CM,
                    annot=True,
                    cbar=True,
                    cbar_kws={'ticks': [-0.1]},
                    annot_kws={'size': 22, 'fontweight': 'bold'},
                    cmap="Blues",
                    fmt=fmt,
                    linewidths=0.5
                    )

    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=3, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=3, color='k', linewidth=4)

    ax.tick_params(direction='out', length=4, width=2, colors='k')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.tight_layout()

    cm_filename = 'cm' + '_' + \
                  str(learning_rate) + '_' + \
                  str(batch_momentum) + '_' + \
                  str(epochs) + '_' + \
                  str(dropout_rate) + \
                  str(batch_size) + \
                  strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
    
    plt.savefig(
                os.path.join(result_dir, cm_filename),
                format='png',
                dpi=600
                )

    plt.close()
    
# ----------------------------------------------------------------------------------
# model evaluation
# ----------------------------------------------------------------------------------
class model_evaluation(object):
    
    '''
    calculate DNN statisical results
    '''
    
    def __init__(self, CM, digit):
        self.cm    = CM
        self.digit = digit
    
    def statistics_1(self):
        
        FP = self.cm[:].sum(axis=0) - np.diag(self.cm[:])
        FN = self.cm[:].sum(axis=1) - np.diag(self.cm[:])
        TP = np.diag(self.cm[:])
        TN = self.cm[:].sum() - (FP+FN+TP)
        
        return FP, FN, TP, TN

    def statistics_2(self):

        FP, FN, TP, TN = self.statistics_1()

        ACC = (TP+TN)/(TP+FP+FN+TN)       
        TPR = TP/(TP+FN)
        TNR = TN/(TN+FP)       
        PPV = TP/(TP+FP)    
        NPV = TN/(TN+FN)
        FPR = FP/(FP+TN)
        FNR = FN/(TP+FN)
        FDR = FP/(TP+FP)

        ACC = np.around(ACC, self.digit)      
        TPR = np.around(TPR, self.digit)
        TNR = np.around(TNR, self.digit)       
        PPV = np.around(PPV, self.digit)    
        NPV = np.around(NPV, self.digit)
        FPR = np.around(FPR, self.digit)
        FNR = np.around(FNR, self.digit)
        FDR = np.around(FDR, self.digit)
          
        return ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR 

# ----------------------------------------------------------------------------------
# run the model
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    alpha          = 0.3
    random_state   = 42
    ELU_alpha      = 1.0
    digit          = 3
    val_test_split = 0.5
    ratio_1        = 0.84
    ratio_2        = 1.0
    ratio_3        = 1.0
    ratio_4        = 0.652
    ratio_5        = 0.716
    ratio_6        = 1.0
    #x_inputs       = [7, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 28, 29]
    x_inputs       = range(7, 30)
    n_inputs       = len(x_inputs)
    n_outputs      = 3
    count          = 0

    Learning_rate  = [0.01, 0.001]
    Momentum       = [0.9]
    Dropout_rate   = [0.3, 0]
    Batch_size     = [500]
    Epochs         = [5]
    N_neurons      = [100]

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
    loss              = 'sparse_categorical_crossentropy'
    output_activation = 'softmax'
    activation        = ELU(alpha=ELU_alpha)     
    
    '''
    keranl initializer: 'he_uniform', 'lecun_normal', 'lecun_uniform'
    optimizer function: 'adam', 'adamax', 'nadam', 'sgd'
    loss function: 'categorical_crossentropy'
    activation function: LeakyReLU(alpha=alpha)
    '''

    # data and results path 
    project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\data'
    result_dir  = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\result'
    log_dir     = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\log'
             
    train_file_1   = 'benign_mpMRI.csv'
    train_file_2   = 'PCa_train.csv'
    test_file_1    = 'benign_biopsy.csv'
    test_file_2    = 'PCa_test.csv'
    pred_file      = 'PCa.csv'
    DBSI_folder    = 'DBSI_results_0.1_0.1_0.8_0.8_2.3_2.3'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------
   
    print("Deep Neural Network for PCa grade classification: start...")

    start = timeit.default_timer()
    
    data_path()
    
    PCa_Data = PCa_data(
                        project_dir,
                        random_state,
                        val_test_split,
                        train_file_1,
                        train_file_2,
                        test_file_1,
                        test_file_2,
                        pred_file,
                        ratio_1, 
                        ratio_2,
                        ratio_3,
                        ratio_4,
                        ratio_5,
                        ratio_6,
                        x_inputs
                        )
    
    x_train, y_train = PCa_Data.data_train()
    
    x_val, x_test, y_val, y_test = PCa_Data.data_val_test()

    total_run = len(Momentum)*len(Epochs)*len(Batch_size)*len(Learning_rate)*len(N_neurons)

    breaking = False

    for i in Batch_size:
        
        for j in Momentum:
            
            for k in Epochs:

                for l in Learning_rate:

                    for m in N_neurons:

                        for n in Dropout_rate:

                            count += 1

                            print('\nRunning times: ' + str(count) + '/' + str(total_run))

                            batch_size     = i
                            batch_momentum = j
                            epochs         = k
                            learning_rate  = l
                            n_neurons      = m
                            dropout_rate   = n
                   
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
                            
                            score, y_pred, y_pred_label, test_loss, test_accuracy = model_training()
                            
                            cm, cm_norm = con_mat(y_pred_label)
                            
                            ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR = model_evaluation(cm, digit).statistics_2()

                            plot_CM(cm_norm, '')

                            # ----------------------------------------------------------------------------------
                            # confusion matrix, sensitivity, specificity, presicion, f-score, model parameters
                            # ----------------------------------------------------------------------------------
                            print('\noverall test loss: ', np.around(score[0], digit))
                            print('overall test accuracy:', np.around(score[1], digit))
                            #print("\nconfusion matrix:")
                            #print(cm)
                            #print("\nnormalized confusion matrix:")
                            print(cm_norm) 
                            #print("sensitivity:", TPR)
                            #print("specificity:", TNR)
                            #print("precision:", PPV)
                            #print("negative predictive value:", NPV)
                            #print("false positive rate:", FPR)
                            #print("false negative rate:", FNR)
                            #print("false discovery rate:", FDR)
                            #print("\nprecision, recall, f1-score, support:")
                            #print(classification_report(y_test, y_pred_label, digits=digit))                              
                            print('epochs:        ', epochs)
                            print('batch size:    ', batch_size)
                            print('dropout rate:  ', dropout_rate)
                            print('batch momentum:', batch_momentum)
                            print('learning rate: ', learning_rate)
                            print('neuron numbers:', n_neurons)

                            if test_accuracy > 0.98:
                                breaking = True

                        if breaking == True:
                            break

                    if breaking == True:
                        break
                    
                if breaking == True:
                    break

            if breaking == True:
                break

        if breaking == True:
            break

    print("train dataset size:", len(x_train))
    print("validation dataset size:", len(x_val))
    print("test dataset size:", len(x_test))       
    stop = timeit.default_timer()
    running_seconds = np.around(stop - start, 0)
    running_minutes = np.around(running_seconds/60, 0)
    print('\nDNN Running Time:', running_minutes, 'minutes')



