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
                 test_split,
                 class0_ratio, 
                 class1_ratio,
                 x_columne
                 ):
                        
        self.file         = file
        self.project_dir  = project_dir
        self.random_state = random_state
        self.train_split  = train_split
        self.test_split   = test_split
        self.class0_ratio = class0_ratio
        self.class1_ratio = class1_ratio
        self.x_columne    = x_columne
        
    def data_loading(self):

        df = pd.read_csv(os.path.join(self.project_dir, self.file))
        
        return df
    
    def data_balancing(self):

        df = self.data_loading()

        #df.loc[df['ROI_Class'].isin(['BPH', 'BPZ', 'SBPH']), 'y_cat'] = 0
        #df.loc[df['ROI_Class'] == 'BPH', 'y_cat'] = 0
        df.loc[df['ROI_Class'] == 'BPZ', 'y_cat'] = 0
        
        #df.loc[df['ROI_Class'] == 'SBPH', 'y_cat'] = 0
        
        df.loc[df['ROI_Class'] == 'PCa', 'y_cat'] = 1

        class0 = df[df['y_cat'] == 0]
        class0_sample = class0.sample(int(class0.shape[0]*self.class0_ratio))
        
        class1 = df[df['y_cat'] == 1]
        class1_sample = class1.sample(int(class1.shape[0]*self.class1_ratio))

        df_2 = pd.concat([class0_sample, class1_sample])

        return df_2

    def dataset_construction(self):
        
        df_2 = self.data_balancing()

        X = df_2.iloc[:, self.x_columne]

        Y = df_2.y_cat.astype('int')

        x_train, x_test_1, y_train, y_test_1 = train_test_split(
                                                                X,
                                                                Y,
                                                                test_size=self.train_split,
                                                                random_state=self.random_state
                                                                )

        x_val, x_test, y_val, y_test = train_test_split(
                                                        x_test_1,
                                                        y_test_1,
                                                        test_size=self.test_split,
                                                        random_state=self.random_state
                                                        )

        return x_train, x_val, x_test, y_train, y_val, y_test
    
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

        model.summary()

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
                        verbose=2,
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
    
    y_pred = model.predict(x_test)  
    y_pred_label = np.argmax(y_pred, axis=1)
    
    return score, y_pred, y_pred_label

# ----------------------------------------------------------------------------------
# calculate confusion matrix and nornalized confusion matrix
# ----------------------------------------------------------------------------------
def con_mat(y_pred_label):
    
    cm = confusion_matrix(y_test, y_pred_label)
    cm = np.around(cm, 2)
    
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
                    annot_kws={'size': 20, 'fontweight': 'bold'},
                    cmap="Blues",
                    fmt=fmt,
                    linewidths=0.5
                    )

    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=2, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=2, color='k', linewidth=4)

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

        ACC_ = np.around(ACC, self.digit)      
        TPR_ = np.around(TPR, self.digit)
        TNR_ = np.around(TNR, self.digit)       
        PPV_ = np.around(PPV, self.digit)    
        NPV_ = np.around(NPV, self.digit)
        FPR_ = np.around(FPR, self.digit)
        FNR_ = np.around(FNR, self.digit)
        FDR_ = np.around(FDR, self.digit)
          
        return ACC_, TPR_, TNR_, PPV_, NPV_, FPR_, FNR_, FDR_ 

# ----------------------------------------------------------------------------------
# run the model
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    epochs         = 10
    learning_rate  = 0.001
    batch_momentum = 0.97
    batch_size     = 200
    dropout_rate   = 0  
    alpha          = 0.3
    random_state   = 42
    ELU_alpha      = 1.0
    digit          = 3
    train_split    = 0.2
    test_split     = 0.5
    class0_ratio   = 1.0
    class1_ratio   = 0.44
    x_columne      = range(6,31)
    n_inputs       = len(x_columne)
    n_outputs      = 2
    
    # DNN hidden layerss
    n_neurons  = 100
    n_hidden1  = n_neurons
    n_hidden2  = n_neurons
    n_hidden3  = n_neurons
    n_hidden4  = n_neurons
    n_hidden5  = n_neurons
    n_hidden6  = n_neurons
    n_hidden7  = n_neurons
    n_hidden8  = n_neurons
    n_hidden9  = n_neurons
    n_hidden10 = n_neurons

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

    # data and results path 
    project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_ex_vivo\ex_vivo_PCa_results'
    result_dir  = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_ex_vivo\ex_vivo_PCa_results'
    log_dir     = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_ex_vivo\ex_vivo_PCa_results\log'
             
    file           = 'PCa_DWI.csv'
    DBSI_folder    = 'DBSI_results_0.1_0.1_0.8_0.8_2.3_2.3'
   
    # running model
    start = timeit.default_timer()
    print("Deep Neural Network for PCa grade classification: start...")
    
    data_path()

    data = data(
                file,
                project_dir,
                random_state,
                train_split,
                test_split,
                class0_ratio, 
                class1_ratio,
                x_columne
                )

    x_train, x_val, x_test, y_train, y_val, y_test = data.dataset_construction()
        
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
    
    score, y_pred, y_pred_label = model_training()
    
    cm, cm_norm = con_mat(y_pred_label)
    
    ACC_, TPR_, TNR_, PPV_, NPV_, FPR_, FNR_, FDR_ = model_evaluation(cm, digit).statistics_2()
    
    plot_CM(cm_norm, '')

    plot_CM(cm, 'd')

    # ----------------------------------------------------------------------------------
    # confusion matrix, sensitivity, specificity, presicion, f-score, model parameters
    # ----------------------------------------------------------------------------------
    print('\noverall test loss:', np.around(score[0], digit))
    print('overall test accuracy:', np.around(score[1], digit))
    print("confusion matrix:")
    print(cm)
    print("normalized confusion matrix:")
    print(cm_norm) 
    print("Benign, PCa:")   
    print("sensitivity:", TPR_)
    print("specificity:", TNR_)
    print("precision:", PPV_)
    print("negative predictive value:", NPV_)
    print("false positive rate:", FPR_)
    print("false negative rate:", FNR_)
    print("false discovery rate:", FDR_)
    print("precision, recall, f1-score, support:")
    print(classification_report(y_test, y_pred_label, digits=digit))       

    
    print("key model parameters:")
    print("train dataset size:", len(x_train))
    print("validation dataset size:", len(x_val))
    print("test dataset size:", len(x_test))
    print("epochs:", epochs)
    print("batch size:", batch_size)
    print("dropout rate:", dropout_rate)
    print("batch momentum:", batch_momentum)
    print("initial learning rate:", learning_rate)
    print("hidden layer neuron numbers:", n_neurons )
    stop = timeit.default_timer()
    print('DNN Running Time:', np.around(stop - start, 0), 'seconds')


