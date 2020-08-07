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
from time import gmtime, strftime

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.utils import resample


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
                 class2_ratio,
                 class3_ratio,
                 class4_ratio,
                 x_range
                 ):
                        
        self.file         = file
        self.project_dir  = project_dir
        self.random_state = random_state
        self.train_split  = train_split
        self.test_split   = test_split
        self.class0_ratio = class0_ratio
        self.class1_ratio = class1_ratio
        self.class2_ratio = class2_ratio
        self.class3_ratio = class3_ratio
        self.class4_ratio = class4_ratio
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
    
    def data_balancing(self):

        df = self.data_loading()

        df.loc[df['ROIClass'] == 1, 'y_cat'] = 0
        df.loc[df['ROIClass'] == 2, 'y_cat'] = 1
        df.loc[df['ROIClass'] == 4, 'y_cat'] = 2
        df.loc[df['ROIClass'] == 5, 'y_cat'] = 3
        df.loc[df['ROIClass'] == 6, 'y_cat'] = 4

        class0 = df[df['y_cat'] == 0]
        class0_sample = class0.sample(int(class0.shape[0]*self.class0_ratio))
        
        class1 = df[df['y_cat'] == 1]
        class1_sample = class1.sample(int(class1.shape[0]*self.class1_ratio))
        
        class2 = df[df['y_cat'] == 2]
        class2_sample = class2.sample(int(class2.shape[0]*self.class2_ratio))
        
        class3 = df[df['y_cat'] == 3]
        class3_sample = class3.sample(int(class3.shape[0]*self.class3_ratio))
        
        class4 = df[df['y_cat'] == 4]
        class4_sample = class4.sample(int(class4.shape[0]*self.class4_ratio))

        df_2 = pd.concat(
                         [class0_sample,
                          class1_sample,
                          class2_sample,
                          class3_sample,
                          class4_sample]
                         )

        return df_2

    def dataset_construction(self):
        
        df_2 = self.data_balancing()

        x = df_2.iloc[:, self.x_range]

        y = df_2.y_cat.astype('int')

        # binarize the output
        y_binary = label_binarize(y, classes=[0, 1, 2, 3, 4])

        x_train, x_test_1, y_train, y_test_1 = train_test_split(
                                                                x,
                                                                y_binary,
                                                                test_size=self.train_split,
                                                                random_state=self.random_state,
                                                                stratify=y_binary
                                                                )

        x_val, x_test, y_val, y_test = train_test_split(
                                                        x_test_1,
                                                        y_test_1,
                                                        test_size=self.test_split,
                                                        random_state=self.random_state,
                                                        stratify=y_test_1
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
                        validation_steps=None            
                        )

    score = model.evaluate(x_test, y_test, verbose=0)
                           
    y_pred = model.predict(x_test)
    
    y_pred_classes = model.predict_classes(x_test)

    return score, y_pred, y_pred_classes

# ----------------------------------------------------------------------------------
# ROC curve
# ----------------------------------------------------------------------------------
def ROC_AUC_bootstrap(y_test, y_pred):

    """
    calculate AUC, TPR, TNR with 1000 iterations
    """
    
    AUC  = []
    THRE = []
    TNR  = []
    TPR  = []

    for i in range(n_classes):
        
        aucs  = []
        tprs  = []
        fprs  = []
        tnrs  = []
        thres = []
        
        for j in range(n_bootstrap):
            
            #print("bootstrap iteration: " + str(j+1) + " out of " + str(n_bootstrap))
            
            index = range(len(y_pred[:, i]))
            
            indices = resample(
                               index,
                               replace=True,
                               n_samples=int(len(y_pred[:, i]))
                               )

            fpr, tpr, thre = roc_curve(y_test[indices, i], y_pred[indices, i])

            q = np.arange(len(tpr))
            
            roc = pd.DataFrame(
                               {
                                'fpr' : pd.Series(fpr, index=q),
                                'tpr' : pd.Series(tpr, index=q),
                                'tnr' : pd.Series(1-fpr, index=q),
                                'tf'  : pd.Series(tpr-(1-fpr), index=q),
                                'thre': pd.Series(thre, index=q)
                                }
                               )
            
            roc_opt = roc.loc[(roc['tpr']-roc['fpr']).idxmax(),:]
       
            aucs.append(roc_auc_score(y_test[indices, i], y_pred[indices, i]))
            tprs.append(roc_opt['tpr'])
            tnrs.append(roc_opt['tnr'])
            thres.append(roc_opt['thre'])

        AUC.append(aucs)
        TPR.append(tprs)
        TNR.append(tnrs)

        total = sum([AUC, TPR, TNR], [])
        
    return total
        
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
    
    ROC_AUC = np.array(total)

    for i in range(len(ROC_AUC)):
        
        stat = ROC_AUC[i]
        stat = mean_CI(stat)
        stat_sum.append(stat)
        
    stat_sum = np.array(stat_sum)
    stat_sum = np.round(stat_sum, decimals=3)
    
    return stat_sum

def stat_report():

    stat_sum = stat_summary()

    stat_df = pd.DataFrame(
                           stat_sum,
                           index=[                           
                                  'AUC_1',  
                                  'AUC_2',
                                  'AUC_3', 
                                  'AUC_4', 
                                  'AUC_5',
                                  'TPR_1',
                                  'TPR_2',
                                  'TPR_3',
                                  'TPR_4',
                                  'TPR_5',
                                  'TNR_1',
                                  'TNR_2',
                                  'TNR_3',
                                  'TNR_4',
                                  'TNR_5'
                                  ],                            
                           columns=[
                                    'mean',
                                    '95% CI -',
                                    '95% CI +'
                                    ]
                           )

    filename = str(MRI) + '_' + \
               str(n_bootstrap) + '_' + \
               str(epochs) + '_' + \
               str(strftime("%d-%b-%Y-%H-%M-%S", gmtime())) + \
               '.csv'

    stat_df.to_csv(os.path.join(result_dir, filename))

    return stat_df
    
# ----------------------------------------------------------------------------------
# run the model
# ----------------------------------------------------------------------------------   
if __name__ == '__main__':

    # model paramters
    n_bootstrap    = 1000
    epochs         = 100
    learning_rate  = 0.001
    batch_momentum = 0.97
    batch_size     = 100
    dropout_rate   = 0    
    alpha          = 0.3
    n_outputs      = 5
    n_classes      = 5
    random_state   = 42
    ELU_alpha      = 1.05
    digit          = 3
    train_split    = 0.2
    test_split     = 0.5
    class0_ratio   = 1.0
    class1_ratio   = 1.0
    class2_ratio   = 1.0
    class3_ratio   = 1.0
    class4_ratio   = 1.0
    x_DBSI         = [16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]     
    x_cMRI         = [28, 29]                                       
    x_MTC          = [28, 29, 30]                                   
    x_DTI          = [15, 16, 17, 18, 29, 30]                       

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

    # data and results path 
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

    x_ranges = [x_DBSI, x_DTI, x_MTC, x_cMRI]    
    MRIs     = ['DBSI', 'DTI', 'MTC', 'cMRI']
    
    x_range = []
    MRI     = []

    for x_range, MRI in zip(x_ranges, MRIs):

        n_inputs = len(x_range)

        x_train, x_val, x_test, y_train, y_val, y_test = data(
                                                              file,
                                                              project_dir,
                                                              random_state,
                                                              train_split,
                                                              test_split,
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
             
        score, y_pred, y_pred_classes = model_training()

        total = ROC_AUC_bootstrap(y_test, y_pred)

        stat_df = stat_report()

        print('\nDNN model:', MRI)

        print(stat_df)

    print('\n')
    print('overall test loss:   ', np.around(score[0], 3))
    print('overall test acc:    ', np.around(score[1], 3))
    print('train dataset size:  ', len(x_train))
    print('val dataset size:    ', len(x_val))
    print('test dataset size:   ', len(x_test))
    print('bootstrap iteration: ', n_bootstrap)    
    print('epochs:              ', epochs)
    print('batch size:          ', batch_size)
    print('dropout rate:        ', dropout_rate)
    print('batch momentum:      ', batch_momentum)
    print('learning rate:       ', learning_rate)
    print('neuron numbers:      ', n_neurons)

    stop = timeit.default_timer()
    running_seconds = np.around(stop - start, -1)
    running_minutes = np.around(running_seconds/60, 1)
    print('DNN running time:    ', running_seconds, 'sec')
    print('DNN running time:    ', running_minutes, 'min')



##rng = np.random.RandomState(random_state)
##
##for j in range(n_bootstrap):
##
##    indices = rng.random_integers(0, len(y_pred[:, i])-1, len(y_pred[:, i]))
##    
##    if len(np.unique(y_true[indices, i])) < 2:
##        # We need at least one positive and one negative sample for ROC AUC
##        # to be defined: reject the sample
##        continue
##
##    roc_auc[i] = roc_auc_score(y_true[indices, i], y_pred[indices, i])
##    
##    AUC.append(roc_auc[i])
##    
##    print('Bootstrap #{} ROC AUC: {:0.3f}'.format(j+1, score))










