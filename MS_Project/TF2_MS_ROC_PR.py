#----------------------------------------------------------------------
# deep learning classifier using a multiple layer perceptron (MLP)
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

##from keras import initializers
##from keras.models import Sequential, Model
##from keras.layers import Input, Dense, Reshape, Activation, Dropout
##from keras.optimizers import RMSprop
##from keras.layers.normalization import BatchNormalization
##from keras.wrappers.scikit_learn import KerasClassifier
##from keras.layers.advanced_activations import ELU, LeakyReLU

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
#from __future__ import absolute_import, division, print_function, unicode_literals


from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve

from absl import logging
logging._warn_preinit_stderr = 0
logging.warning('Worrying Stuff')


# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
def data_path(result_dir, log_dir):

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
                 x_columne
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
        self.x_columne    = x_columne
        
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

        df_2 = pd.concat([class0_sample, class1_sample, class2_sample, class3_sample, class4_sample])

        return df_2

    def dataset_construction(self):
        
        df_2 = self.data_balancing()

        X = df_2.iloc[:, self.x_columne]

        Y = df_2.y_cat.astype('int')

        # binarize the output
        Y_binary = label_binarize(Y, classes=[0, 1, 2, 3, 4])

        x_train, x_test_1, y_train, y_test_1 = train_test_split(
                                                                X,
                                                                Y_binary,
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
class tf_model(object):
    
    def __init__(
                 self,
                 n_input,
                 n_output,
                 activation,
                 kernel_init,
                 dropout_rate,
                 momentum,
                 output_activation,
                 **kwargs
                 ):
                 
        super(tf_model, self).__init__(**kwargs)
        
        self.n_input           = n_input
        self.n_output          = n_output
        self.activation        = activation
        self.kernel_init       = kernel_init
        self.dropout_rate      = dropout_rate
        self.momentum          = momentum
        self.output_activation = output_activation

        layer_Dense = partial(
                              Dense,
                              kernel_initializer=self.kernel_init, 
                              use_bias=False,
                              activation=None,
                              kernel_regularizer=None
                              )

        layer_BatchNorm = partial(
                                  BatchNormalization,
                                  axis=-1,
                                  momentum=self.momentum,
                                  epsilon=0.001,
                                  beta_initializer='zeros',
                                  gamma_initializer='ones',
                                  beta_regularizer=None,
                                  gamma_regularizer=None
                                  )

        layer_Dropout = partial(
                                Dropout,
                                self.dropout_rate,
                                noise_shape=None,
                                seed=None
                                )

        model = tf.keras.Sequential()
                                                                
        # input layer                              
        model.add(Dense(self.n_input, input_dim=self.n_input))
        model.add(layer_BatchNorm())
        model.add(Activation('relu'))
        model.add(layer_Dropout())

        # hidden layer 1
        model.add(layer_Dense(n_hidden1))
        model.add(layer_BatchNorm())
        model.add(Activation('relu'))
        model.add(layer_Dropout())

        # hidden layer 2
        model.add(layer_Dense(n_hidden1))
        model.add(layer_BatchNorm())
        model.add(Activation('relu'))
        model.add(layer_Dropout())

        # hidden layer 3
        model.add(layer_Dense(n_hidden1))
        model.add(layer_BatchNorm())
        model.add(Activation('relu'))
        model.add(layer_Dropout())

        # hidden layer 4
        model.add(layer_Dense(n_hidden1))
        model.add(layer_BatchNorm())
        model.add(Activation('relu'))
        model.add(layer_Dropout())

        # output layer
        model.add(layer_Dense(n_hidden1))
        model.add(layer_BatchNorm())
        model.add(Activation(self.output_activation))

        self.model = model

    def call(self, x):

        x = self.model(x)

        return x

# ----------------------------------------------------------------------------------
# trainning DNN model
# ----------------------------------------------------------------------------------
def model_training():

    model = tf.model()

    model.compile(
                  loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy']
                  )

    model.summary()
    
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
                        validation_steps=None            
                        )

    score = model.evaluate(x_test, y_test, verbose=0)
                           
    y_pred = model.predict(x_test)
    
    y_pred_classes = model.predict_classes(x_test)

    return score, y_pred, y_pred_classes

# ----------------------------------------------------------------------------------
# ROC curve
# ----------------------------------------------------------------------------------
def plot_ROC():

    """
    plot ROC with five classes in one figure
    """
    fpr       = dict()
    tpr       = dict()
    roc_auc   = dict()
    threshold = dict()
    
    for i in range(n_classes):
        
        fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_pred[:, i])
        
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    colors = cycle(['aqua', 'red', 'purple', 'royalblue', 'black'])
    
    for i, color in zip(range(n_classes), colors):
        
        plt.plot(
                 fpr[i],
                 tpr[i],
                 color=color,
                 linewidth=3,
                 #label='Class {0} (AUC {1:0.3f})'
                 label='AUC {1:0.2f}'              
                 ''.format(i+1, roc_auc[i])
                 )

    #plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
    plt.xlim([-0.03, 1])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.03, color='k', linewidth=4)
    ax.axvline(x=-0.03, color='k', linewidth=4)
    ax.axvline(x=1, color='k', linewidth=4) 
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    #plt.xlabel('False Positive Rate', fontweight='bold', fontsize=16)
    #plt.ylabel('True Positive Rate', fontweight='bold', fontsize=16)
    plt.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'}) 
    plt.grid(True)

    ROC_filename = 'ROC' + '_' + \
                   str(DNN_Model) + '_' + \
                   strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
     
    plt.savefig(
                os.path.join(result_dir, ROC_filename),
                format='png',
                dpi=600
                )

    plt.show()
    plt.close()

# ----------------------------------------------------------------------------------
# precision recall curve
# ----------------------------------------------------------------------------------
def plot_PRC():
    
    precision = dict()
    recall    = dict()
    prc_auc   = []
    
    for i in range(n_classes):
        
        precision[i], recall[i], _ = precision_recall_curve(
                                                            y_test[:, i],
                                                            y_pred[:, i]
                                                            )

        RP_2D = np.array([recall[i], precision[i]])
        RP_2D = RP_2D[np.argsort(RP_2D[:,0])]

        prc_auc.append(auc(RP_2D[1], RP_2D[0]))
        
        print('area=%.2f' % auc(RP_2D[1], RP_2D[0]))    
        
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    colors = cycle(['aqua', 'red', 'purple', 'royalblue', 'black'])
    
    for i, color in zip(range(n_classes), colors):
        
        plt.plot(                       
                 recall[i],
                 precision[i],
                 color=color,
                 linewidth=3,
                 label='Area {1:0.2f}'              
                 ''.format(i+1, prc_auc[i])
                 )

    #plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
    plt.xlim([0, 1.03])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.03, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=1.03, color='k', linewidth=4) 
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
    #plt.xlabel('recall', fontweight='bold', fontsize=16)
    #plt.ylabel('precision', fontweight='bold', fontsize=16)
    plt.legend(loc='lower left', prop={'size': 14, 'weight': 'bold'}) 
    plt.grid(True)

    PRC_filename = 'PRC' + '_' + \
                   str(DNN_Model) + '_' + \
                   strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
     
    plt.savefig(
                os.path.join(result_dir, PRC_filename),
                format='png',
                dpi=600
                )
    

    plt.show()
    plt.close()


# ----------------------------------------------------------------------------------
# run the model
# ----------------------------------------------------------------------------------   
if __name__ == '__main__':

    epochs         = 100
    learning_rate  = 0.001
    momentum       = 0.97
    batch_size     = 100
    dropout_rate   = 0    
    n_output       = 5
    n_classes      = 5
    random_state   = 42
    train_split    = 0.2
    test_split     = 0.5
    class0_ratio   = 1.0
    class1_ratio   = 1.0
    class2_ratio   = 1.0
    class3_ratio   = 1.0
    class4_ratio   = 1.0
    x_col_DBSI     = [16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]     
    x_col_cMRI     = [28, 29]                                       
    x_col_MTC      = [28, 29, 30]                                   
    x_col_DTI      = [15, 16, 17, 18, 29, 30]                       

    x_columne      = x_col_cMRI
    DNN_Model      = 'cMRI'
    n_input        = len(x_columne)

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
    kernel_init       = tf.keras.initializers.he_uniform()          
    optimizer         = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)               
    loss              = tf.keras.losses.CategoricalCrossentropy()
    output_activation = 'softmax'
    activation        = tf.keras.activations.elu

    '''
    keranl initializer: 'he_uniform', 'lecun_normal', 'lecun_uniform'
    optimizer function: 'adam', 'adamax', 'nadam', 'sgd'
    loss function: 'categorical_crossentropy'
    activation function: LeakyReLU(alpha=alpha)
    '''

    # data and results path 
    project_dir = r'\\10.39.42.102\temp\2019_MS\AI'
    result_dir  = r'\\10.39.42.102\temp\2019_MS\AI\result'
    log_dir     = r'\\10.39.42.102\temp\2019_MS\AI\log'
    file        = '20190302.csv'

    # ----------------------------------------------------------------------------------
    # run functions
    # ---------------------------------------------------------------------------------- 

    print("Deep Neural Network for PCa grade classification: start...")

    #os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.random.set_seed(22)
    np.random.seed(22)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    start = timeit.default_timer()

    data_path(result_dir, log_dir)


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
                                                          x_columne
                                                          ).dataset_construction()


    model = tf_model(
                     n_input,
                     n_output,
                     kernel_init,
                     activation,
                     dropout_rate,
                     momentum,
                     output_activation
                     )
         
    score, y_pred, y_pred_classes = model_training()

    print('\noverall test loss:', np.around(score[0], digit))
    print('overall test accuracy:', np.around(score[1], digit))

    plot_ROC()
    plot_PRC()

    print('key model parameters:')
    print('train dataset size:  ', len(x_train))
    #print('val dataset size:    ', np.around(len(x_train)*val_split, 0))  
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



    












