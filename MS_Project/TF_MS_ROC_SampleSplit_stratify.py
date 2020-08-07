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
import random
import tensorflow
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
import nibabel as nib
import matplotlib.pyplot as plt

from functools import partial
from datetime import datetime
from scipy import interp
from itertools import cycle
from time import gmtime, strftime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import elu, relu
from tensorflow.keras.metrics import categorical_accuracy

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

from imblearn.over_sampling import SMOTE, SVMSMOTE, SMOTENC
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

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
class MS_data(object):
    
    '''
    calculate DNN statisical results
    '''
    
    def __init__(
                 self,
                 train_file,
                 project_dir,
                 random_state,
                 val_split, 
                 x_input,
                 test_split
                 ):
                        
        self.train_file   = train_file
        self.project_dir  = project_dir
        self.random_state = random_state
        self.val_split    = val_split
        self.x_input      = x_input
        self.test_split   = test_split

    def map_list():

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
        
        return map_list
        
    def data_loading(self):

        df = pd.read_csv(os.path.join(self.project_dir, self.train_file))
        
        return df
    
    def data_balancing(self):

        df = self.data_loading()

        df.loc[df['ROIClass'] == 1, 'y_cat'] = 0
        df.loc[df['ROIClass'] == 2, 'y_cat'] = 1
        df.loc[df['ROIClass'] == 4, 'y_cat'] = 2
        df.loc[df['ROIClass'] == 5, 'y_cat'] = 3
        df.loc[df['ROIClass'] == 6, 'y_cat'] = 4

        class1 = df[df['y_cat'] == 0]
        class2 = df[df['y_cat'] == 1]
        class3 = df[df['y_cat'] == 2]
        class4 = df[df['y_cat'] == 3]
        class5 = df[df['y_cat'] == 4]

        df2 = pd.concat([class1, class2, class3, class4, class5]) 
                                                                 
        return df2

    def data_sample_split(self):
    
        df2 = self.data_balancing()

        df_class = df2['ROIClass'].unique()
        
        df_test_list  = []
        df_train_list = []
        
        for roi in df_class:

            df_roi = df2[df2['ROIClass'] == roi]
            roi_id_unique = df_roi['ID'].unique()

            df_roi_test = np.random.choice(
                                           list(roi_id_unique),
                                           size=int(len(roi_id_unique)/self.test_split),
                                           replace=False
                                           )
            df_test_list.append(df_roi[df_roi['ID'].isin(df_roi_test)])
            df_train_list.append(df_roi[~df_roi['ID'].isin(df_roi_test)])
            
        df_test  = pd.concat(df_test_list)
        df_train = pd.concat(df_train_list)
        
        x_test = df_test.iloc[:, self.x_input].values
        y_test = df_test.y_cat.astype('int').values
        y_test = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
        
        x_train_val = df_train.iloc[:, self.x_input].values
        y_train_val = df_train.y_cat.astype('int').values
        y_train_val = label_binarize(y_train_val, classes=[0, 1, 2, 3, 4])

        x_train, x_val, y_train, y_val = train_test_split(
                                                          x_train_val,
                                                          y_train_val,
                                                          test_size=self.val_split,
                                                          random_state=self.random_state
                                                          )

        return x_train, x_val, x_test, y_train, y_val, y_test

    def data_oversample(self):

        x_train, x_val, x_test, y_train, y_val, y_test = self.data_sample_split()
        
        print("Presampled train dataset: %s" % Counter(y_train[:, 0]))

        resample = SMOTE(random_state=42)          # SVMSMOTE, SMOTENC
        
        x_train, y_train = resample.fit_resample(x_train, y_train)
        x_val, y_val     = resample.fit_resample(x_val, y_val)

        print("Resampled train dataset: %s" % Counter(y_train[:, 0]))
        
##        x_test, y_test   = resample.fit_resample(x_test, y_test)
        
        return x_train, x_val, x_test, y_train, y_val, y_test
  
# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# ----------------------------------------------------------------------------------
class tensorflow_model(object):
    
    def __init__(
                 self,
                 kernel_init,
                 dropout_rate,
                 momentum,
                 n_input,
                 n_output,
                 activation,
                 output_activation,
                 n_node,
                 n_layer
                 ):
                 
        self.kernel_init       = kernel_init
        self.dropout_rate      = dropout_rate
        self.momentum          = momentum
        self.n_input           = n_input
        self.n_output          = n_output
        self.activation        = activation
        self.output_activation = output_activation
        self.n_node            = n_node
        self.n_layer           = n_layer
           
    def build_model(self):
    
        model = Sequential()

        layer_dense = partial(
                              Dense,
                              kernel_initializer=self.kernel_init, 
                              use_bias=False,
                              activation=None
                              )

        layer_BN = partial(
                           BatchNormalization,
                           axis=-1,
                           momentum=self.momentum,
                           epsilon=0.001,
                           beta_initializer='zeros',
                           gamma_initializer='ones',
                           beta_regularizer=None,
                           gamma_regularizer=None                             
                           )

        layer_dropout = partial(
                                Dropout,
                                self.dropout_rate,
                                noise_shape=None,
                                seed=None
                                )
                                                                  
        # input layer                              
        model.add(layer_dense(self.n_input, input_dim=self.n_input))
        model.add(layer_BN())
        model.add(Activation(self.activation))
        model.add(layer_dropout())

        for i in range(self.n_layer):

            # hidden layer
            model.add(layer_dense(self.n_node))
            model.add(layer_BN())
            model.add(Activation(self.activation))
            model.add(layer_dropout())
            
        # output layer
        model.add(layer_dense(self.n_output))
        model.add(layer_BN())
        model.add(Activation(self.output_activation))

        #model.summary()
     
        return model

# ----------------------------------------------------------------------------------
# trainning DNN model
# ----------------------------------------------------------------------------------
class model_training(object):

    def __init__(
                 self,
                 loss,
                 optimizer,
                 batch_size,
                 epoch,
                 verbose,
                 model
                 ):
             
        self.loss       = loss
        self.optimizer  = optimizer
        self.batch_size = batch_size
        self.epoch      = epoch
        self.verbose    = verbose
        self.model      = model

    def train(self):

        self.model.compile(
                           loss=self.loss,
                           optimizer=self.optimizer,
                           metrics=['accuracy']
                           )

        history = self.model.fit(
                                 x=x_train,
                                 y=y_train,
                                 batch_size=self.batch_size,
                                 epochs=self.epoch,
                                 verbose=self.verbose,
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

        score = self.model.evaluate(
                                    x_test,
                                    y_test,
                                    verbose=0
                                    )

        y_pred       = self.model.predict(x_test)
        y_pred_class = self.model.predict_classes(x_test)

        test_loss = np.around(score[0], 3)
        test_acc  = np.around(score[1], 3)

        return y_pred, y_pred_class, test_loss, test_acc

# ----------------------------------------------------------------------------------
# ROC and AUC
# ----------------------------------------------------------------------------------
def DNN_ROC():

    print('ROC:')
    
    fpr       = dict()
    tpr       = dict()
    roc_auc   = dict()
    threshold = dict()

##    fig = plt.figure()
##    ax  = fig.add_subplot(1, 1, 1)
##    ax.set_aspect('equal')
        
    colors = cycle(['aqua', 'red', 'purple', 'royalblue'])

    for i, color in zip(range(n_class), colors):

        fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_pred[:, i])
        
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        print('ROC AUC %.3f' % roc_auc[i])
        
##        plt.plot(
##                 fpr[i],
##                 tpr[i],
##                 color=color,
##                 linewidth=3,
##                 label='AUC %0.2f' % roc_auc[i]
##                 )
##
##    plt.xlim([-0.03, 1])
##    plt.ylim([0, 1.03])
##    ax.axhline(y=0, color='k', linewidth=4)
##    ax.axhline(y=1.03, color='k', linewidth=4)
##    ax.axvline(x=-0.03, color='k', linewidth=4)
##    ax.axvline(x=1, color='k', linewidth=4) 
##    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
##    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
##    #plt.xlabel('False Positive Rate', fontweight='bold', fontsize=15)
##    #plt.ylabel('True Positive Rate', fontweight='bold', fontsize=15)
##    plt.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'}) 
##    plt.grid(True)
##
##    ROC_filename = 'ROC' + '_' + \
##                str(count) + \
##                str(learning_rate) + '_' + \
##                str(momentum) + '_' + \
##                str(epoch) + '_' + \
##                str(dropout_rate) + \
##                str(batch_size) + \
##                strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
##                  
##    plt.savefig(os.path.join(result_dir, ROC_filename), format='png', dpi=600)
##    plt.show()
##    plt.close()


# ----------------------------------------------------------------------------------
# precision recall curve
# ----------------------------------------------------------------------------------
def DNN_PRC():

    print('PRC:')
    
    precision = dict()
    recall    = dict()
    threshold = dict()
    prc_auc   = []

##    fig = plt.figure()
##    ax  = fig.add_subplot(1, 1, 1)
##    ax.set_aspect('equal')
        
    colors = cycle(['aqua', 'red', 'purple', 'royalblue'])

    for i, color in zip(range(n_class), colors):

        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_pred[:, i])                                                
        
        RP_2D = np.array([recall[i], precision[i]])
        RP_2D = RP_2D[np.argsort(RP_2D[:,0])]

        prc_auc.append(auc(RP_2D[1], RP_2D[0]))

        print('PRC AUC %.3f' % auc(RP_2D[1], RP_2D[0]))
                
##        plt.plot(
##                 recall[i],
##                 precision[i],
##                 color=color,
##                 linewidth=3,
##                 label='AUC %0.2f' % prc_auc[i]
##                 )
##
##    plt.xlim([0, 1.03])
##    plt.ylim([0, 1.03])
##    ax.axhline(y=0, color='k', linewidth=4)
##    ax.axhline(y=1.03, color='k', linewidth=4)
##    ax.axvline(x=0, color='k', linewidth=4)
##    ax.axvline(x=1.03, color='k', linewidth=4) 
##    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
##    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=16, fontweight='bold')
##    #plt.xlabel('recall', fontweight='bold', fontsize=16)
##    #plt.ylabel('precision', fontweight='bold', fontsize=16)
##    plt.legend(loc='lower left', prop={'size': 14, 'weight': 'bold'}) 
##    plt.grid(True)
##
##    PRC_filename = 'PRC' + '_' + \
##                   str(count) + \
##                   str(learning_rate) + '_' + \
##                   str(momentum) + '_' + \
##                   str(epoch) + '_' + \
##                   str(dropout_rate) + \
##                   str(batch_size) + \
##                   strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
##     
##    plt.savefig(os.path.join(result_dir, PRC_filename), format='png', dpi=600)
##                
##    plt.show()
##    plt.close()

# ----------------------------------------------------------------------------------
# model hyper parameters
# ----------------------------------------------------------------------------------

if __name__ == '__main__':

    test_split        = 10
    n_node            = 100
    n_layer           = 10
    random_state      = 42
    verbose           = 0
    digit             = 3
    val_split         = 0.1
    count             = 0
    x_input_DBSI      = [16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]     
    x_input_cMRI      = [28, 29]                                       
    x_input_MTC       = [28, 29, 30]                                   
    x_input_DTI       = [15, 16, 17, 18, 29, 30]                       
    x_input           = x_input_DBSI
    DNN_Model         = 'DBSI'
    n_input           = len(x_input)
    n_output          = 5
    n_class           = n_output

    list_LR           = [0.1, 0.01, 0.001]            
    list_BM           = [0.97, 0.99]            
    list_DR           = [0]
    list_BS           = [200, 100]
    list_EP           = [10, 20]
    list_NN           = [100]
    list_seed         = [21, 5, 23, 15, 16, 14, 13] #range(50)
    
    kernel_init       = 'he_uniform' 
    optimizer         = 'adam'          
    loss              = 'categorical_crossentropy'  # 'sparse_categorical_crossentropy'
    output_activation = 'softmax'
    activation        = 'elu'                       # relu, leaky_relu
    
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
    train_file  = '20190302.csv'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------
    
    print("Deep Neural Network for PCa grade classification: start...")
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2.')

    start = timeit.default_timer()
    
    data_path()

    breaking = False

    total_run = len(list_BM)*len(list_EP)*len(list_BS)*len(list_LR)*len(list_seed)

    for n_seed in list_seed:

        for batch_size in list_BS:
            
            for momentum in list_BM:
                
                for epoch in list_EP:

                    for learning_rate in list_LR:

                        for dropout_rate in list_DR:

                            count += 1
                            print('\nRunning times: ' + str(count) + '/' + str(total_run))

                            np.random.seed(seed=n_seed)
                          
                            x_train, x_val, x_test, y_train, y_val, y_test = MS_data(
                                                                                      train_file,
                                                                                      project_dir,
                                                                                      random_state,
                                                                                      val_split,
                                                                                      x_input,
                                                                                      test_split
                                                                                      ).data_oversample()

                            model = tensorflow_model(
                                                     kernel_init,
                                                     dropout_rate,
                                                     momentum,
                                                     n_input,
                                                     n_output,
                                                     activation,
                                                     output_activation,
                                                     n_node,
                                                     n_layer
                                                     ).build_model()
                            
                            y_pred, y_pred_class, test_loss, test_acc = model_training(
                                                                                       loss,
                                                                                       optimizer,
                                                                                       batch_size,
                                                                                       epoch,
                                                                                       verbose,
                                                                                       model
                                                                                       ).train()

                            
                            DNN_ROC()
                            DNN_PRC()

                            print('model parameters:')
                            print('seed number:   ', str(n_seed))
                            print('test loss:     ', test_loss)
                            print('test acc:      ', test_acc)
                            print('test split:    ', 1/test_split)
                            print('epochs:        ', epoch)
                            print('batch size:    ', batch_size)
                            print('dropout rate:  ', dropout_rate)
                            print('batch momentum:', momentum)
                            print('learning rate: ', learning_rate)
                            print('neuron numbers:', n_node)
                            print('layer numbers: ', n_layer)
                    
                            if test_acc > 0.95:
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
            
    print("train size:", len(x_train))
    print("val size:  ", len(x_val))
    print("test size: ", len(x_test))
        
    stop = timeit.default_timer()
    running_seconds = np.around(stop-start, 0)
    running_minutes = np.around(running_seconds/60, 0)
    print('\nDNN Running Time:', running_minutes, 'minutes')


