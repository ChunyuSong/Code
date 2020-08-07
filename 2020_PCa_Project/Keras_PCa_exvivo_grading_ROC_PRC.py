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
import numpy as np
import pandas as pd
import seaborn as sn
import glob2 as glob
import nibabel as nib
from functools import partial
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle

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
from sklearn.preprocessing import label_binarize
from time import gmtime, strftime


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
                 train_file,
                 project_dir,
                 random_state,
                 train_split,
                 test_split,
                 class0_ratio, 
                 class1_ratio,
                 class2_ratio, 
                 class3_ratio,
                 x_input
                 ):
                        
        self.train_file   = train_file
        self.project_dir  = project_dir
        self.random_state = random_state
        self.train_split  = train_split
        self.test_split   = test_split
        self.class0_ratio = class0_ratio
        self.class1_ratio = class1_ratio
        self.class2_ratio = class2_ratio
        self.class3_ratio = class3_ratio
        self.x_input      = x_input

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
        
    def data_loading(self):

        df = pd.read_csv(os.path.join(self.project_dir, self.train_file))
        
        return df
    
    def data_balancing(self):

        df = self.data_loading()

        df.loc[df['ROI_Class'] == 'G1', 'y_cat'] = 0
        df.loc[df['ROI_Class'] == 'G2', 'y_cat'] = 1
        df.loc[df['ROI_Class'] == 'G3', 'y_cat'] = 2
        df.loc[df['ROI_Class'] == 'G5', 'y_cat'] = 3
        df.loc[df['ROI_Class'] == 'G4', 'y_cat'] = 4


        class0        = df[df['y_cat'] == 0]
        class0_sample = class0.sample(int(class0.shape[0]*self.class0_ratio))
        class1        = df[df['y_cat'] == 1]
        class1_sample = class1.sample(int(class1.shape[0]*self.class1_ratio))
        class2        = df[df['y_cat'] == 2]
        class2_sample = class2.sample(int(class2.shape[0]*self.class2_ratio))
        class3        = df[df['y_cat'] == 3]
        class3_sample = class3.sample(int(class3.shape[0]*self.class3_ratio))

        df_2 = pd.concat([
                          class0_sample,
                          class1_sample,
                          class2_sample,
                          class3_sample,
                          ])

        return df_2

    def dataset_construction(self):
        
        df_2 = self.data_balancing()

        X    = df_2.iloc[:, self.x_input]

        Y    = df_2.y_cat.astype('int')

        Y_2  = label_binarize(Y, classes=[0, 1, 2, 3])

        x_train, x_test_1, y_train, y_test_1 = train_test_split(
                                                                X,
                                                                Y_2,
                                                                test_size=self.train_split,
                                                                random_state=self.random_state,
                                                                stratify=Y_2
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

    score = model.evaluate(
                           x_test,
                           y_test,
                           verbose=0
                           )
    
    y_pred       = model.predict(x_test)  
    y_pred_label = np.argmax(y_pred, axis=1)

    test_loss     = score[0]
    test_accuracy = score[1]

    test_loss     = np.around(test_loss, 3)
    test_accuracy = np.around(test_accuracy, 3)
    
    return score, y_pred, y_pred_label, test_loss, test_accuracy

# ----------------------------------------------------------------------------------
# ROC and AUC
# ----------------------------------------------------------------------------------
def DNN_ROC():
    
    fpr       = dict()
    tpr       = dict()
    roc_auc   = dict()
    threshold = dict()

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
        
    colors = cycle(['aqua', 'red', 'purple', 'royalblue'])

    for i, color in zip(range(n_classes), colors):

        fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_pred[:, i])
        
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        print('ROC AUC %.2f' % roc_auc[i])
        
        plt.plot(
                 fpr[i],
                 tpr[i],
                 color=color,
                 linewidth=3,
                 label='AUC %0.2f' % roc_auc[i]
                 )

    plt.xlim([-0.03, 1])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.03, color='k', linewidth=4)
    ax.axvline(x=-0.03, color='k', linewidth=4)
    ax.axvline(x=1, color='k', linewidth=4) 
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
    #plt.xlabel('False Positive Rate', fontweight='bold', fontsize=15)
    #plt.ylabel('True Positive Rate', fontweight='bold', fontsize=15)
    plt.legend(loc='lower right', prop={'size': 14, 'weight': 'bold'}) 
    plt.grid(True)

    ROC_filename = 'ROC' + '_' + \
                str(count) + \
                str(learning_rate) + '_' + \
                str(batch_momentum) + '_' + \
                str(epochs) + '_' + \
                str(dropout_rate) + \
                str(batch_size) + \
                strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
                  
    plt.savefig(os.path.join(result_dir, ROC_filename), format='png', dpi=600)
    #plt.show()
    plt.close()

# ----------------------------------------------------------------------------------
# precision recall curve
# ----------------------------------------------------------------------------------
def DNN_PRC():
    
    precision = dict()
    recall    = dict()
    threshold = dict()
    prc_auc   = []

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
        
    colors = cycle(['aqua', 'red', 'purple', 'royalblue'])

    for i, color in zip(range(n_classes), colors):

        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                            y_pred[:, i])
        
        RP_2D = np.array([recall[i], precision[i]])
        RP_2D = RP_2D[np.argsort(RP_2D[:,0])]

        prc_auc.append(auc(RP_2D[1], RP_2D[0]))
        
        print('PRC AUC %.2f' % auc(RP_2D[1], RP_2D[0]))
                
        plt.plot(
                 recall[i],
                 precision[i],
                 color=color,
                 linewidth=3,
                 label='AUC %0.2f' % prc_auc[i]
                 )

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
                   str(count) + \
                   str(learning_rate) + '_' + \
                   str(batch_momentum) + '_' + \
                   str(epochs) + '_' + \
                   str(dropout_rate) + \
                   str(batch_size) + \
                   strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
     
    plt.savefig(
                os.path.join(result_dir, PRC_filename),
                format='png',
                dpi=600
                )
    
    #plt.show()
    plt.close()

# ----------------------------------------------------------------------------------
# model hyper parameters
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    epochs             = 10
    batch_size         = 200
    alpha              = 0.3
    random_state       = 42
    ELU_alpha          = 1.0
    digit              = 3
    train_split        = 0.3
    test_split         = 0.5
    class0_ratio       = 1
    class1_ratio       = 1
    class2_ratio       = 0.7
    class3_ratio       = 0.12
    count              = 0
    x_input            = range(7, 30)
    n_inputs           = len(x_input)
    n_outputs          = 4
    n_classes          = n_outputs
    
    list_learning_rate = [0.0001, 0.001, 0.005, 0.01]
    list_momentum      = [0.97, 0.98, 0.99]
    dropout_rate       = 0
    list_batch_size    = [100, 200]
    list_epochs        = [20, 50]
    
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
    project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_ex_vivo\Deep_Learning'
    result_dir  = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_ex_vivo\Deep_Learning\grading_ROC\result'
    log_dir     = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_ex_vivo\Deep_Learning\grading_ROC\log'             
    train_file  = 'Gleason.csv'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------
    
    print("Deep Neural Network for PCa grade classification: start...")

    start = timeit.default_timer()
    
    data_path()

    PCa_Data = PCa_data(
                        train_file,
                        project_dir,
                        random_state,
                        train_split,
                        test_split,
                        class0_ratio, 
                        class1_ratio,
                        class2_ratio, 
                        class3_ratio,
                        x_input
                        )

    x_train, x_val, x_test, y_train, y_val, y_test = PCa_Data.dataset_construction()

    total_run = len(list_momentum)*len(list_epochs)*len(list_batch_size)*len(list_learning_rate)
    
    breaking = False

    for i in list_batch_size:
        
        for j in list_momentum:
            
            for k in list_epochs:

                for l in list_learning_rate:

                    count += 1

                    print('\niteration: ' + str(count) + '/' + str(total_run))

                    batch_size     = i
                    batch_momentum = j
                    epochs         = k
                    learning_rate  = l
                        
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
                                                            
                    DNN_ROC()
                    DNN_PRC()

                    print('\noverall test loss:  ', test_loss)
                    print('overall test accuracy:', test_accuracy)
                    
                    print('epochs:        ', epochs)
                    print('batch size:    ', batch_size)
                    print('dropout rate:  ', dropout_rate)
                    print('batch momentum:', batch_momentum)
                    print('learning rate: ', learning_rate)
                    print('neuron numbers:', n_neurons)
                
                    if test_accuracy > 0.8:
                        breaking = True

                if breaking == True:
                    break

            if breaking == True:
                break
            
        if breaking == True:
            break

    print('train size:     ', len(x_train))
    print('validation size:', len(x_val))
    print('test size:      ', len(x_test))       
    stop = timeit.default_timer()
    running_seconds = np.around(stop - start, 0)
    running_minutes = np.around(running_seconds/60, 0)
    print('DNN running time:', running_seconds, 'seconds')
    print('DNN running time:', running_minutes, 'minutes')




### ----------------------------------------------------------------------------------
### plot summed ROC curves zoom in view
### ----------------------------------------------------------------------------------
##
##def DNN_ROC_Zoom():
##    
##    # Zoom in view of the upper left corner.
##    fig = plt.figure()
##    ax = fig.add_subplot(1, 1, 1)
##    ax.set_aspect('equal')
##    plt.grid(True)
##    plt.plot(
##             fpr["micro"],
##             tpr["micro"],
##             label='Micro Avg (AUC {0:0.2f})'
##                   ''.format(roc_auc["micro"]),
##             color='orange',
##             linestyle=':',
##             linewidth=3
##             )
##
##    plt.plot(
##             fpr["macro"],
##             tpr["macro"],
##             label='Macro Avg (AUC {0:0.2f})'
##                   ''.format(roc_auc["macro"]),
##             color='navy',
##             linestyle=':',
##             linewidth=3
##             )
##
##    colors = cycle(['aqua', 'red', 'purple', 'royalblue'])
##
##    for i, color in zip(range(n_classes), colors):
##        plt.plot(
##                 fpr[i],
##                 tpr[i],
##                 color=color,
##                 linewidth=3,
##                 #label='Class {0} (area = {1:0.2f})'
##                 #''.format(i+1, roc_auc[i])
##                 )
##        
##    #plt.title('ROC', fontweight='bold', fontsize=14)
##    #plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
##    plt.xlim(0, 0.41)
##    plt.ylim(0.6, 1.01)
##    ax.axhline(y=0.6, color='k', linewidth=4)
##    ax.axhline(y=1.01, color='k', linewidth=4)
##    ax.axvline(x=0, color='k', linewidth=4)
##    ax.axvline(x=0.41, color='k', linewidth=4) 
##    plt.xticks([0, 0.1, 0.2, 0.3, 0.4], fontsize=14, fontweight='bold')
##    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=14, fontweight='bold')
##    plt.xlabel('False Positive Rate',  fontweight='bold', fontsize=15)
##    plt.ylabel('True Positive Rate',  fontweight='bold', fontsize=15)
##    plt.legend(loc='lower right', prop={'size': 13, 'weight':'bold'})
##    #plt.show()
##    plt.savefig(os.path.join(result_dir, 'ROC_sum_2' + '.png'), format='png', dpi=600)
##    plt.close()

