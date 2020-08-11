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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve


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

    # model paramters
    epochs         = 100
    learning_rate  = 0.001
    batch_momentum = 0.97
    batch_size     = 100
    dropout_rate   = 0    
    alpha          = 0.3
    n_outputs      = 5
    n_classes      = 5
    random_state   = 42
    ELU_alpha      = 1.0
    digit          = 3
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
    n_inputs       = len(x_columne)

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
    project_dir = r'\\10.39.42.102\temp\2019_MS\AI'
    result_dir  = r'\\10.39.42.102\temp\2019_MS\AI\result'
    log_dir     = r'\\10.39.42.102\temp\2019_MS\AI\log'
    file        = '20190302.csv'

    # ----------------------------------------------------------------------------------
    # run functions
    # ---------------------------------------------------------------------------------- 

    print("Deep Neural Network for PCa grade classification: start...")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    start = timeit.default_timer()
    
    data_path()


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

    print('\noverall test loss:', np.around(score[0], digit))
    print('overall test accuracy:', np.around(score[1], digit))

    plot_ROC()
    plot_PRC()
   
    # ----------------------------------------------------------------------------------
    # print key DNN model parameters
    # ----------------------------------------------------------------------------------
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






# ----------------------------------------------------------------------------------
# plot individual ROC curve
# ----------------------------------------------------------------------------------
#def plot_ROC_individual():

#    fpr[i], tpr[i], roc_auc[i] = ROC()
#    print('Class {0}: AUC = {1:0.2f}'
#             ''.format(i+1, roc_auc[i]))
#    fig = plt.figure()
#    ax = fig.add_subplot(1, 1, 1)
#    ax.set_aspect('equal')
    
#    plt.plot(
#             fpr[i],
#             tpr[i],
# #            color='royalblue',
#             linewidth=3,
#             label='AUC = %0.3f'%roc_auc[i]
#             )
    
#    plt.legend(loc='lower right', prop={'size': 13, 'weight':'bold'})
#    plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
#    plt.xlim([-0.03, 1.0])
#    plt.ylim([0, 1.03])
#    ax.axhline(y=0, color='k', linewidth=3)
#    ax.axhline(y=1.03, color='k', linewidth=3)
#    ax.axvline(x=-0.03, color='k', linewidth=3)
#    ax.axvline(x=1, color='k', linewidth=3)
#   plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
#   plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
#    ax.tick_params(direction='out', length=6, width=2, colors='k', grid_color='k', grid_alpha=0.5)                                                                                     
#    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
#    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)      
#    plt.grid(True)
    #plt.show()
    #plt.savefig(os.path.join(result_dir, lambda x: 'ROC_'+str(x+1)+'.png', format='png', dpi=600))
#    plt.savefig(os.path.join(result_dir, 'ROC_' + str(i+1) + '.png'), format='png', dpi=600)
#    plt.close()


### ----------------------------------------------------------------------------------
### zoom in ROC
### ----------------------------------------------------------------------------------
##def plot_ROC_zoom():
##
##    """
##    plot zoom in view of ROC with five classes and average ROC in one figure
##    """
##    fpr = dict()
##    tpr = dict()
##    roc_auc = dict()
##    threshold = dict()
##    for i in range(n_classes):   
##        fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_pred[:, i]) 
##        roc_auc[i] = auc(fpr[i], tpr[i])
##        
##    # compute micro ROC    
##    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_pred.ravel())                                                                          
##    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
##                 
##    # compute macro ROC
##    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
##    mean_tpr = np.zeros_like(all_fpr)
##    for i in range(n_classes):
##        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
##        
##    mean_tpr /= n_classes
##    fpr['macro'] = all_fpr
##    tpr['macro'] = mean_tpr
##    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])
##
##    # plot ROC curve
##    fig = plt.figure()
##    ax = fig.add_subplot(1, 1, 1)
##    ax.set_aspect('equal')
##    plt.grid(True)
##    plt.plot(
##             fpr['micro'],
##             tpr['micro'],
##             label='Micro Avg (AUC {0:0.3f})'  # decimal numbers = 3
##                   ''.format(roc_auc["micro"]),
##             color='orange',
##             linestyle=':',
##             linewidth=3
##             )
##
##    plt.plot(
##             fpr["macro"],
##             tpr["macro"],
##             label='Macro Avg (AUC {0:0.3f})'    # decimal numbers = 3
##                   ''.format(roc_auc["macro"]),
##             color='navy',
##             linestyle=':',
##             linewidth=3
##             )
##
##    colors = cycle(['aqua', 'red', 'purple', 'royalblue', 'black'])
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
##    plt.xlim(-0.01, 0.3)
##    plt.ylim(0.7, 1.01)
##    ax.axhline(y=0.7, color='k', linewidth=4)
##    ax.axhline(y=1.01, color='k', linewidth=4)
##    ax.axvline(x=-0.01, color='k', linewidth=4)
##    ax.axvline(x=0.3, color='k', linewidth=4) 
##    plt.xticks([0, 0.1, 0.2, 0.3], fontsize=14, fontweight='bold')
##    plt.yticks([0.7, 0.8, 0.9, 1.0], fontsize=14, fontweight='bold')
##    plt.xlabel('False Positive Rate',  fontweight='bold', fontsize=15)
##    plt.ylabel('True Positive Rate',  fontweight='bold', fontsize=15)
##    plt.legend(loc='lower right', prop={'size': 12, 'weight':'bold'})
##
##    ROC_filename = 'ROC_zoom' + '_' + \
##                   str(learning_rate) + '_' + \
##                   str(batch_momentum) + '_' + \
##                   str(epochs) + '_' + \
##                   str(dropout_rate) + \
##                   str(batch_size) + \
##                   strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
##     
##    plt.savefig(
##                os.path.join(result_dir, ROC_filename),
##                format='png',
##                dpi=600
##                )
##
##    plt.show()
##    plt.close()











