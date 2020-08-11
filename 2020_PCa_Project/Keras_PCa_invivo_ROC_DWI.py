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

    def DBSI_map_list():
        
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

        df = pd.read_csv(os.path.join(self.project_dir, self.file))
        
        return df

    def replace_model(self):
        
        df_list = []

        for i in ['p','c','benign']:
            df = self.data_loading()

            df['y_cat'] = 2 # preset y_cat to be 2 then pass new values
            df.loc[df['ROI_Name'] == 't', 'y_cat'] = 1
        
            if i == 'benign':
            	df.loc[df['ROI_Name'].isin(['p', 'c']), 'y_cat'] = 0
            else:
            	df.loc[df['ROI_Name'] == i, 'y_cat'] = 0

            df_list.append(df)

        return df_list
    
    def data_balancing(self):
        
        df_list = self.replace_model()
        new_list = []
        
        for df in df_list:
            class0 = df[df['y_cat'] == 0]
            class0_sample = class0.sample(int(class0.shape[0]*self.class0_ratio))

            class1 = df[df['y_cat'] == 1]
            class1_sample = class1.sample(int(class1.shape[0]*self.class1_ratio))
            
            df_2 = pd.concat([class0_sample, class1_sample])
            new_list.append(df_2)
            
        return new_list

    def dataset_construction(self):
        
        df_list = self.data_balancing()
        
        x_train_list = []
        x_val_list = []
        x_test_list = []
        y_train_list = []
        y_val_list = []
        y_test_list = []

        for df_2 in df_list:
            
            x = df_2.iloc[:, self.x_columne]
            y = df_2[['y_cat']].astype('int')

            x_train, x_test_, y_train, y_test_ = train_test_split(
                                                                  x,
                                                                  y,
                                                                  test_size=self.train_split,
                                                                  random_state=self.random_state
                                                                  )

            x_val, x_test, y_val, y_test = train_test_split(
                                                            x_test_,
                                                            y_test_,
                                                            test_size=self.test_split,
                                                            random_state=self.random_state
                                                            )

            x_train_list.append(x_train)
            x_val_list.append(x_val)
            x_test_list.append(x_test)
            y_train_list.append(y_train)
            y_val_list.append(y_val)
            y_test_list.append(y_test)
		
        return x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list
    
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
        self.activation     = activation
        self.dropout_rate   = dropout_rate
        self.batch_momentum = batch_momentum
        self.n_inputs       = n_inputs
        self.n_outputs      = n_outputs
         
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
# train model
# ----------------------------------------------------------------------------------
def train_model(x_train, x_val, x_test, y_train, y_val, y_test):
    
    history = model.fit(
                        x=x_train,
                        y=y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        callbacks=None,
                        validation_split=None,
                        shuffle=True,
                        class_weight=None,
                        sample_weight=None,
                        initial_epoch=0,
                        steps_per_epoch=None,
                        validation_steps=None,
                        validation_data=(x_val, y_val)
                        )
    
    score = model.evaluate(
                           x_test,
                           y_test,
                           verbose=0
                           )
    
    y_prob = model.predict_proba(x_test)[:,1]
    y_pred = model.predict(x_test)  
    y_pred_label = np.argmax(y_pred, axis=1)
    
    return score, y_prob, y_pred, y_pred_label

# ----------------------------------------------------------------------------------
# train model on list
# ----------------------------------------------------------------------------------

def train_onList(x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list):

    '''
    generate list for train, validation and test dataset
    '''
    
    score_list        = []
    y_prob_list       = []
    y_pred_list       = []
    y_pred_label_list = []
    
    for x_train, x_val, x_test, y_train, y_val, y_test in zip(
                                                              x_train_list,
                                                              x_val_list,
                                                              x_test_list,
                                                              y_train_list,
                                                              y_val_list,
                                                              y_test_list
                                                              ):

        score, y_prob, y_pred, y_pred_label = train_model(
                                                          x_train,
                                                          x_val,
                                                          x_test,
                                                          y_train,
                                                          y_val,
                                                          y_test
                                                          )

        score_list.append(score)
        y_prob_list.append(y_prob)
        y_pred_list.append(y_pred)
        y_pred_label_list.append(y_pred_list)

    return score_list, y_prob_list, y_pred_list, y_pred_label_list

# ----------------------------------------------------------------------------------
# calculate confusion matrix and nornalized confusion matrix
# ----------------------------------------------------------------------------------
def con_mat(y_test_list, y_pred_label_list):

    for y_test, y_pred_label in zip(y_test_list, y_pred_label_list):
    
        cm = confusion_matrix(y_test, y_pred_label)
        cm = np.around(cm, 2)
        
        cm_norm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.around(cm_norm, 2)
        
        return cm, cm_norm

# ----------------------------------------------------------------------------------
# draw multiple ROC
# ----------------------------------------------------------------------------------
def plot_ROC():

    """
    plot ROC with five classes in one figure
    """
    
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    colors = ['aqua', 'red', 'royalblue']
    for i, color in zip(range(len(y_prob_list)), colors):
        fpr[i], tpr[i], threshold[i] = roc_curve(y_test_list[i], y_prob_list[i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        print("AUC:", np.around(roc_auc[i], 3)) 
        # plot ROC curve
        plt.plot(
                 fpr[i],
                 tpr[i],
                 color=color,
                 linewidth=3,
                 label='AUC %0.3f'%roc_auc[i]
                 )

    plt.legend(loc='lower right', prop={'size': 13, 'weight': 'bold'}) 
    #plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlim([-0.03, 1.0])
    plt.ylim([0, 1.03])
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='k', linewidth=3)
    ax.axhline(y=1.03, color='k', linewidth=3)
    ax.axvline(x=-0.03, color='k', linewidth=3)
    ax.axvline(x=1, color='k', linewidth=3)

    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    ax.tick_params(direction='out', length=6, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(result_dir, 'ROC_exvivo_1.png'), format='png', dpi=600)
    plt.close()


# ----------------------------------------------------------------------------------
# draw multiple ROC zoom in view
# ----------------------------------------------------------------------------------
def plot_ROC_zoom():

    """
    plot zoom in view of ROC with five classes and average ROC in one figure
    """
    
    fpr = dict()
    tpr = dict()
    threshold = dict()
    roc_auc = dict()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')

    colors = ['aqua', 'red', 'royalblue']
    for i, color in zip(range(len(y_prob_list)), colors):
           fpr[i], tpr[i], threshold[i] = roc_curve(y_test_list[i], y_prob_list[i])
           roc_auc[i] = auc(fpr[i], tpr[i])
           # plot ROC curve
           plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    linewidth=3,
                    label='AUC %0.3f'%roc_auc[i]
                    )

    plt.legend(loc='lower right', prop={'size': 13, 'weight': 'bold'})
    # plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
    plt.xlim([-0.01, 0.20])
    plt.ylim([0.8, 1.01])
    plt.xticks([0, 0.05, 0.10, 0.15, 0.20], fontsize=14, fontweight='bold')
    plt.yticks([0.80, 0.85, 0.90, 0.95, 1.00], fontsize=14, fontweight='bold')
    ax.axhline(y=0.8, color='k', linewidth=3)
    ax.axhline(y=1.01, color='k', linewidth=3)
    ax.axvline(x=-0.01, color='k', linewidth=3)
    ax.axvline(x=0.20, color='k', linewidth=3)

    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    ax.tick_params(direction='out', length=6, width=2, colors='k',
                   grid_color='k', grid_alpha=0.5)

    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(result_dir, 'ROC_exvivo_2.png'), format='png', dpi=600)
    plt.close()
    
# ----------------------------------------------------------------------------------
# run the model
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    epochs         = 1
    learning_rate  = 0.1
    batch_momentum = 0.9
    batch_size     = 200
    dropout_rate   = 0.4    
    alpha          = 0.3
    random_state   = 42
    ELU_alpha      = 1.0
    digit          = 3
    train_split    = 0.2
    test_split     = 0.5
    class0_ratio   = 1.0
    class1_ratio   = 1.0
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
    DNN model functions:
    
    keranl initializer:  'he_uniform', 'lecun_normal', 'lecun_uniform'
    optimizer function:  'adam', 'adamax', 'nadam', 'sgd'
    loss function:       'categorical_crossentropy'
    activation function: 'LeakyReLU(alpha=alpha)'
    '''

    # data and results path 
    project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Deep_Learning'
    result_dir  = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Deep_Learning'
    log_dir     = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Deep_Learning\log'
             
    file           = 'PCa_DWI.csv'
    DBSI_folder    = 'DBSI_results_0.1_0.1_0.8_0.8_2.3_2.3'
   
    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------
    print("Deep Neural Network for PCa grade classification: start...")
    
    start = timeit.default_timer()
    
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
    
    x_train_list, x_val_list, x_test_list, y_train_list, y_val_list, y_test_list = data.dataset_construction()
    
    score_list, y_prob_list, y_pred_list, y_pred_label_list = train_onList(
                                                                           x_train_list,
                                                                           x_val_list,
                                                                           x_test_list,
                                                                           y_train_list,
                                                                           y_val_list,
                                                                           y_test_list
                                                                           )

    cm, cm_norm = con_mat(y_test_list, y_pred_label_list)

    plot_ROC()
    plot_ROC_zoom()

    # ----------------------------------------------------------------------------------
    # confusion matrix, sensitivity, specificity, presicion, f-score, model parameters
    # ----------------------------------------------------------------------------------
    print('\noverall test loss:', np.around(score[0], digit))
    print('overall test accuracy:', np.around(score[1], digit))
##    print("\nconfusion matrix:")
##    print(cm)
##    print("\nnormalized confusion matrix:")
##    print(cm_norm) 
##    print("\nBenign, PCa:")   
##    print("sensitivity:", TPR_)
##    print("specificity:", TNR_)
##    print("precision:", PPV_)
##    print("negative predictive value:", NPV_)
##    print("false positive rate:", FPR_)
##    print("false negative rate:", FNR_)
##    print("false discovery rate:", FDR_)
##    print("\nprecision, recall, f1-score, support:")
##    print(classification_report(y_test, y_pred_label, digits=digit))       

    
    print("key model parameters:")
    print("train set size:", len(x_train_list[0].shape[0]))
    print("validation set size:", len(x_val_list[0].shape[0]))
    print("test set size:", len(x_test_list[0].shape[0]))
    print("epochs:", epochs)
    print("batch size:", batch_size)
    print("dropout rate:", dropout_rate)
    print("batch momentum:", batch_momentum)
    print("initial learning rate:", learning_rate)
    print("hidden layer neuron numbers:", n_neurons )
    stop = timeit.default_timer()
    print('DNN Running Time:', np.around(stop - start, 0), 'seconds')









