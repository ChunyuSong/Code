#----------------------------------------------------------------------
# deep learning classifier using a multiple layer perceptron (MLP)
# batch normalization was used
# 
# Modified to accommodate specific patient-wise split
# Modifier: Anthony Wu
# Date: 5/20/2020
# 
#-------------------------------------------------------------------------------------------

import os
import timeit
import tensorflow
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
from time import gmtime, strftime

import keras
from keras import initializers
from keras.optimizers import RMSprop
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import ELU, LeakyReLU

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

import warnings
warnings.filterwarnings("ignore") #NOTE: COMMENT THIS OUT IF DEBUGGING!!!

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
                 test_split,
                 non_PCa_non_biopsy,
                 PCa_biopsy,
                 non_PCa_biopsy,
                 sample_1, 
                 sample_2,
                 sample_3,
                 sample_4,
                 sample_5,
                 sample_6,
                 x_input
                 ):
                        
        self.project_dir  = project_dir
        self.random_state = random_state
        self.test_split   = test_split
        self.non_PCa_non_biopsy = non_PCa_non_biopsy
        self.PCa_biopsy = PCa_biopsy
        self.non_PCa_biopsy  = non_PCa_biopsy
        self.sample_1     = sample_1
        self.sample_2     = sample_2 
        self.sample_3     = sample_3
        self.sample_4     = sample_4 
        self.sample_5     = sample_5
        self.sample_6     = sample_6 
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

    def sample_split_discrete(self, data, sample_size, meta = False):
        #does patient split sampling given data and discrete sampling size.
        #returns split dataset
        PIDs = 0 #patient ID column is 0
        unique_IDs = [] #contains all distinct patient IDs
        # print(len(data.iloc[:,PIDs].unique())) #unique is lost here; something is missing
        for i in data.iloc[:, PIDs]:
            
            if i not in unique_IDs: #code cannot see 47 for some reason
                unique_IDs.append(i)
            if i == "47_SUN_QING_HE":
                print("shitocane 47 found but not appended")

        print("number of unique patients: " + str(len(unique_IDs)))
        print("number of samples: " + str(sample_size))
        print("data size: " + str(data.shape[0]))
        print(unique_IDs)
        if len(unique_IDs) == 0:
            if meta:
                return pd.DataFrame({'A' : []}), pd.DataFrame({'A' : []})
            return pd.DataFrame({'A' : []})
        sample_IDs = np.random.choice(unique_IDs, size = sample_size, replace=False)
        print(len(sample_IDs))
        df_sample = pd.DataFrame({'A' : []})
        for PID in sample_IDs: #put data of all sample patients into dataframe
            
            if df_sample.empty:
                df_sample = data[data.iloc[:, PIDs] == PID]
            else:
                df_sample = df_sample.append(data[data.iloc[:, PIDs] == PID])
        print("shape of this sample: ", df_sample.shape)
        if meta: #return dataset of leftover
            df_rest = pd.DataFrame({'A' : []})
            for PID in unique_IDs: #put data of all sample patients into dataframe
                if PID not in sample_IDs:
                    if df_rest.empty:
                        df_rest = data[data.iloc[:, PIDs] == PID]
                    else:
                        df_rest = df_rest.append(data[data.iloc[:, PIDs] == PID])
                    
            return df_sample, df_rest

        return df_sample

    def data_split(self):
        
        '''
        construct train, test, validation datasets
        '''

        non_PCa_non_biopsy = pd.read_csv(os.path.join(project_dir, self.non_PCa_non_biopsy))
        PCa_biopsy = pd.read_csv(os.path.join(project_dir, self.PCa_biopsy))
        non_PCa_biopsy = pd.read_csv(os.path.join(project_dir, self.non_PCa_biopsy))
        
        class0s = []
        class1s = []
        for data in [non_PCa_non_biopsy, PCa_biopsy, non_PCa_biopsy]:
            #prepare datasets
            data['y_cat'] = 2

            data.loc[data['ROI_Class'] == 't', 'y_cat'] = 1

            if histology == 'Benign':
                data.loc[data['ROI_Class'].isin(['p', 'c']), 'y_cat'] = 0
            
            else:
                data.loc[data['ROI_Class'] == histology, 'y_cat'] = 0
            
            class0s.append(data[data['y_cat'] == 0])
            class1s.append(data[data['y_cat'] == 1])

        print("data shapes: ", non_PCa_non_biopsy.shape, PCa_biopsy.shape, non_PCa_biopsy.shape)
        #for class0s and class1s, order of data type is: [non_PCa_non_biopsy, PCa_biopsy, non_PCa_biopsy]
        #NOTE: put edits here
        #class0 means voxelwise roi is benign. Does NOT mean it's non_PCa, since it is voxelwise. Remember this is
        #not the conclusion; however, when printing shape, you will see half of it is zero because in this case it
        #correlates perfectly.

        for i in range(len(class0s)):
            print("shape of class 0, ", i, " :", class0s[i].shape)
            print("shape of class 1, ", i, " :", class1s[i].shape)


        #CREATING TRAIN SET
        print("train:")
        # train_class0_sample_NPNB = self.sample_split_discrete(class0s[0], sample_1) #data from non_PCa_non_biopsy
        train_class0_sample_NPNB = class0s[0]
        train_class0_sample_PB, rest_class0s1 = self.sample_split_discrete(class0s[1], sample_2, meta= True) #do initial split of PCa_biopsy
        train_class0_sample = pd.concat([train_class0_sample_NPNB, train_class0_sample_PB])
        #the dataframes with "rest" will be used for the test/val datasets to prevent repeats
        # train_class1_sample_NPNB = self.sample_split_discrete(class1s[0], sample_1) #data from non_PCa_non_biopsy
        train_class1_sample_NPNB = class1s[0]
        train_class1_sample_PB, rest_class1s1 = self.sample_split_discrete(class1s[1], sample_2, meta= True) #do initial split for PCa_biopsy
        train_class1_sample = pd.concat([train_class1_sample_NPNB, train_class1_sample_PB])
        #the dataframes with "rest" will be used for the test/val datasets to prevent repeats
        df_train_sum  = pd.concat([train_class0_sample, train_class1_sample])
        x_train       = df_train_sum.iloc[:, self.x_input]
        y_train       = df_train_sum.y_cat.astype('int')



        #CREATING TEST SET
        print("test:")
        test_class0_sample_NPB, rest_class0s2 = self.sample_split_discrete(class0s[2], sample_3, meta= True) #do initial split of non_PCa_biopsy
        test_class0_sample_PB, rest_class0s1 = self.sample_split_discrete(rest_class0s1, sample_4, meta= True) #do secondary split of PCa_biopsy
        test_class0_sample = pd.concat([test_class0_sample_NPB, test_class0_sample_PB])
        #the dataframes with "rest" will be used for the val dataset to prevent repeats
        test_class1_sample_NPB, rest_class1s2 = self.sample_split_discrete(class1s[2], sample_3, meta= True) #do initial split of non_PCa_biopsy
        test_class1_sample_PB, rest_class1s1 = self.sample_split_discrete(rest_class1s1, sample_4, meta= True) #do secondary split for PCa_biopsy
        test_class1_sample = pd.concat([test_class1_sample_NPB, test_class1_sample_PB])
        #the dataframes with "rest" will be used for the val dataset to prevent repeats
        df_test_sum  = pd.concat([test_class0_sample, test_class1_sample])
        x_test       = df_test_sum.iloc[:, self.x_input]
        y_test       = df_test_sum.y_cat.astype('int')



        #CREATING VAL SET
        print("val:")
        val_class0_sample_NPB = self.sample_split_discrete(rest_class0s2, sample_5) #take rest of non_PCa_biopsy
        val_class0_sample_PB = self.sample_split_discrete(rest_class0s1, sample_6) #take rest of PCa_biopsy
        val_class0_sample = pd.concat([val_class0_sample_NPB, val_class0_sample_PB])
        #the dataframes with "rest" will be used for the val dataset to prevent repeats
        val_class1_sample_NPB = self.sample_split_discrete(rest_class1s2, sample_5) #take rest of non_PCa_biopsy
        val_class1_sample_PB = self.sample_split_discrete(rest_class1s1, sample_6) #take rest of PCa_biopsy
        val_class1_sample = pd.concat([val_class1_sample_NPB, val_class1_sample_PB])
        #the dataframes with "rest" will be used for the val dataset to prevent repeats
        df_val_sum  = pd.concat([val_class0_sample, val_class1_sample])
        x_val       = df_val_sum.iloc[:, self.x_input]
        y_val       = df_val_sum.y_cat.astype('int')

        return x_train, y_train, x_test, y_test, x_val, y_val
               
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
def model_training(x_test, y_test, x_train, y_train, x_val, y_val):

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
    print("x_train shape: ", x_train.shape)
    print("x_val shape: ", x_val.shape)
    print("test shape: ", x_test.shape)
    y_pred        = model.predict(x_test)
    y_pred_label  = np.argmax(y_pred, axis=1)
    y_prob        = model.predict_proba(x_test)[:,1]

    test_loss     = score[0]
    test_accuracy = score[1]

    test_loss     = np.around(test_loss, 3)
    test_accuracy = np.around(test_accuracy, 3)

    return y_pred, y_prob, test_loss, test_accuracy, y_pred_label

# ----------------------------------------------------------------------------------
# ROC and AUC
# ----------------------------------------------------------------------------------
def con_mat(y_pred_label):
    
    cm = confusion_matrix(y_test, y_pred_label)
    print(cm)
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
                    annot_kws={'size': 18, 'fontweight': 'bold'},
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

    #plt.show()
    plt.close()

def DNN_ROC():
    
    fpr       = dict()
    tpr       = dict()
    roc_auc   = dict()
    threshold = dict()

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
        
    colors = cycle(['aqua', 'red', 'purple', 'royalblue'])

    for i, color in zip(range(len(y_prob_list)), colors):

        fpr[i], tpr[i], _ = roc_curve(y_test_list[i], y_prob_list[i])
        
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        print('ROC AUC %.2f' % roc_auc[i])
        
        plt.plot(
                 fpr[i],
                 tpr[i],
                 color=color,
                 linewidth=3,
                 label='AUC %0.3f' % roc_auc[i]
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

    for i, color in zip(range(len(y_prob_list)), colors):

        precision[i], recall[i], _ = precision_recall_curve(y_test_list[i],
                                                            y_prob_list[i])
        
        RP_2D = np.array([recall[i], precision[i]])
        RP_2D = RP_2D[np.argsort(RP_2D[:,0])]

        prc_auc.append(auc(RP_2D[1], RP_2D[0]))
        
        print('PRC AUC %.3f' % auc(RP_2D[1], RP_2D[0]))
                
        plt.plot(
                 recall[i],
                 precision[i],
                 color=color,
                 linewidth=3,
                 label='AUC %0.3f' % prc_auc[i]
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
    alpha          = 0.3
    random_state   = 42
    ELU_alpha      = 1.0
    digit          = 3
    test_split     = 0.5
    count          = 0
    x_input        = list(range(7, 30))
    x_input.remove(8) #take out adc; comment this line out if you want it
    n_inputs       = len(x_input)
    n_outputs      = 2
    n_classes      = n_outputs
    sample_1       = 96 #train sample 0; not used, used all samples for training
    sample_2       = 60 #train sample 1
    sample_3       = 20 #test sample 0
    sample_4       = 10 #test sample 1
    sample_5       = 32 #val sample 0
    sample_6       = 25 #val sample 1

    Learning_rate  = [0.01, 0.1, 0.2]
    Momentum       = [0.9, 0.8]
    Dropout_rate   = [0.3, 0.2]
    Batch_size     = [500]
    Epochs         = [5]
    N_neurons      = [100]
    histology_list = ['p', 'c', 'Benign']
    
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
    
    non_PCa_non_biopsy = 'benign_mpMRI_voxel.csv'
    PCa_biopsy         = 'PCa.csv'
    non_PCa_biopsy     = 'benign_biopsy.csv'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------
    
    print("Deep Neural Network for PCa grade classification: start...")

    start = timeit.default_timer()

    data_path()

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

                            x_test_list    = []
                            y_test_list    = []

                            x_val_list     = []
                            y_val_list     = []

                            x_train_list   = []
                            y_train_list   = []

                            y_prob_list    = []
                            y_pred_list    = []
                            loss_list      = []
                            accuracy_list  = []

                            batch_size     = i
                            batch_momentum = j
                            epochs         = k
                            learning_rate  = l
                            n_neurons      = m
                            dropout_rate   = n

                            PCa_Data = PCa_data(
                                                project_dir,
                                                random_state,
                                                test_split,
                                                non_PCa_non_biopsy,
                                                PCa_biopsy,
                                                non_PCa_biopsy,
                                                sample_1, 
                                                sample_2,
                                                sample_3,
                                                sample_4,
                                                sample_5,
                                                sample_6,
                                                x_input
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

                            for histology in histology_list:

                                x_train, y_train, x_test, y_test, x_val, y_val = PCa_Data.data_split()
                                print("hist x_train shape: ", x_train.shape)
                                x_test_list.append(x_test)
                                y_test_list.append(y_test)
                                x_train_list.append(x_train)
                                y_train_list.append(y_train)
                                x_val_list.append(x_val)
                                y_val_list.append(y_val)

                            for x_test, y_test, x_train, y_train, x_val, y_val in zip(x_test_list, y_test_list, x_train_list, y_train_list, x_val_list, y_val_list):

                                y_pred, y_prob, test_loss, test_accuracy, y_pred_label = model_training(x_test, y_test, x_train, y_train, x_val, y_val)
                                
                                y_prob_list.append(y_prob)
                                y_pred_list.append(y_pred)
                                loss_list.append(test_loss)
                                accuracy_list.append(test_accuracy)

                            cm, cm_norm = con_mat(y_pred_label)
                            plot_CM(cm_norm, '')
                            DNN_ROC()
                            DNN_PRC()

                            print('\noverall loss:  ', loss_list)
                            print('overall accuracy:', accuracy_list)
                            print('epochs:          ', epochs)
                            print('batch size:      ', batch_size)
                            print('dropout rate:    ', dropout_rate)
                            print('batch momentum:  ', batch_momentum)
                            print('learning rate:   ', learning_rate)
                            print('neuron numbers:  ', n_neurons)
                        
                            if test_accuracy > 0.999:
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

    print('train size:     ', len(x_train))
    print('validation size:', len(x_val))
    print('test size:      ', len(x_test))
    
    stop = timeit.default_timer()
    running_seconds = np.around(stop - start, 0)
    running_minutes = np.around(running_seconds/60, 0)
    print('DNN running time:', running_seconds, 'seconds')
    print('DNN running time:', running_minutes, 'minutes')

