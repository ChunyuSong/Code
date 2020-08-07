#----------------------------------------------------------------------
# deep learning classifier using a multiple layer perceptron (MLP)
# batch normalization was used
#
# Author: Zezhong Ye;
# Date: 03.14.2019
# 
# Modified to accommodate patient split
# Modifier: Anthony Wu
# Last Modified: 2/29/2020
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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from time import gmtime, strftime
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler

from sklearn.multiclass import OneVsRestClassifier
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
                 project_dir,
                 random_state,
                 train_split,
                 val_split,
                 test_split,
                 ratio_1, 
                 ratio_2,
                 ratio_3, 
                 ratio_4,
                 ratio_5,
                 x_input
                 ):
                        
        self.project_dir  = project_dir
        self.random_state = random_state
        self.train_split  = train_split
        self.val_split    = val_split
        self.test_split   = test_split
        self.ratio_1      = ratio_1
        self.ratio_2      = ratio_2
        self.ratio_3      = ratio_3
        self.ratio_4      = ratio_4
        self.ratio_5      = ratio_5
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

        files = glob.glob(self.project_dir + "/*.xlsx")
        
        df = pd.DataFrame()
        i = 0
        for f in files:
            #print(i)
            i = i + 1
            data = pd.read_excel(f, 'Sheet1', header=None)
            data.iloc[:, 1:] = (data.iloc[:, 1:])/(data.iloc[:, 1:].max())
            data = data.iloc[data[data.iloc[:, 0] != 0].index]
            df = df.append(data)
            #print(df.shape)
        
        return df

    def data_sample_split(self): #does train, test, validation split. First by sample, then by voxel
    
        files = glob.glob(self.project_dir + "/*.xlsx")
        
        dfs = []
        i = 0
        for f in files:
            #print(i)
            #i = i + 1
            data = pd.read_excel(f, 'Sheet1', header=None)
            data = data.iloc[data[data.iloc[:, 0] != 0].index]
            if (data.shape[0] != 0):
                dfs.append([])
                data.iloc[:, 1:] = (data.iloc[:, 1:])/(data.iloc[:, 1:].max())
                data.loc[data.iloc[:, 0] == 1, 'y_cat'] = 0
                data.loc[data.iloc[:, 0] == 2, 'y_cat'] = 1
                data.loc[data.iloc[:, 0] == 3, 'y_cat'] = 2
                data.loc[data.iloc[:, 0] == 4, 'y_cat'] = 3
                data.loc[data.iloc[:, 0] == 5, 'y_cat'] = 4
                #print(set(data.iloc[:,0]))
                #print(data.y_cat.isnull().values.any())
                data.fillna(data.mean(), inplace=True)
                dfs[i].append(data)
                #print(data.isnull().values.any())
                #print(dfs[i])
                i = i+1

        num_test  = int(len(dfs)/8)
        num_train = len(dfs) - num_test
        print("number of samples: " + str(len(dfs)))

        test_idxs = np.random.choice(range(len(dfs)), size=num_test, replace=False)
        train_idxs = []
        for i in range(len(dfs)):
            if i not in test_idxs:
                train_idxs.append(i)
        #print(train_idxs)

        df_test  = pd.DataFrame()
        df_train = pd.DataFrame()
        #print(type(dfs[0]))
        for index in test_idxs:
            
            if df_test.empty:
                #df_test = pd.DataFrame(dfs[index])
                df_test = df_test.append(dfs[index])
            else:
                df_test = df_test.append(dfs[index])

        for index in train_idxs:
            
            if df_train.empty:
                df_train = df_train.append(dfs[index])
                #df_train = pd.DataFrame(dfs[index])
            else:
                df_train = df_train.append(dfs[index])
        
        print(df_train.shape)
        df_train.fillna(df_train.mean(), inplace=True)
        df_test.fillna(df_test.mean(), inplace=True)
        #print(df_train.isnull().values.any())
        #print(df_test.isnull().values.any())
        #print(set(df_train.y_cat))

        #df_train.to_csv(r'\\10.39.42.102\temp\Anthony_Wu\df_train.csv')
        df_train.loc[df_train.iloc[:, 0] == 1, 'y_cat'] = 0
        df_train.loc[df_train.iloc[:, 0] == 2, 'y_cat'] = 1
        df_train.loc[df_train.iloc[:, 0] == 3, 'y_cat'] = 2
        df_train.loc[df_train.iloc[:, 0] == 4, 'y_cat'] = 3
        df_train.loc[df_train.iloc[:, 0] == 5, 'y_cat'] = 4

        # class1        = df_train[df_train['y_cat'] == 0]
        # class1_sample = class1.sample(int(class1.shape[0]*self.ratio_1))
        
        # class2        = df_train[df_train['y_cat'] == 1]
        # class2_sample = class2.sample(int(class2.shape[0]*self.ratio_2))
        
        # class3        = df_train[df_train['y_cat'] == 2]
        # class3_sample = class3.sample(int(class3.shape[0]*self.ratio_3))
        
        # class4        = df_train[df_train['y_cat'] == 3]
        # class4_sample = class4.sample(int(class4.shape[0]*self.ratio_4))
        
        # class5        = df_train[df_train['y_cat'] == 4]
        # class5_sample = class5.sample(int(class5.shape[0]*self.ratio_5))

        # df_train = pd.concat([
        #                   class1_sample,
        #                   class2_sample,
        #                   class3_sample,
        #                   class4_sample,
        #                   class5_sample
        #                   ])

        x_train_ = df_train.iloc[:, self.x_input].values
        y_train_ = df_train.y_cat.astype('int').values

        x_test   = df_test.iloc[:, self.x_input].values
        y_test   = df_test.y_cat.astype('int').values

        x_train, x_val, y_train, y_val = train_test_split(
                                                          x_train_,
                                                          y_train_,
                                                          test_size=self.val_split,
                                                          random_state=self.random_state
                                                          ) 

        for i in [0, 1, 2, 3, 4]:
            if i not in y_train or i not in y_val or i not in y_test:
                print("gleason " + str(i) + " not in y_train. Redoing sample split...")
                print("error. plz rerun")
                self.data_sample_split()
        
        print("Presampled dataset collection: %s" % Counter(y_train))

        resample = SMOTE(random_state=42) #can also try randomoversampler
        #resample = RandomOverSampler(random_state=42)
        x_train_res, y_train_res = resample.fit_resample(x_train, y_train)
        print("Resampled dataset collection: %s" % Counter(y_train_res))

        return x_train_res, x_val, x_test, y_train_res, y_val, y_test

    def data_balancing(self):

        df = self.data_loading()

        df.loc[df.iloc[:, 0] == 1, 'y_cat'] = 0
        df.loc[df.iloc[:, 0] == 2, 'y_cat'] = 1
        df.loc[df.iloc[:, 0] == 3, 'y_cat'] = 2
        df.loc[df.iloc[:, 0] == 4, 'y_cat'] = 3
        df.loc[df.iloc[:, 0] == 5, 'y_cat'] = 4

        class1        = df[df['y_cat'] == 0]
        class1_sample = class1.sample(int(class1.shape[0]*self.ratio_1))
        
        class2        = df[df['y_cat'] == 1]
        class2_sample = class2.sample(int(class2.shape[0]*self.ratio_2))
        
        class3        = df[df['y_cat'] == 2]
        class3_sample = class3.sample(int(class3.shape[0]*self.ratio_3))
        
        class4        = df[df['y_cat'] == 3]
        class4_sample = class4.sample(int(class4.shape[0]*self.ratio_4))
        
        class5        = df[df['y_cat'] == 4]
        class5_sample = class5.sample(int(class5.shape[0]*self.ratio_5))

        df_2 = pd.concat([
                          class1_sample,
                          class2_sample,
                          class3_sample,
                          class4_sample,
                          class5_sample
                          ])
        #print(class1_sample.shape)

        return df_2

    def dataset_construction(self):
        
        df_2 = self.data_balancing()

        X    = df_2.iloc[:, self.x_input]

        Y    = df_2.y_cat.astype('int')

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
    
    y_pred       = model.predict(x_test)  
    y_pred_label = np.argmax(y_pred, axis=1)

    test_loss     = score[0]
    test_accuracy = score[1]

    test_loss     = np.around(test_loss, 3)
    test_accuracy = np.around(test_accuracy, 3)
    
    return score, y_pred, y_pred_label, test_loss, test_accuracy

# ----------------------------------------------------------------------------------
# calculate confusion matrix and nornalized confusion matrix
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
def DNN_ROC(Learning_Rate,batch_momentum,epochs,dropout_rate,batch_size, n=1):

    fpr       = dict()
    tpr       = dict()
    roc_auc   = dict()
    threshold = dict()

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
        
    colors = cycle(['aqua', 'red', 'purple', 'royalblue', 'black'])

    for i, color in zip(range(n_classes), colors):
        # print("y_test: ", np.array(y_test).shape)
        # print(y_test)
        # print("y_pred: ", np.array(y_pred)[:,i].shape)
        # print(y_pred[:,i])
        fpr[i], tpr[i], threshold[i] = roc_curve(y_test == i, y_pred[:,i], pos_label=1)
        
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
                str(Learning_Rate) + '_' + \
                str(batch_momentum) + '_' + \
                str(epochs) + '_' + \
                str(dropout_rate) + '_' + \
                str(batch_size) + '_' + str(n) + \
                strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
                
    plt.savefig(os.path.join(result_dir, ROC_filename), format='png', dpi=600)
    #plt.show()
    plt.close()

def DNN_PRC(Learning_Rate,batch_momentum,epochs,dropout_rate,batch_size, n=1):
    
    precision = dict()
    recall    = dict()
    threshold = dict()
    prc_auc   = []

    fig = plt.figure()
    ax  = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
        
    colors = cycle(['aqua', 'red', 'purple', 'royalblue', 'black'])

    for i, color in zip(range(n_classes), colors):

        precision[i], recall[i], _ = precision_recall_curve(y_test == i,
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
                str(Learning_Rate) + '_' + \
                str(batch_momentum) + '_' + \
                str(epochs) + '_' + \
                str(dropout_rate) + '_' + \
                str(batch_size) + '_' + str(n) +\
                strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
    
    plt.savefig(
                os.path.join(result_dir, PRC_filename),
                format='png',
                dpi=600
                )
    
    #plt.show()
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
# model hyper parameters
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    alpha         = 0.3
    random_state  = 42
    ELU_alpha     = 1.0
    digit         = 3
    train_split   = 0.2 #not used for patient split
    val_split     = 0.1
    test_split    = 0.5 #not used for patient split
    ratio_1       = 1
    ratio_2       = 1
    ratio_3       = 1
    ratio_4       = 1
    ratio_5       = 1
    count         = 0
    n_seed        = 100
    x_input       = [1, 2, 3, 8, 11, 13, 14, 18, 19, 20, 22, 26,
                     27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40]
    n_inputs      = len(x_input)
    n_outputs     = 5
    n_classes     = n_outputs
    list_seed     = range(n_seed)
    
    Learning_Rate = [0.1]
    Momentum      = [0.97]
    Dropout_Rate  = [0]
    Batch_Size    = [100]
    Epochs        = [10]
    
    n_neurons     = 100
    n_hidden1     = n_neurons
    n_hidden2     = n_neurons
    n_hidden3     = n_neurons
    n_hidden4     = n_neurons
    n_hidden5     = n_neurons
    n_hidden6     = n_neurons
    n_hidden7     = n_neurons
    n_hidden8     = n_neurons
    n_hidden9     = n_neurons
    n_hidden10    = n_neurons

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
    project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCA_in_vivo_data_excel'
    result_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2019_PCa_AI\invivo_grading\result'
    log_dir = r'\\10.39.42.102\temp\Zezhong_Ye\2019_PCa_AI\invivo_grading\log'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------
    
    print("Deep Neural Network for PCa grade classification: start...")

    start = timeit.default_timer()
    
    data_path()

    PCa_Data = PCa_data(
                        project_dir,
                        random_state,
                        train_split,
                        val_split,
                        test_split,
                        ratio_1, 
                        ratio_2,
                        ratio_3, 
                        ratio_4,
                        ratio_5,
                        x_input
                        )

    x_train, x_val, x_test, y_train, y_val, y_test = PCa_Data.data_sample_split()

    total_run = len(Momentum)*len(Epochs)*len(Batch_Size)*len(Learning_Rate)*len(list_seed)*len(Dropout_Rate)
    
    breaking = False

    for n in list_seed:
        for i in Batch_Size:
            
            for j in Momentum:
                
                for k in Epochs:

                    for l in Learning_Rate:

                        for m in Dropout_Rate:

                            np.random.seed(seed=n)

                            print('seed number:', str(n))

                            count += 1

                            print('\nRunning times: ' + str(count) + '/' + str(total_run))

                            batch_size     = i
                            batch_momentum = j
                            epochs         = k
                            learning_rate  = l
                            dropout_rate   = m
                                
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

                            DNN_ROC(learning_rate,batch_momentum,epochs,dropout_rate,batch_size, n)
                            DNN_PRC(learning_rate,batch_momentum,epochs,dropout_rate,batch_size, n)
                            #plot_CM(cm, 'd')                                                               

                            # ----------------------------------------------------------------------------------
                            # confusion matrix, sensitivity, specificity, presicion, f-score, model parameters
                            # ----------------------------------------------------------------------------------
                            print('\noverall test loss:  ', test_loss)
                            print('overall test accuracy:', test_accuracy)

                            #print('confusion matrix:')
                            #print(cm)
                            #print('normalized confusion matrix:')
                            print(cm_norm) 
                            #print('sensitivity:         ', TPR)
                            #print('specificity:         ', TNR)
                            #print('precision:           ', PPV)
                            #print('neg predictive value:', NPV)
                            #print('false positive rate: ', FPR)
                            #print('false negative rate: ', FNR)
                            #print('false discovery rate:', FDR)
                            print(classification_report(y_test, y_pred_label, digits=digit))   
                            print('epochs:        ', epochs)
                            print('batch size:    ', batch_size)
                            print('dropout rate:  ', dropout_rate)
                            print('batch momentum:', batch_momentum)
                            print('learning rate: ', learning_rate)
                            print('neuron numbers:', n_neurons)
                        
                            if test_accuracy > 0.99:
                                breaking = True

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