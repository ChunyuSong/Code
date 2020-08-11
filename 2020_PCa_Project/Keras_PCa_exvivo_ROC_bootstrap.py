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
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
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
                 x_input,
                 ):
                        
        self.train_file   = train_file
        self.project_dir  = project_dir
        self.random_state = random_state
        self.train_split  = train_split
        self.test_split   = test_split
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
    
    def data_restructure(self):

        df = self.data_loading()

        df['y_cat'] = 2
        
        df.loc[df['ROI_Class'] == 'PCa', 'y_cat'] = 1
            
        if histology == 'Benign':
           df.loc[df['ROI_Class'].isin(['SBPH', 'BPZ', 'BPH']), 'y_cat'] = 0
           
        else:
             df.loc[df['ROI_Class'] == histology, 'y_cat'] = 0
         
        class0 = df[df['y_cat'] == 0]
        class1 = df[df['y_cat'] == 1]

        class0_sample = class0.sample(int(class0.shape[0]))
        class1_sample = class1.sample(int(class1.shape[0]))

        df2 = pd.concat([class0_sample, class1_sample])

        return df2

    def data_construction(self):
        
        df2 = self.data_restructure()

        x = df2.iloc[:, self.x_input]

        y = df2.y_cat.values.astype('int')

        x_train, x_test_1, y_train, y_test_1 = train_test_split(
                                                                x,
                                                                y,
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
    
    y_pred    = model.predict(x_test)[:, 1]
    test_loss = np.around(score[0], 3)
    test_acc  = np.around(score[1], 3)

    return y_pred, test_loss, test_acc

# ----------------------------------------------------------------------------------
# AUC, sensitivity, specificity with 1000 bootstrap itertations
# ----------------------------------------------------------------------------------

def ROC_AUC_bootstrap():

    """
    calculate AUC, TPR, TNR with 1000 iterations
    """
    
    AUC  = []
    THRE = []
    TNR  = []
    TPR  = []

    for i in range(4):
        
        aucs  = []
        tprs  = []
        fprs  = []
        tnrs  = []
        thres = []

        y_test_target = y_test_list[i]
        y_pred_target = y_pred_list[i]
        
        for j in range(n_bootstrap):
                        
            index = range(len(y_pred_target))

            indices = resample(
                               index,
                               replace=True,
                               n_samples=int(len(y_pred_target))
                               )

            fpr, tpr, thre = roc_curve(
                                       y_test_target[indices],
                                       y_pred_target[indices]
                                       )

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
       
            aucs.append(roc_auc_score(y_test_target[indices], y_pred_target[indices]))
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
                                  'TPR_1',
                                  'TPR_2',
                                  'TPR_3',
                                  'TPR_4',
                                  'TNR_1',
                                  'TNR_2',
                                  'TNR_3',
                                  'TNR_4'
                                  ],                            
                           columns=[
                                    'mean',
                                    '95% CI -',
                                    '95% CI +'
                                    ]
                           )

    filename = str('grading_invivo') + '_' + \
               str(n_bootstrap) + '_' + \
               str(epochs) + '_' + \
               str(strftime("%d-%b-%Y-%H-%M-%S", gmtime())) + \
               '.csv'

    stat_df.to_csv(os.path.join(result_dir, filename))

    return stat_df

# ----------------------------------------------------------------------------------
# model hyper parameters
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    n_bootstrap    = 1000
    alpha          = 0.3
    random_state   = 42
    ELU_alpha      = 1.0
    digit          = 3
    train_split    = 0.3
    test_split     = 0.5
    count          = 0
    x_input        = range(7, 30)
    n_inputs       = len(x_input)
    n_outputs      = 2
    n_classes      = n_outputs

    Learning_rate  = [0.1, 0.01, 0.001, 0.0001]
    Momentum       = [0.97, 0.99]
    Dropout_rate   = [0]
    Batch_size     = [200]
    Epochs         = [20, 10]
    N_neurons      = [100]
    histology_list = ['SBPH', 'BPZ', 'BPH', 'Benign']
    
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
    project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning'
    result_dir  = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\ROC\result'
    log_dir     = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\ROC\log'             
    train_file  = 'PCa.csv'

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

                            x_test_list = []
                            y_test_list = []
                            y_prob_list = []
                            y_pred_list = []
                            loss_list   = []
                            acc_list    = []

                            batch_size     = i
                            batch_momentum = j
                            epochs         = k
                            learning_rate  = l
                            n_neurons      = m
                            dropout_rate   = n

                            PCa_Data = PCa_data(
                                                train_file,
                                                project_dir,
                                                random_state,
                                                train_split,
                                                test_split,
                                                x_input,
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
        
                                x_train, x_val, x_test, y_train, y_val, y_test = PCa_Data.data_construction()

                                x_test_list.append(x_test)
                                y_test_list.append(y_test)

                            for x_test, y_test in zip(x_test_list, y_test_list):

                                y_pred, test_loss, test_acc = model_training()
                                
                                y_pred_list.append(y_pred)
                                loss_list.append(test_loss)
                                acc_list.append(test_acc)
                                                            
                            total = ROC_AUC_bootstrap()

                            stat_df = stat_report()

                            print(stat_df)
                                    
                            print('\noverall loss:  ', loss_list)
                            print('overall accuracy:', acc_list)
                            print('epochs:          ', epochs)
                            print('batch size:      ', batch_size)
                            print('dropout rate:    ', dropout_rate)
                            print('batch momentum:  ', batch_momentum)
                            print('learning rate:   ', learning_rate)
                            print('neuron numbers:  ', n_neurons)
                        
                            if test_acc > 0.999:
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
    running_sec = np.around(stop - start, 0)
    running_min = np.around(running_sec/60, 0)
    print('DNN running time:', running_sec, 'sec')
    print('DNN running time:', running_min, 'min')

