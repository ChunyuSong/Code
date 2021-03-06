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
                 project_dir,
                 random_state,
                 train_split,
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
        
        for f in files:
            data = pd.read_excel(f, 'Sheet1', header=None)
            data.iloc[:, 1:] = (data.iloc[:, 1:])/(data.iloc[:, 1:].max())
            data = data.iloc[data[data.iloc[:, 0] != 0].index]
            df = df.append(data)
        
        return df
    
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

        return df_2

    def dataset_construction(self):
        
        df_2 = self.data_balancing()

        X    = df_2.iloc[:, self.x_input]

        Y    = df_2.y_cat.astype('int')

        Y_2  = label_binarize(Y, classes=[0, 1, 2, 3, 4])

        x_train, x_test_1, y_train, y_test_1 = train_test_split(
                                                                X,
                                                                Y_2,
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

    for i in range(n_classes):
        
        fpr[i], tpr[i], threshold[i] = roc_curve(y_test[:, i], y_pred[:, i])
        
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())                                                                          
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    
    #plt.title('ROC', fontweight='bold', fontsize=14)
    ##plt.plot(
    ##         fpr["micro"],
    ##         tpr["micro"],
    ##         label='micro-average ROC curve (area = {0:0.2f})'
    ##               ''.format(roc_auc["micro"]),
    ##         color='deeppink',
    ##         linestyle=':',
    ##         linewidth=3
    ##)
    ##plt.plot(
    ##         fpr["macro"],
    ##         tpr["macro"],
    ##         label='macro-average ROC curve (area = {0:0.2f})'
    ##               ''.format(roc_auc["macro"]),
    ##         color='navy',
    ##         linestyle=':',
    ##         linewidth=3
    ##)
    
    colors = cycle(['aqua', 'red', 'purple', 'royalblue', 'black'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
                 fpr[i],
                 tpr[i],
                 color=color,
                 linewidth=3,
                 label='AUC {1:0.3f}'
                 ''.format(i+1, roc_auc[i])
                 )
        print(np.around(roc_auc[i], 3))

    #plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
    plt.xlim([-0.03, 1])
    plt.ylim([0, 1.03])
    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=1.03, color='k', linewidth=4)
    ax.axvline(x=-0.03, color='k', linewidth=4)
    ax.axvline(x=1, color='k', linewidth=4) 
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate', fontweight='bold', fontsize=15)
    plt.ylabel('True Positive Rate', fontweight='bold', fontsize=15)
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
    plt.close()

# ----------------------------------------------------------------------------------
# plot summed ROC curves zoom in view
# ----------------------------------------------------------------------------------

def DNN_ROC_Zoom():
    
    # Zoom in view of the upper left corner.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_aspect('equal')
    plt.grid(True)
    plt.plot(
             fpr["micro"],
             tpr["micro"],
             label='Micro Avg (AUC {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='orange',
             linestyle=':',
             linewidth=3
             )

    plt.plot(
             fpr["macro"],
             tpr["macro"],
             label='Macro Avg (AUC {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy',
             linestyle=':',
             linewidth=3
             )

    colors = cycle(['aqua', 'red', 'purple', 'royalblue'])

    for i, color in zip(range(n_classes), colors):
        plt.plot(
                 fpr[i],
                 tpr[i],
                 color=color,
                 linewidth=3,
                 #label='Class {0} (area = {1:0.2f})'
                 #''.format(i+1, roc_auc[i])
                 )
        
    #plt.title('ROC', fontweight='bold', fontsize=14)
    #plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
    plt.xlim(0, 0.41)
    plt.ylim(0.6, 1.01)
    ax.axhline(y=0.6, color='k', linewidth=4)
    ax.axhline(y=1.01, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=0.41, color='k', linewidth=4) 
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4], fontsize=14, fontweight='bold')
    plt.yticks([0.6, 0.7, 0.8, 0.9, 1.0], fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate',  fontweight='bold', fontsize=15)
    plt.ylabel('True Positive Rate',  fontweight='bold', fontsize=15)
    plt.legend(loc='lower right', prop={'size': 13, 'weight':'bold'})
    #plt.show()
    plt.savefig(os.path.join(result_dir, 'ROC_sum_2' + '.png'), format='png', dpi=600)
    plt.close()

# ----------------------------------------------------------------------------------
# model hyper parameters
# ---------------------------------------------------------------------------------- 
if __name__ == '__main__':

    # model paramters
    alpha         = 0.3
    random_state  = 42
    ELU_alpha     = 1.0
    digit         = 3
    train_split   = 0.2
    test_split    = 0.5
    ratio_1       = 1
    ratio_2       = 1
    ratio_3       = 1
    ratio_4       = 1
    ratio_5       = 1
    count         = 0
    x_input       = [1, 2, 3, 8, 11, 13, 14, 18, 19, 20, 22, 26,
                     27, 28, 29, 31, 32, 33, 35, 36, 37, 38, 39, 40]
    n_inputs      = len(x_input)
    n_outputs     = 5
    n_classes     = n_outputs
    
    Learning_Rate = [0.01, 0.001]
    Momentum      = [0.97, 0.98, 0.99]
    Dropout_Rate  = [0]
    Batch_Size    = [100, 200]
    Epochs        = [10, 20]
    
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
    project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCA_in_vivo_data_excel'
    result_dir = r'\\10.39.42.102\temp\2019_PCa_AI\invivo_grading\result'
    log_dir = r'\\10.39.42.102\temp\2019_PCa_AI\invivo_grading\log'

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
                        test_split,
                        ratio_1, 
                        ratio_2,
                        ratio_3, 
                        ratio_4,
                        ratio_5,
                        x_input
                        )

    x_train, x_val, x_test, y_train, y_val, y_test = PCa_Data.dataset_construction()

    total_run = len(Momentum)*len(Epochs)*len(Batch_Size)*len(Learning_Rate)
    
    breaking = False

    for i in Batch_Size:
        
        for j in Momentum:
            
            for k in Epochs:

                for l in Learning_Rate:

                    for m in Dropout_Rate:

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
                                                                
                        DNN_ROC()

                        # ----------------------------------------------------------------------------------
                        # confusion matrix, sensitivity, specificity, presicion, f-score, model parameters
                        # ----------------------------------------------------------------------------------
                        print('\noverall test loss:  ', test_loss)
                        print('overall test accuracy:', test_accuracy)
                        
                        #print('key model parameters:')
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

