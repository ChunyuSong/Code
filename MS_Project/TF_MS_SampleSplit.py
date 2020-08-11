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

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import confusion_matrix

from tensorflow.keras.metrics import categorical_accuracy
from imblearn.over_sampling import SMOTE
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
class PCa_data(object):
    
    '''
    calculate DNN statisical results
    '''
    
    def __init__(
                 self,
                 train_file,
                 project_dir,
                 random_state,
                 val_split,
                 class0_ratio, 
                 class1_ratio,
                 class2_ratio, 
                 class3_ratio,
                 x_input
                 ):
                        
        self.train_file   = train_file
        self.project_dir  = project_dir
        self.random_state = random_state
        self.val_split    = val_split
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

        print('class0:', len(class0_sample))
        print('class1:', len(class1_sample))
        print('class2:', len(class2_sample))
        print('class3:', len(class3_sample))
        
        df2 = pd.concat([class0_sample, class1_sample, class2_sample, class3_sample])

        return df2

    def data_sample_split(self): #does train, test, validation split. First by sample, then by voxel
    
        column = 0 #this is the column that holds all of the unique sample ids

        df2 = self.data_balancing()

        unique_samples = []
        
        for i in df2.iloc[:, 0]:
            
            if i not in unique_samples:
                unique_samples.append(i)
                print("Sample name: " + i + " grade collection: %s" %Counter(df2[df2.iloc[:, column] == i].y_cat.astype('int').values))

        #print(unique_samples)
        #print(len(unique_samples))

        #9:1 samplewise tts;
        num_test  = int(np.ceil((len(unique_samples)/5)))
        num_train = len(unique_samples) - num_test

        print('\ntotal train number:', num_train)
        print('total test number:', num_test)

        sample_idxs = list(range(len(unique_samples)))

        test_idxs   = np.random.choice(sample_idxs, size=num_test, replace=False)
     
        for i in test_idxs:
            sample_idxs.remove(i)
            
        train_idxs = sample_idxs
        print('test sample ID:', test_idxs)
        #print('train sample ID:', train_idxs)

        test_names  = []
        train_names = []

        df_test  = pd.DataFrame({'A': []})
        df_train = pd.DataFrame({'A': []})

        for index in test_idxs:
            
            test_names.append(unique_samples[index])
            
            if df_test.empty:
                df_test = df2[df2.iloc[:, column] == unique_samples[index]]
            else:
                df_test = df_test.append(df2[df2.iloc[:, column] == unique_samples[index]])

        for index in train_idxs:
            
            train_names.append(unique_samples[index])
            
            if df_train.empty:
                df_train = df2[df2.iloc[:, column] == unique_samples[index]]
            else:
                df_train = df_train.append(df2[df2.iloc[:, column] == unique_samples[index]])
        
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

        for i in [0, 1, 2, 3]:
            if i not in y_train:
                print("gleason " + i + " not in y_train. Redoing sample split...")
                data_sample_split()
        
        print("Presampled dataset collection: %s" % Counter(y_train))

        resample = SMOTE(random_state=42)
        x_train_res, y_train_res = resample.fit_resample(x_train, y_train)
        print("Resampled dataset collection: %s" % Counter(y_train_res))

        return x_train_res, x_val, x_test, y_train_res, y_val, y_test

    def data_manual_sample_split(self): #does train, test, validation split. First by sample, then by voxel
        
            column = 0 #this is the column that holds all of the unique sample ids

            df2 = self.data_balancing()

            unique_samples = []
            
            for i in df2.iloc[:, 0]:
                
                if i not in unique_samples:
                    unique_samples.append(i)
                    print("Sample name: " + i + " grade collection: %s" %Counter(df2[df2.iloc[:, column] == i].y_cat.astype('int').values))

            #print(unique_samples)
            #print(len(unique_samples))

            #manual tts to ensure proper test set
            #test_names  = ['WU_006_A23','WU_005_B19','WU_010_B45','WU_001_C5','WU_009_C27'] #test_names for 1:1:1:1 sample_class ratios
            test_names  = ['WU_006_A23','WU_005_B19','WU_010_B45','WU_001_C5','WU_009_C27', 'WU_007_F13'] #test_names for 1:1:1:0.2 sample_class ratios

            train_names = list(unique_samples)
            print(train_names)

            for i in test_names:
                train_names.remove(i)

            df_test  = pd.DataFrame({'A': []})
            df_train = pd.DataFrame({'A': []})

            for index in test_names:
                if df_test.empty:
                    df_test = df2[df2.iloc[:, column] == index]
                else:
                    df_test = df_test.append(df2[df2.iloc[:, column] == index])

            for index in train_names:                
                if df_train.empty:
                    df_train = df2[df2.iloc[:, column] == index]
                else:
                    df_train = df_train.append(df2[df2.iloc[:, column] == index])
            
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

            for i in [0, 1, 2, 3]:
                if i not in y_train:
                    print("gleason " + i + " not in y_train. Redoing sample split...")
                    data_manual_sample_split()
            
            print("Presampled dataset collection: %s" % Counter(y_train))

            resample = SMOTE(random_state=42, k_neighbors=5, sampling_strategy="auto") #k_neighbors = 5 by default
            #resample = RandomOverSampler(random_state=42, sampling_strategy='minority')
            x_train_res, y_train_res = resample.fit_resample(x_train, y_train)
            print("Resampled dataset collection: %s" % Counter(y_train_res))

            return x_train_res, x_val, x_test, y_train_res, y_val, y_test
    
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
                           metrics=[categorical_accuracy]
                           #metrics=['accuracy']
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

        y_pred       = self.model.predict(x_test)[:, 1]
        y_pred_class = self.model.predict_classes(x_test)

        test_loss = np.around(score[0], 3)
        test_acc  = np.around(score[1], 3)

        return y_pred, y_pred_class, test_loss, test_acc

# ----------------------------------------------------------------------------------
# calculate confusion matrix and nornalized confusion matrix
# ----------------------------------------------------------------------------------
def con_mat():
    
    cm = confusion_matrix(y_test, y_pred_class)
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
                    annot_kws={'size': 20, 'fontweight': 'bold'},
                    cmap="Blues",
                    fmt=fmt,
                    linewidths=0.5
                    )

    ax.axhline(y=0, color='k', linewidth=4)
    ax.axhline(y=4, color='k', linewidth=4)
    ax.axvline(x=0, color='k', linewidth=4)
    ax.axvline(x=4, color='k', linewidth=4)

    ax.tick_params(direction='out', length=4, width=2, colors='k')
    ax.xaxis.set_ticks_position('top')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.tight_layout()

    cm_filename = 'cm' + '_' + \
                  str(learning_rate) + '_' + \
                  str(momentum) + '_' + \
                  str(epoch) + '_' + \
                  str(dropout_rate) + \
                  str(batch_size) + \
                  strftime("%d-%b-%Y-%H-%M-%S", gmtime()) + '.png'
    
    #plt.savefig(os.path.join(result_dir, cm_filename), format='png', dpi=600)                                
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

    n_seed            = 100
    n_node            = 100
    n_layer           = 10
    random_state      = 42
    verbose           = 0
    digit             = 3
    val_split         = 0.1
    count             = 0
    x_input           = range(7, 30)
    n_input           = len(x_input)
    n_output          = 4
    n_class           = n_output
    class0_ratio      = 1
    class1_ratio      = 1
    class2_ratio      = 1
    class3_ratio      = 0.2

    list_LR           = [0.1]            
    list_BM           = [0.98]            
    list_DR           = [0]
    list_BS           = [200]
    list_EP           = [90]
    list_NN           = [100]

    list_seed         = range(n_seed)
    #list_seed         = []
    
    kernel_init       = 'he_uniform' 
    optimizer         = 'adam'          
    loss              = 'sparse_categorical_crossentropy'
    output_activation = 'softmax'
    activation        = 'elu'  
    
    '''
    keranl initializer: 'he_uniform', 'lecun_normal', 'lecun_uniform'
    optimizer function: 'adam', 'adamax', 'nadam', 'sgd'
    loss function: 'categorical_crossentropy'
    activation function: LeakyReLU(alpha=alpha)
    '''

    # data and results path 
    project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning'
    result_dir  = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\grading\result'
    log_dir     = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\grading\log'             
    train_file  = 'Gleason.csv'

    # ----------------------------------------------------------------------------------
    # run the model
    # ----------------------------------------------------------------------------------
    
    print("Deep Neural Network for PCa grade classification: start...")

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    start = timeit.default_timer()
    
    data_path()

    breaking = False

    total_run = len(list_BM)*len(list_EP)*len(list_BS)*len(list_LR)*len(list_seed)

    for n in list_seed:

        for batch_size in list_BS:
            
            for momentum in list_BM:
                
                for epoch in list_EP:

                    for learning_rate in list_LR:

                        for dropout_rate in list_DR:

                            count += 1
                            print('\nRunning times: ' + str(count) + '/' + str(total_run))

                            np.random.seed(seed=n)
                            print('seed number:', str(n))
                          
                            x_train, x_val, x_test, y_train, y_val, y_test = PCa_data(
                                                                                      train_file,
                                                                                      project_dir,
                                                                                      random_state,
                                                                                      val_split,
                                                                                      class0_ratio, 
                                                                                      class1_ratio,
                                                                                      class2_ratio, 
                                                                                      class3_ratio,
                                                                                      x_input
                                                                                      ).data_manual_sample_split()

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

                            
                            cm, cm_norm = con_mat()
                            
                            #ACC, TPR, TNR, PPV, NPV, FPR, FNR, FDR = model_evaluation(cm, digit).statistics_2()
                            
                            plot_CM(cm_norm, '')

                            plot_CM(cm, 'd')

                            print('\ntest loss:', test_loss)
                            print('categorical test acc:', test_acc)
                            print('\nconfusion matrix:')
                            print(cm)
                            print('\nnorm confusion matrix:')
                            print(cm_norm)
                            print('\n', classification_report(y_test, y_pred_class, digits=3))
                            print('epochs:        ', epoch)
                            print('batch size:    ', batch_size)
                            print('dropout rate:  ', dropout_rate)
                            print('batch momentum:', momentum)
                            print('learning rate: ', learning_rate)
                            print('neuron numbers:', n_node)
                            print('layer numbers: ', n_layer)
                    
                            if test_acc > 0.8:
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


