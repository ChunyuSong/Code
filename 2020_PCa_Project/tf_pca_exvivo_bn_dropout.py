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
#----------------------------------------------------------------------


import pandas as pd
import glob2 as glob
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import string
from matplotlib.pylab import *
import os
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from functools import partial
from datetime import datetime
import shutil


print("\nDeep Neural Network for PCa grade classification: start...")

# DNN parameters
n_epochs = 50
d = 3
n_inputs = 22
n_outputs = 3
test_size = 0.5              # split validation_test set into validation and test set
n_neurons = 200
dropout_rate = 0.2
learning_rate = 0.0001
batch_momentum = 0.97      # momentum value is greater or equal to 0.9 (very close to 1)
batch_size = 100            # larger data sets, smaller mini-batches -> bigger value of momentum
random_state = 42
display_step = 10
momentum = 0.97
n_hidden1 = n_neurons
n_hidden2 = n_neurons
n_hidden3 = n_neurons
n_hidden4 = n_neurons
n_hidden5 = n_neurons
n_hidden6 = n_neurons
n_hidden7 = n_neurons
n_hidden8 = n_neurons
n_hidden9 = n_neurons
n_hidden10 = n_neurons

print("loading data: start...")

# data and results path for windows system
project_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\data'
result_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\result'
log_dir = r'\\10.39.42.102\temp\Zezhong_Ye\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification\log'

# data path for linux system
# project_dir = '/bmrp092temp/Zezhong_Ye/Prostate_Cancer_Project_Shanghai/PCa_Machine_Learning/PCa_Benign_Classification/data/'
# result_dir = '/bmrp092temp/Zezhong_Ye/Prostate_Cancer_Project_Shanghai/PCa_Machine_Learning/PCa_Benign_Classification/data/'
# log_dir = '/bmrp092temp/Zezhong_Ye/Prostate_Cancer_Project_Shanghai/PCa_Machine_Learning/PCa_Benign_Classification/data/'

if not os.path.exists(log_dir):
    print('log directory does not exist - creating...')
    os.makedirs(log_dir)
    os.makedirs(log_dir + '/train')
    os.makedirs(log_dir + '/validation')
    print('log directory created.')
else:
    print('log directory already exists ...')

if not os.path.exists(result_dir):
    print('results directory does not exist - creating...')
    os.makedirs(result_dir)
    print('results directory created.')
else:
    print('result directory already exists ...')

# all the result lists
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

# construct training dataset dataset
df_1 = pd.read_csv(os.path.join(project_dir, 'benign_mpMRI.csv'))
df_2 = pd.read_csv(os.path.join(project_dir, 'PCa_train.csv'))
df_3 = df_1.append(df_2)
df_3.loc[df_3['ROI_Class'] == 'p', 'y_cat'] = 0
df_3.loc[df_3['ROI_Class'] == 'c', 'y_cat'] = 1
df_3.loc[df_3['ROI_Class'] == 't', 'y_cat'] = 2

x_train = df_3.iloc[:, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                        19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]

y_train = df_3.y_cat.astype('int')

# construct validation and testing dataset
df_4 = pd.read_csv(os.path.join(project_dir, 'benign_biopsy.csv'))
df_5 = pd.read_csv(os.path.join(project_dir, 'PCa_test.csv'))
df_6 = df_4.append(df_5)
df_6.loc[df_6['ROI_Class'] == 'p', 'y_cat'] = 0
df_6.loc[df_6['ROI_Class'] == 'c', 'y_cat'] = 1
df_6.loc[df_6['ROI_Class'] == 't', 'y_cat'] = 2

class0 = df_6[df_6['y_cat'] == 0]
class0_sample = class0.sample(int(class0.shape[0]*0.647))
class1 = df_6[df_6['y_cat'] == 1]
class1_sample = class1.sample(int(class1.shape[0]*0.706))
class2 = df_6[df_6['y_cat'] == 2]
class2_sample = class2.sample(int(class2.shape[0]))

df_7 = pd.concat([class0_sample, class1_sample, class2_sample])

x_val_test = df_7.iloc[:, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                           19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]

y_val_test = df_7.y_cat.astype('int')

# construct validation set and test set with 1:1 ratio
x_val, x_test, y_val, y_test = train_test_split(
                                                x_val_test,
                                                y_val_test,
                                                test_size=test_size,
                                                random_state=random_state
)

train_size = len(x_train)
print("data loading: complete!")
print("train set size:", len(x_train))
print("validation set size:", len(x_val))
print("test set size:", len(x_test))
print("loading data from csv file: complete!!!")
print("deep neuronetwork construction: start...")

def reset_tf_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_tf_graph()

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    training = tf.placeholder_with_default(False, shape=(), name="train")

with tf.name_scope("dnn"):
    he_init = tf.contrib.layers.variance_scaling_initializer()

    dense_layer = partial(
                          tf.layers.dense,
                          kernel_initializer=he_init
    )

    batch_normalization_layer = partial(
                                        tf.layers.batch_normalization,
                                        training=training,
                                        momentum=batch_momentum
    )

    hidden1 = dense_layer(x, n_hidden1, name="hidden1")
    bn1 = batch_normalization_layer(hidden1, name="bn1")
    tf.summary.histogram("batch_normalization", bn1)
    bn1_act = tf.nn.elu(bn1, name="elu_bn1")
    # bn1_act = tf.nn.leaky_relu(bn1, alpha=0.2, name="leaky_relu_bn1")
    hidden1_drop = tf.layers.dropout(bn1_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn1_act)


    hidden2 = dense_layer(hidden1_drop, n_hidden2, name="hidden2")
    bn2 = batch_normalization_layer(hidden2, name="bn2")
    tf.summary.histogram("batch_normalization", bn2)
    bn2_act = tf.nn.elu(bn2, name="elu_bn2")
    # bn2_act = tf.nn.leaky_relu(bn2, alpha=0.2, name="leaky_relu_bn2")
    hidden2_drop = tf.layers.dropout(bn2_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn2_act)

    hidden3 = dense_layer(bn2_act, n_hidden3, name="hidden3")
    bn3 = batch_normalization_layer(hidden3, name="bn3")
    tf.summary.histogram("batch_normalization", bn3)
    bn3_act = tf.nn.elu(bn3, name="elu_bn3")
    # bn3_act = tf.nn.leaky_relu(bn3, alpha=0.2, name="leaky_relu_bn3")
    hidden3_drop = tf.layers.dropout(bn3_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn3_act)

    hidden4 = dense_layer(hidden3_drop, n_hidden4, name="hidden4")
    bn4 = batch_normalization_layer(hidden4, name="bn4")
    tf.summary.histogram("batch_normalization", bn4)
    bn4_act = tf.nn.elu(bn4, name="elu_bn4")
    # bn4_act = tf.nn.leaky_relu(bn4, alpha=0.2, name="leaky_relu_bn4")
    hidden4_drop = tf.layers.dropout(bn4_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn4_act)

    hidden5 = dense_layer(bn4_act, n_hidden5, name="hidden5")
    bn5 = batch_normalization_layer(hidden5, name="bn5")
    tf.summary.histogram("batch_normalization", bn5)
    bn5_act = tf.nn.elu(bn5, name="elu_bn5")
    # bn5_act = tf.nn.leaky_relu(bn5, alpha=0.2, name="leaky_relu_bn5")
    hidden5_drop = tf.layers.dropout(bn5_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn5_act)

    hidden6 = dense_layer(hidden5_drop, n_hidden6, name="hidden6")
    bn6 = batch_normalization_layer(hidden6, name="bn6")
    bn6_act = tf.nn.elu(bn5, name="elu_bn6")
    hidden6_drop = tf.layers.dropout(bn6_act, dropout_rate, training=training)

    hidden7 = dense_layer(bn6_act, n_hidden7, name="hidden7")
    bn7 = batch_normalization_layer(hidden7, name="bn7")
    bn7_act = tf.nn.elu(bn7, name="elu_bn7")
    hidden7_drop = tf.layers.dropout(bn7_act, dropout_rate, training=training)

    hidden8 = dense_layer(hidden7_drop, n_hidden8, name="hidden8")
    bn8 = batch_normalization_layer(hidden8, name="bn8")
    bn8_act = tf.nn.elu(bn8, name="elu_bn8")
    hidden8_drop = tf.layers.dropout(bn8_act, dropout_rate, training=training)

    hidden9 = dense_layer(bn8_act, n_hidden9, name="hidden9")
    bn9 = batch_normalization_layer(hidden9, name="bn9")
    bn9_act = tf.nn.elu(bn9, name="elu_bn9")
    hidden9_drop = tf.layers.dropout(bn9_act, dropout_rate, training=training)

    hidden10 = dense_layer(hidden9_drop, n_hidden10, name="hidden10")
    bn10 = batch_normalization_layer(hidden10, name="bn10")
    bn10_act = tf.nn.elu(bn10, name="elu_bn10")
    hidden10_drop = tf.layers.dropout(bn10_act, dropout_rate, training=training)

    logits_before_bn = dense_layer(bn10_act, n_outputs, name="outputs")
    logits = batch_normalization_layer(logits_before_bn, name="bn11")

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                   logits=logits)
    with tf.name_scope("total"):
        loss = tf.reduce_mean(cross_entropy, name="loss")
tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
    #                                       momentum=momentum,
    #                                       decay=0.9,
    #                                       epsilon=1e-10)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
    #                                        momentum=momentum,
    #                                        use_nesterov=True)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss)

with tf.name_scope("evaluation"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.nn.in_top_k(logits, y, 1)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

for op in (x, y, accuracy, train_op):
    tf.add_to_collection("important_ops", op)

print("deep neural network construction: complete!!!")

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()

    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(log_dir + '/validation', sess.graph)
    saver = tf.train.Saver()

    print('Training deep neural network: start....')
    for epoch in range(n_epochs):
        n_batches = train_size // batch_size
        for iteration in range(n_batches):
            randidx = np.random.randint(int(train_size), size=batch_size)
            x_batch = x_train.iloc[randidx, :]
            y_batch = y_train.iloc[randidx]

            sess.run(
                     [train_op, extra_update_ops],
                     feed_dict={training: True, x: x_batch, y: y_batch}
            )

        # output the data into TensorBoard summaries every 10 epochs
        if epoch % display_step == 0:
            train_summary, train_accuracy = sess.run([merged_summary, accuracy],
                                                     feed_dict={x: x_batch,
                                                                y: y_batch})

            val_summary, val_accuracy = sess.run([merged_summary, accuracy],
                                                 feed_dict={x: x_val,
                                                            y: y_val})

            train_writer.add_summary(train_summary, epoch)
            validation_writer.add_summary(val_summary, epoch)
            print("Epoch:", epoch,
                  "Train accuracy:", np.around(train_accuracy, 2),
                  "Validation accuracy:", np.around(val_accuracy, 2))

        # output metadata every 100 epochs
        if epoch % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run(
                                  [merged_summary, train_op],
                                  feed_dict={x: x_batch, y: y_batch},
                                  options=run_options,
                                  run_metadata=run_metadata
            )
            train_writer.add_run_metadata(run_metadata, 'step%03d' % epoch)
            train_writer.add_summary(summary, epoch)
            print('adding run metadata for', epoch)

        # Record a summary
        else:
            summary, _ = sess.run(
                                  [merged_summary, train_op],
                                  feed_dict={x: x_batch,
                                             y: y_batch}
            )
            train_writer.add_summary(summary, epoch)

    saver.save(sess, os.path.join(log_dir, 'batch_normalization_dnn.ckpt'))
    train_writer.close()
    validation_writer.close()

    # print('\nEvaluating final accuracy of the model (1/3)')
    # train_accuracy = sess.run(accuracy, feed_dict={x: x_train,
    #                                                y: y_train)
    #
    # print('Evaluating final accuracy of the model (2/3)')
    # test_accuracy = sess.run(accuracy, feed_dict={x: x_val,
    #                                               y: y_val)
    #
    # print('Evaluating final accuracy of the model (3/3)')
    # val_accuracy = sess.run(accuracy, feed_dict={x: x_test,
    #                                              y: y_test)

    # calculate confusion matrix and make the plots
    y_pred = tf.argmax(logits.eval(feed_dict={x: x_val}), axis=1)

    con_mat = tf.confusion_matrix(
                                  labels=y_val,
                                  predictions=y_pred,
                                  num_classes=n_outputs,
                                  dtype=tf.int32,
                                  name="confusion_matrix"
    )

    with sess.as_default():
        cm_1 = sess.run(con_mat)
        cm_2 = cm_1.astype('float')/cm_1.sum(axis=1)[:, np.newaxis]
        cm_2 = np.around(cm_2, 2)
        print("\nconfusion matrix: print...")
        print(cm_1)
        print("\nnormalized confusion matrix: print...")
        print(cm_2)
        print("\nprecision, recall, f1-score: print...")
        y_prediction = sess.run(y_pred)
        print(classification_report(y_val, y_prediction, digits=d))
        ax_2 = sn.heatmap(cm_2, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.5)
        # plt.figure(figsize = (10,7))
        # sn.set(font_scale=1.4)#for label size
        # plt.ylabel('True label', fontsize=13, fontweight='bold')
        # plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
        ax_2.axhline(y=0, color='k', linewidth=3)
        ax_2.axhline(y=3, color='k', linewidth=3)
        ax_2.axvline(x=0, color='k', linewidth=3)
        ax_2.axvline(x=3, color='k', linewidth=3)
        ax_2.set_aspect('equal')
        plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), format='png', dpi=600)
        plt.tight_layout()
        # plt.show()
        # plt.close()
        print("plotting confusion matrix_2: complete!")

    save_path = saver.save(sess, log_dir)

sess.close()
print('session close!')
print("Run the command line:\n"\
      "--> tensorboard --logdir="\
      "\nThen open http://RADEST03BMR042:6006/ into your web browser")
print("deep neural network classification: complete!")


# initial_learning_rate = 0.1
# decay_steps = 10000
# decay_rate = 1/10
# global_step = tf.Variable(0, trainable=False, name="global_step")
# learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step,
# decay_steps, decay_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
# training_op = optimizer.minimize(loss, global_step=global_step)

# Calculate batch normalization for the inputs of the second hidden layer
    # The BN algorithm uses 'exponential decay' to compute the running averages.
    # 'momentum' parameter is used for this purpose.
    # Given a value v then the running average v.hat is upated as follow:
    #   v.hat <- v.hat * momentum + v * (1 - momentum)
    # Normally, momentum value is greater or equal to 0.9 (very close to 1)
    # Larger datasets, smaller mini-batches -> bigger value of momentum





