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
import pandas as pd
import glob2 as glob
import numpy as np
import seaborn as sn
import string
import itertools
import matplotlib.pyplot as plt
from functools import partial
from datetime import datetime
import shutil
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector


print("\nDeep Neural Network for PCa grade classification: start...")

# ----------------------------------------------------------------------------------
# DNN paramters
# ----------------------------------------------------------------------------------
# key model parameters
n_epochs = 50
n_inputs = 18
n_outputs = 4
dropout_rate = 0
batch_size = 200
batch_momentum = 0.97
learning_rate = 0.00008

# routine parameters
d = 3
test_size = 0.5
random_state = 42
momentum = 0.97
display_step = 5
val_test_size = 0.3

# neuro numbers in each layer
n_neurons = 200
n_hidden1 = 250
n_hidden2 = 250
n_hidden3 = 200
n_hidden4 = 200
n_hidden5 = 150
n_hidden6 = 150
n_hidden7 = 150
n_hidden8 = 100
n_hidden9 = 100
n_hidden10 = 50

# ----------------------------------------------------------------------------------
# preparing data and folders
# ----------------------------------------------------------------------------------
print("loading data: start...")

# data path for windows system
project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning'
result_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\grading\result'
log_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\grading\log'

# # data path for linux or mac system
# project_dir = '/bmrp092temp/Prostate_Cancer_ex_vivo/Deep_Learning/'
# result_dir = '/bmrp092temp/Prostate_Cancer_ex_vivo/Deep_Learning/result/'
# log_dir = '/bmrp092temp/Prostate_Cancer_ex_vivo/Deep_Learning/log/'

if not os.path.exists(result_dir):
    print('result directory does not exist - creating...')
    os.makedirs(result_dir)
    print('log directory created.')
else:
    print('result directory already exists ...')

"""
overwrite the initial log directory, in case too many old summary files messed up
in single log_dir folder and caused problems in tensorboard.
"""

if not os.path.exists(log_dir):
       print('log directory does not exist - creating...')
       os.makedirs(log_dir)
       os.makedirs(log_dir + '/train')
       os.makedirs(log_dir + '/validation')
       print('log directory created.')
else:
    print('log directory already exists - overwriting...')
    tmp = tempfile.mktemp(dir=os.path.dirname(log_dir))
    shutil.move(log_dir, tmp)
    shutil.rmtree(log_dir, ignore_errors = True)
    os.makedirs(log_dir)
    os.makedirs(log_dir + '/train')
    os.makedirs(log_dir + '/validation')
    print('log directory overwritten.')

# ----------------------------------------------------------------------------------
# construct train, validation and test dataset
# ----------------------------------------------------------------------------------
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

# load data from csv files and define x and y
df = pd.read_csv(os.path.join(project_dir, 'Gleason.csv'))
# df = df[~df['Sub_ID'].str.contains("SH")]

# define label class
df.loc[df['ROI_Class'] == 'G1', 'y_cat'] = 0
df.loc[df['ROI_Class'] == 'G2', 'y_cat'] = 1
df.loc[df['ROI_Class'] == 'G3', 'y_cat'] = 2
df.loc[df['ROI_Class'] == 'G5', 'y_cat'] = 3
df.loc[df['ROI_Class'] == 'G4', 'y_cat'] = 4

# balance train and validation data from each class
class0 = df[df['y_cat'] == 0]
class0_sample = class0.sample(int(class0.shape[0]))
class1 = df[df['y_cat'] == 1]
class1_sample = class1.sample(int(class1.shape[0]))
class2 = df[df['y_cat'] == 2]
class2_sample = class2.sample(int(class2.shape[0]*0.7))
class3 = df[df['y_cat'] == 3]
class3_sample = class3.sample(int(class3.shape[0]*0.12))
class4 = df[df['y_cat'] == 4]
class4_sample = class3.sample(int(class4.shape[0]))

# reconstruct dataset from balanced data
df_2 = pd.concat([class0_sample, class1_sample, class2_sample, class3_sample])

# select train, validation and test features 
X = df_2.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 24, 25, 26, 27, 28, 29]]

# define train, validation and test labels
Y = df_2.y_cat.astype('int')

# construct train dataset with 70% slpit
x_train, x_val_test, y_train, y_val_test = train_test_split(
                                                            X,
                                                            Y,
                                                            test_size=val_test_size,
                                                            random_state=random_state
)

# construct validation and test dataset with 50% split
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

# reset tensorflow graph
def reset_tf_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_tf_graph()

# set placeholders
with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    
    """
    The training placeholder. During training, we'll set it to True.
    It's used to tell the tf.layers.batch_normalization() function
    whether it should use the current mini-batch or the whole trainset
    mean and standard deviation (when testing - set it to False)
    """
    training = tf.placeholder_with_default(False, shape=(), name="train")

# ----------------------------------------------------------------------------------
# construct DNN model with batch normalization layers and dropout layers
# ----------------------------------------------------------------------------------
with tf.name_scope("dnn"):

    # choose He initializer (default is Xavier initializer)
    he_init = tf.contrib.layers.variance_scaling_initializer()

    # create fully connected neural netwrok with dense layer function
    dense_layer = partial(
                          tf.layers.dense,
                          kernel_initializer=he_init
    )
    # create batch normalization
    batch_normalization_layer = partial(
                                        tf.layers.batch_normalization,
                                        training=training,
                                        momentum=batch_momentum
    )

    # construct DNN with batch normalization and dropout layers
    hidden1 = dense_layer(x, n_hidden1, name="hidden1")
    bn1 = batch_normalization_layer(hidden1, name="bn1")
    tf.summary.histogram("batch_normalization", bn1)
    bn1_act = tf.nn.elu(bn1, name="elu_bn1")
    hidden1_drop = tf.layers.dropout(bn1_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn1_act)

    hidden2 = dense_layer(hidden1_drop, n_hidden2, name="hidden2")
    bn2 = batch_normalization_layer(hidden2, name="bn2")
    tf.summary.histogram("batch_normalization", bn2)
    bn2_act = tf.nn.elu(bn2, name="elu_bn2")
    hidden2_drop = tf.layers.dropout(bn2_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn2_act)

    hidden3 = dense_layer(hidden2_drop, n_hidden3, name="hidden3")
    bn3 = batch_normalization_layer(hidden3, name="bn3")
    tf.summary.histogram("batch_normalization", bn3)
    bn3_act = tf.nn.elu(bn3, name="elu_bn3")
    hidden3_drop = tf.layers.dropout(bn3_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn3_act)

    hidden4 = dense_layer(hidden3_drop, n_hidden4, name="hidden4")
    bn4 = batch_normalization_layer(hidden4, name="bn4")
    tf.summary.histogram("batch_normalization", bn4)
    bn4_act = tf.nn.elu(bn4, name="elu_bn4")
    hidden4_drop = tf.layers.dropout(bn4_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn4_act)

    hidden5 = dense_layer(hidden4_drop, n_hidden5, name="hidden5")
    bn5 = batch_normalization_layer(hidden5, name="bn5")
    tf.summary.histogram("batch_normalization", bn5)
    bn5_act = tf.nn.elu(bn5, name="elu_bn5")
    hidden5_drop = tf.layers.dropout(bn5_act, dropout_rate, training=training)
    tf.summary.histogram("activations", bn5_act)

    hidden6 = dense_layer(hidden5_drop, n_hidden6, name="hidden6")
    bn6 = batch_normalization_layer(hidden6, name="bn6")
    bn6_act = tf.nn.elu(bn5, name="elu_bn6")
    hidden6_drop = tf.layers.dropout(bn6_act, dropout_rate, training=training)

    hidden7 = dense_layer(hidden6_drop, n_hidden7, name="hidden7")
    bn7 = batch_normalization_layer(hidden7, name="bn7")
    bn7_act = tf.nn.elu(bn7, name="elu_bn7")
    hidden7_drop = tf.layers.dropout(bn7_act, dropout_rate, training=training)

    hidden8 = dense_layer(hidden7_drop, n_hidden8, name="hidden8")
    bn8 = batch_normalization_layer(hidden8, name="bn8")
    bn8_act = tf.nn.elu(bn8, name="elu_bn8")
    hidden8_drop = tf.layers.dropout(bn8_act, dropout_rate, training=training)

    hidden9 = dense_layer(hidden8_drop, n_hidden9, name="hidden9")
    bn9 = batch_normalization_layer(hidden9, name="bn9")
    bn9_act = tf.nn.elu(bn9, name="elu_bn9")
    hidden9_drop = tf.layers.dropout(bn9_act, dropout_rate, training=training)

    hidden10 = dense_layer(hidden9_drop, n_hidden10, name="hidden10")
    bn10 = batch_normalization_layer(hidden10, name="bn10")
    bn10_act = tf.nn.elu(bn10, name="elu_bn10")
    hidden10_drop = tf.layers.dropout(bn10_act, dropout_rate, training=training)

    logits_before_bn = dense_layer(hidden10_drop, n_outputs, name="outputs")
    logits = batch_normalization_layer(logits_before_bn, name="bn11")


# ----------------------------------------------------------------------------------
# create loss function with cross entropy
# ----------------------------------------------------------------------------------
with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                   logits=logits)
    
    with tf.name_scope("total"):
        loss = tf.reduce_mean(cross_entropy, name="loss")
tf.summary.scalar('cross_entropy', loss)

# ----------------------------------------------------------------------------------
# select different optimization function
# ----------------------------------------------------------------------------------
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    
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

# ----------------------------------------------------------------------------------
# define accuracy
# ----------------------------------------------------------------------------------
with tf.name_scope("evaluation"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.nn.in_top_k(logits, y, 1)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Create a collection containing all important operations.
# It makes it easier for other people to reuse the trained model
for op in (x, y, accuracy, train_op):
    tf.add_to_collection("important_ops", op)

print("deep neural network construction: complete!!!")

# ----------------------------------------------------------------------------------
# create tensorflow session for DNN model training
# ----------------------------------------------------------------------------------  
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()

    # creat tensorflow summary writer for train and validation
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(log_dir + '/validation', sess.graph)
    saver = tf.train.Saver()

    # ----------------------------------------------------------------------------------
    # start DNN model training
    # ----------------------------------------------------------------------------------  
    print('DNN Training: start...')
    for epoch in range(n_epochs):
        n_batches = train_size // batch_size
        for iteration in range(n_batches):
            randidx = np.random.randint(int(train_size), size=batch_size)
            x_batch = x_train.iloc[randidx, :]
            y_batch = y_train.iloc[randidx]

            # For operations depending on batch normalization, set the training placeholder to True
            sess.run(
                     [train_op, extra_update_ops],
                     feed_dict={training: True,
                                       x: x_batch,
                                       y: y_batch}
            )

        # output the data into tensorboard summaries every 10 epochs
        if epoch % display_step == 0:
            
            train_summary, train_accuracy = sess.run([merged_summary, accuracy],
                                                     feed_dict={x: x_batch,
                                                                y: y_batch})
            
            val_summary, val_accuracy = sess.run([merged_summary, accuracy],
                                                 feed_dict={x: x_val,
                                                            y: y_val})
            
            train_writer.add_summary(train_summary, epoch)
            validation_writer.add_summary(val_summary, epoch)
            
            print(
                  "Epoch:", epoch,
                  "Train accuracy:", np.around(train_accuracy, 2),
                  "Validation accuracy:", np.around(val_accuracy, 2)
            )

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
            
    # ----------------------------------------------------------------------------------
    # creat tensorboard embedding
    # ----------------------------------------------------------------------------------            
    # with tf.device("/cpu:0"):
    #    tf_embedding = tf.Variable(embedding, trainable = False, name = "embedding")
    
    # write labels
    metadata = os.path.join(log_dir, 'metadata.tsv')
    with open(metadata, 'w') as metadata_file:
        for i in range(len(y_val)):
            metadata_file.write('%d\n' % i)

    # create a tensorboard summary writer
    writer = tf.summary.FileWriter(log_dir, sess.graph)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = "embedding:0"
    embedding.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

    # save the model
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_dir, "model.ckpt"))

    train_writer.close()
    validation_writer.close()

    ### evaluating final accuracy of the model
    accuracy_train = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
    accuracy_val = sess.run(accuracy, feed_dict={x: x_val, y: y_val})
    accuracy_test = sess.run(accuracy, feed_dict={x: x_test, y: y_test})

    print('\nfinal training accuracy of the model:', accuracy_train)
    print('final training accuracy of the model:', accuracy_val)
    print('final training accuracy of the model:', accuracy_test)
    
    # ----------------------------------------------------------------------------------
    # calculate and plot confusion matrix
    # ----------------------------------------------------------------------------------
    y_pred = tf.argmax(logits.eval(feed_dict={x: x_val}), axis=1)
    y_prediction = sess.run(y_pred)

    con_mat = tf.confusion_matrix(
                                  labels=y_val,
                                  predictions=y_pred,
                                  num_classes=n_outputs,
                                  dtype=tf.int32,
                                  name="confusion_matrix"
    )

    cm_1 = sess.run(con_mat)
    cm_2 = cm_1.astype('float')/cm_1.sum(axis=1)[:, np.newaxis]
    cm_2 = np.around(cm_2, 2)
    print("confusion matrix: print...")
    print(cm_1)
    print("normalized confusion matrix: print...")
    print(cm_2)
    print("precision, recall, f1-score: print...")
    print(classification_report(y_val, y_prediction, digits=d))

    # plot confusion matrix
    ax_2 = sn.heatmap(cm_2, annot=True, annot_kws={"size": 15}, cmap="Blues", linewidths=.5)
    # plt.figure(figsize = (10,7))
    # sn.set(font_scale=1.4) #for label size
    # plt.ylabel('True label', fontsize=13, fontweight='bold')
    # plt.xlabel('Predicted label',fontsize=13, fontweight='bold')
    ax_2.axhline(y=0, color='k', linewidth=3)
    ax_2.axhline(y=4, color='k', linewidth=3)
    ax_2.axvline(x=0, color='k', linewidth=3)
    ax_2.axvline(x=4, color='k', linewidth=3)
    ax_2.set_aspect('equal')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), format='png', dpi=600)
    plt.tight_layout()
    # plt.show()
    # plt.close()
    print("plotting confusion matrix_2: complete!")
    
    # ----------------------------------------------------------------------------------
    # construt a ROC curve
    # ----------------------------------------------------------------------------------
##    fpr = dict()
##    tpr = dict()
##    threshold = dict()
##    roc_auc = dict()
##    for i in range(n_outputs):
##        fpr[i], tpr[i], threshold[i] = roc_curve(y_val[:, i], y_prediction[:, i])
##        roc_auc[i] = auc(fpr[i], tpr[i])
##        print(tpr[i])
##        print(fpr[i])
##        print(roc_auc[i])
##        ax = fig.add_subplot(4, 1, i)
##        print()
##        plt.title('Receiver Operating Characteristic')
##        plt.plot(fpr[i], tpr[i], 'b', label='AUC = %0.2f' % roc_auc[i])
##        plt.legend(loc='lower right')
##        plt.plot([0, 1], [0, 1], 'r--')
##        plt.xlim([0, 1])
##        plt.ylim([0, 1])
##        plt.ylabel('True Positive Rate')
##        plt.xlabel('False Positive Rate')
##        plt.show()

    save_path = saver.save(sess, log_dir)


# ----------------------------------------------------------------------------------
# DNN traning complete
# ----------------------------------------------------------------------------------
sess.close()
print('session close!')
print("Run the command line:\n"\
      "--> tensorboard --logdir="\
      "\nThen open http://localhost:6006/ into your web browser")
print("deep neural network classification: complete!")



