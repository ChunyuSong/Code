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
n_epochs = 100
d = 3
n_inputs = 11
n_outputs = 5
n_neurons = 200
dropout_rate = 0.1
learning_rate = 0.0001
batch_momentum = 0.97
batch_size = 200
random_state = 42
test_size = 0.5
val_test_size = 0.2
display_step = 10

n_hidden1 = 200
n_hidden2 = 200
n_hidden3 = 200
n_hidden4 = 200
n_hidden5 = 200
n_hidden6 = 200
n_hidden7 = 200
n_hidden8 = 200
n_hidden9 = 200
n_hidden10 = 200

print("loading data: start...")
# working directory in windows system
project_dir = r'\\10.39.42.102\temp\2019_MS\AI'
result_dir = r'\\10.39.42.102\temp\2019_MS\AI\result'
log_dir = r'\\10.39.42.102\temp\2019_MS\AI\log'

# working directory in Mac system
# project_dir = '/Users/Desktop/deep_learning'
# results_dir = '/Users/Desktop/deep_learning'

if not os.path.exists(log_dir):
    print('log directory does not exist - creating...')
    os.makedirs(log_dir)
    os.makedirs(log_dir + '/train')
    os.makedirs(log_dir + '/validation')
    print('log directory created.')
else:
    print('log directory already exists ...')

if not os.path.exists(result_dir):
    print('result directory does not exist - creating...')
    os.makedirs(result_dir)
    print('result directory created.')
else:
    print('result directory already exists ...')

# all the result lists
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

df = pd.read_csv(os.path.join(project_dir, '20190302.csv'))

df.loc[df['ROIClass'] == 1, 'y_cat'] = 0
df.loc[df['ROIClass'] == 2, 'y_cat'] = 1
df.loc[df['ROIClass'] == 4, 'y_cat'] = 2
df.loc[df['ROIClass'] == 5, 'y_cat'] = 3
df.loc[df['ROIClass'] == 6, 'y_cat'] = 4

class0 = df[df['y_cat'] == 0]
class0_sample = class0.sample(int(class0.shape[0]))
class1 = df[df['y_cat'] == 1]
class1_sample = class1.sample(int(class1.shape[0]))
class2 = df[df['y_cat'] == 2]
class2_sample = class2.sample(int(class2.shape[0]))
class3 = df[df['y_cat'] == 3]
class3_sample = class3.sample(int(class3.shape[0]))
class4 = df[df['y_cat'] == 4]
class4_sample = class4.sample(int(class4.shape[0]))

df_2 = pd.concat([class0_sample, class1_sample, class2_sample, class3_sample, class4_sample])

Y = df_2.y_cat.astype('int')
X = df_2.iloc[:, [16, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29]]         # DBSI + FLAIR + MPRAGE
# X = data.iloc[:, [28, 29]]                                           # FLAIR + MPRAGE
# X = data.iloc[:, [28, 29, 30]]                                       # MTC + FLAIR + MPRAGE

x_train, x_val_test, y_train, y_val_test = train_test_split(
                                                            X,
                                                            Y,
                                                            test_size=val_test_size,
                                                            random_state=random_state
)

x_val, x_test, y_val, y_test = train_test_split(
                                                x_val_test,
                                                y_val_test,
                                                test_size=test_size,
                                                random_state=random_state
)

train_size = len(x_train)
print("data loading: complete!")
print("training set size:", len(x_train))
print("validation set size:", len(x_val))
print("test set size:", len(x_test))

def reset_tf_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_tf_graph()

with tf.name_scope("input"):
    x = tf.placeholder(tf.float32, shape=(None, n_inputs), name="x")
    y = tf.placeholder(tf.int64, shape=(None), name="y")
    training = tf.placeholder_with_default(False, shape=(), name="training")

with tf.name_scope("dnn"):
    he_init = tf.contrib.layers.variance_scaling_initializer()

    dense_layer = partial(
                          tf.layers.dense,
                          kernel_initializer=he_init,
                          kernel_regularizer=None,
                          trainable=True,
    )

    batch_normalization = partial(
                                  tf.layers.batch_normalization,
                                  training=training,
                                  momentum=batch_momentum,
                                  epsilon=0.001,
                                  center=True,
                                  scale=True
    )

    hidden1 = dense_layer(x, n_hidden1, name="hidden1")
    bn1 = batch_normalization(hidden1, name="bn1")
    tf.summary.histogram("batch_normalization", bn1)
    bn1_act = tf.nn.elu(bn1, name="elu_bn1")
    tf.summary.histogram("activations", bn1_act)

    hidden2 = dense_layer(bn1_act, n_hidden2, name="hidden2")
    bn2 = batch_normalization(hidden2, name="bn2")
    tf.summary.histogram("batch_normalization", bn2)
    bn2_act = tf.nn.elu(bn2, name="elu_bn2")
    hidden2_drop = tf.layers.dropout(bn2_act, rate=dropout_rate, training=training)
    tf.summary.histogram("activations", bn2_act)

    hidden3 = dense_layer(hidden2_drop, n_hidden3, name="hidden3")
    bn3 = batch_normalization(hidden3, name="bn3")
    tf.summary.histogram("batch_normalization", bn3)
    bn3_act = tf.nn.elu(bn3, name="elu_bn3")
    tf.summary.histogram("activations", bn3_act)

    hidden4 = dense_layer(bn3_act, n_hidden4, name="hidden4")
    bn4 = batch_normalization(hidden4, name="bn4")
    tf.summary.histogram("batch_normalization", bn4)
    bn4_act = tf.nn.elu(bn4, name="elu_bn4")
    hidden4_drop = tf.layers.dropout(bn4_act, rate=dropout_rate, training=training)
    tf.summary.histogram("activations", bn4_act)

    hidden5 = dense_layer(hidden4_drop, n_hidden5, name="hidden5")
    bn5 = batch_normalization(hidden5, name="bn5")
    tf.summary.histogram("batch_normalization", bn5)
    bn5_act = tf.nn.elu(bn5, name="elu_bn5")
    tf.summary.histogram("activations", bn5_act)

    hidden6 = dense_layer(bn5_act, n_hidden6, name="hidden6")
    bn6 = batch_normalization(hidden6, name="bn6")
    bn6_act = tf.nn.elu(bn5, name="elu_bn6")
    hidden6_drop = tf.layers.dropout(bn6_act, rate=dropout_rate, training=training)

    hidden7 = dense_layer(hidden4_drop, n_hidden7, name="hidden7")
    bn7 = batch_normalization(hidden7, name="bn7")
    bn7_act = tf.nn.elu(bn7, name="elu_bn7")

    hidden8 = dense_layer(bn7_act, n_hidden8, name="hidden8")
    bn8 = batch_normalization(hidden8, name="bn8")
    bn8_act = tf.nn.elu(bn8, name="elu_bn8")
    hidden4_drop = tf.layers.dropout(bn8_act, dropout_rate, training=training)

    hidden9 = dense_layer(hidden4_drop, n_hidden9, name="hidden9")
    bn9 = batch_normalization(hidden9, name="bn9")
    bn9_act = tf.nn.elu(bn9, name="elu_bn9")

    hidden10 = dense_layer(bn9_act, n_hidden10, name="hidden10")
    bn10 = batch_normalization(hidden10, name="bn10")
    bn10_act = tf.nn.elu(bn10, name="elu_bn10")
    hidden10_drop = tf.layers.dropout(bn10_act, rate=dropout_rate, training=training)

    logits_before_bn = dense_layer(hidden10_drop, n_outputs, name="outputs")
    logits = batch_normalization(logits_before_bn, name="bn11")

with tf.name_scope("cross_entropy"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                   logits=logits)
    with tf.name_scope("total"):
        loss = tf.reduce_mean(cross_entropy, name="loss")
tf.summary.scalar('cross_entropy', loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss)

with tf.name_scope("evaluation"):
    with tf.name_scope("correct_prediction"):
        correct_prediction = tf.nn.in_top_k(logits, y, 1)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)
print("DNN successfully built!")

for op in (x, y, accuracy, train_op):
    tf.add_to_collection("important_ops", op)

# create session
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    init.run()

    # Merge all the summaries and write them out to tensorboard folder
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    val_writer = tf.summary.FileWriter(log_dir + '/val', sess.graph)
    saver = tf.train.Saver()

    print('Training DNN: start...')

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
            val_writer.add_summary(val_summary, epoch)
            print("Epoch:", epoch,
                  "Train accuracy:", np.around(train_accuracy, 2),
                  "Test accuracy:", np.around(val_accuracy, 2))

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

        else:  # Record a summary
            summary, _ = sess.run([merged_summary, train_op],
                                  feed_dict={x: x_batch,
                                             y: y_batch})
            train_writer.add_summary(summary, epoch)

    saver.save(sess, os.path.join(log_dir, 'batch_normalization_dnn.ckpt'))
    train_writer.close()
    val_writer.close()

    print('\nEvaluating final accuracy of the model (1/3)')
    train_accuracy = sess.run(accuracy, feed_dict={x: x_train,
                                                   y: y_train})

    print('Evaluating final accuracy of the model (2/3)')
    test_accuracy = sess.run(accuracy, feed_dict={x: x_val,
                                                  y: y_val})

    print('Evaluating final accuracy of the model (3/3)')
    val_accuracy = sess.run(accuracy, feed_dict={x: x_test,
                                                 y: y_test})

    #calculate confusion matrix and make the plots
    y_pred = tf.argmax(logits.eval(feed_dict={x: x_test}), axis=1)

    con_mat = tf.confusion_matrix(
                                  labels=y_test,
                                  predictions=y_pred,
                                  num_classes=n_outputs,
                                  dtype=tf.int32,
                                  name="confusion_matrix"
    )

    cm_1 = sess.run(con_mat)
    cm_2 = cm_1.astype('float')/cm_1.sum(axis=1)[:, np.newaxis]
    cm_2 = np.around(cm_2, 2)
    print("\nconfusion matrix: print...")
    print(cm_1)
    print("\nnormalized confusion matrix: print...")
    print(cm_2)
    print("\nprecision, recall, f1-score: print...")
    y_prediction = sess.run(y_pred)
    print(classification_report(y_test, y_prediction, digits=d))
    ax_2 = sn.heatmap(cm_2, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.5)
    ax_2.axhline(y=0, color='k', linewidth=3)
    ax_2.axhline(y=5, color='k', linewidth=3)
    ax_2.axvline(x=0, color='k', linewidth=3)
    ax_2.axvline(x=5, color='k', linewidth=3)
    ax_2.set_aspect('equal')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), format='png', dpi=600)
    plt.tight_layout()
    # plt.show()
    # plt.close()
    print("plotting confusion matrix_2: complete!")

    save_path = saver.save(sess, log_dir)

sess.close()
# file_writer.close()
print('session close!')
print("Run the command line:\n"\
      "--> tensorboard --logdir="\
      "\nThen open http://RADEST03BMR042:6006/ into your web browser")







