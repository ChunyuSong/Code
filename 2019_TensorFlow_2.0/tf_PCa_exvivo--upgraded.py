#----------------------------------------------------------------------------------------------------------------
# deep learning classifier using a multiple layer perceptron (MLP)
# batch normalization was used
#
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
#-------------------------------------------------------------------------------------------------------------

import os
import pandas as pd
import glob2 as glob
import numpy as np
import itertools
import seaborn as sn
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from functools import partial
import shutil


print("Deep Neural Network for PCa grade classification: start...")

# DNN parameters

n_epochs = 30
learning_rate = 0.001
batch_momentum = 0.97
dropout_rate = 0.2
batch_size = 200
display_step = 5
n_inputs = 18
n_outputs = 2

d = 3
seed = 42
test_size = 0.5
val_test_size = 0.2
n_neurons = 200
momentum = 0.97
random_state = 42

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
project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning'
result_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\result'
log_dir = r'\\10.39.42.102\temp\Prostate_Cancer_ex_vivo\Deep_Learning\log'

# # data path for linux system
# project_dir = '/bmrp092temp/Prostate_Cancer_ex_vivo/Deep_Learning/'
# result_dir = '/bmrp092temp/2019_Nature_Medicine/deep_learning/PCa_BPH_exvivo/result/'
# log_dir = '/bmrp092temp/2019_Nature_Medicine/deep_learning/PCa_BPH_exvivo/log/'


if not os.path.exists(log_dir):
    print('log directory does not exist - creating...')
    os.makedirs(log_dir)
    os.makedirs(log_dir + '/train')
    os.makedirs(log_dir + '/validation')
    print('log directory created!!!')
else:
    print('log directory already exists...')

if not os.path.exists(result_dir):
    print('result directory does not exist - creating...')
    os.makedirs(result_dir)
    print('result directory created!!!')
else:
    print('result directory already exists...')

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

# load data from csv files and define x and y
df = pd.read_csv(os.path.join(project_dir, 'PCa.csv'))
# df = df[~df['Sub_ID'].str.contains("SH")]

df.loc[df['ROI_Class'] == 'BPH', 'y_cat'] = 0
df.loc[df['ROI_Class'] == 'PCa', 'y_cat'] = 1

class0 = df[df['y_cat'] == 0]
class0_sample = class0.sample(int(class0.shape[0]))

class1 = df[df['y_cat'] == 1]
class1_sample = class1.sample(int(class1.shape[0]))

# class1 = df[df['y_cat'] == 1]
# class1_sample = class1.sample(int(class1.shape[0]*0.8))
#
# class1 = df[df['y_cat'] == 2]
# class1_sample = class1.sample(int(class1.shape[0]*0.849))

df_2 = pd.concat([class0_sample, class1_sample])
# df_2 = pd.concat([class0_sample, class1_sample, class2_sample])

X = df_2.iloc[:, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 21, 23, 24, 25, 26, 27, 28, 29]]

Y = df_2.y_cat.astype('int')

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
print("train set size:", len(x_train))
print("validation set size:", len(x_val))
print("test set size:", len(x_test))
print("loading data from csv file: complete!!!")
print("deep neuronetwork construction: start...")

##def reset_tf_graph(seed=seed):
##    tf.reset_default_graph()
##    tf.set_random_seed(seed)
##    np.random.seed(seed)
##
##reset_tf_graph()

with tf.compat.v1.name_scope("input"):
    x = tf.compat.v1.placeholder(tf.float32, shape=(None, n_inputs), name="x")
    y = tf.compat.v1.placeholder(tf.int64, shape=(None), name="y")
    training = tf.compat.v1.placeholder_with_default(False, shape=(), name="train")

with tf.compat.v1.name_scope("dnn"):
    he_init = tf.contrib.layers.variance_scaling_initializer()

    dense_layer = partial(
                          tf.compat.v1.layers.dense,
                          kernel_initializer=he_init
    )

    batch_normalization_layer = partial(
                                        tf.compat.v1.layers.batch_normalization,
                                        training=training,
                                        momentum=batch_momentum
    )

    hidden1 = dense_layer(x, n_hidden1, name="hidden1")
    bn1 = batch_normalization_layer(hidden1, name="bn1")
    tf.compat.v1.summary.histogram("batch_normalization", bn1)
    bn1_act = tf.nn.elu(bn1, name="elu_bn1")
    # bn1_act = tf.nn.leaky_relu(bn1, alpha=0.2, name="leaky_relu_bn1")
    hidden1_drop = tf.compat.v1.layers.dropout(bn1_act, dropout_rate, training=training)
    tf.compat.v1.summary.histogram("activations", bn1_act)

    hidden2 = dense_layer(hidden1_drop, n_hidden2, name="hidden2")
    bn2 = batch_normalization_layer(hidden2, name="bn2")
    tf.compat.v1.summary.histogram("batch_normalization", bn2)
    bn2_act = tf.nn.elu(bn2, name="elu_bn2")
    # bn2_act = tf.nn.leaky_relu(bn2, alpha=0.2, name="leaky_relu_bn2")
    hidden2_drop = tf.compat.v1.layers.dropout(bn2_act, dropout_rate, training=training)
    tf.compat.v1.summary.histogram("activations", bn2_act)

    hidden3 = dense_layer(bn2_act, n_hidden3, name="hidden3")
    bn3 = batch_normalization_layer(hidden3, name="bn3")
    tf.compat.v1.summary.histogram("batch_normalization", bn3)
    bn3_act = tf.nn.elu(bn3, name="elu_bn3")
    # bn3_act = tf.nn.leaky_relu(bn3, alpha=0.2, name="leaky_relu_bn3")
    hidden3_drop = tf.compat.v1.layers.dropout(bn3_act, dropout_rate, training=training)
    tf.compat.v1.summary.histogram("activations", bn3_act)

    hidden4 = dense_layer(hidden3_drop, n_hidden4, name="hidden4")
    bn4 = batch_normalization_layer(hidden4, name="bn4")
    tf.compat.v1.summary.histogram("batch_normalization", bn4)
    bn4_act = tf.nn.elu(bn4, name="elu_bn4")
    # bn4_act = tf.nn.leaky_relu(bn4, alpha=0.2, name="leaky_relu_bn4")
    hidden4_drop = tf.compat.v1.layers.dropout(bn4_act, dropout_rate, training=training)
    tf.compat.v1.summary.histogram("activations", bn4_act)

    hidden5 = dense_layer(bn4_act, n_hidden5, name="hidden5")
    bn5 = batch_normalization_layer(hidden5, name="bn5")
    tf.compat.v1.summary.histogram("batch_normalization", bn5)
    bn5_act = tf.nn.elu(bn5, name="elu_bn5")
    # bn5_act = tf.nn.leaky_relu(bn5, alpha=0.2, name="leaky_relu_bn5")
    hidden5_drop = tf.compat.v1.layers.dropout(bn5_act, dropout_rate, training=training)
    tf.compat.v1.summary.histogram("activations", bn5_act)

    hidden6 = dense_layer(hidden5_drop, n_hidden6, name="hidden6")
    bn6 = batch_normalization_layer(hidden6, name="bn6")
    bn6_act = tf.nn.elu(bn5, name="elu_bn6")
    hidden6_drop = tf.compat.v1.layers.dropout(bn6_act, dropout_rate, training=training)

    hidden7 = dense_layer(bn6_act, n_hidden7, name="hidden7")
    bn7 = batch_normalization_layer(hidden7, name="bn7")
    bn7_act = tf.nn.elu(bn7, name="elu_bn7")
    hidden7_drop = tf.compat.v1.layers.dropout(bn7_act, dropout_rate, training=training)


    hidden8 = dense_layer(hidden7_drop, n_hidden8, name="hidden8")
    bn8 = batch_normalization_layer(hidden8, name="bn8")
    bn8_act = tf.nn.elu(bn8, name="elu_bn8")
    hidden8_drop = tf.compat.v1.layers.dropout(bn8_act, dropout_rate, training=training)


    hidden9 = dense_layer(bn8_act, n_hidden9, name="hidden9")
    bn9 = batch_normalization_layer(hidden9, name="bn9")
    bn9_act = tf.nn.elu(bn9, name="elu_bn9")
    hidden9_drop = tf.compat.v1.layers.dropout(bn9_act, dropout_rate, training=training)


    hidden10 = dense_layer(hidden9_drop, n_hidden10, name="hidden10")
    bn10 = batch_normalization_layer(hidden10, name="bn10")
    bn10_act = tf.nn.elu(bn10, name="elu_bn10")
    hidden10_drop = tf.compat.v1.layers.dropout(bn10_act, dropout_rate, training=training)


    logits_before_bn = dense_layer(bn10_act, n_outputs, name="outputs")
    logits = batch_normalization_layer(logits_before_bn, name="bn11")

with tf.compat.v1.name_scope("cross_entropy"):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                                   logits=logits)

    with tf.compat.v1.name_scope("total"):
        loss = tf.reduce_mean(input_tensor=cross_entropy, name="loss")
tf.compat.v1.summary.scalar('cross_entropy', loss)

with tf.compat.v1.name_scope("train"):
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate,
    #                                       momentum=momentum,
    #                                       decay=0.9,
    #                                       epsilon=1e-10)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
    #                                        momentum=momentum,
    #                                        use_nesterov=True)

    extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss)

with tf.compat.v1.name_scope("evaluation"):
    with tf.compat.v1.name_scope("correct_prediction"):
        correct_prediction = tf.nn.in_top_k(predictions=logits, targets=y, k=1)
    with tf.compat.v1.name_scope('accuracy'):
        accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))
tf.compat.v1.summary.scalar('accuracy', accuracy)

for op in (x, y, accuracy, train_op):
    tf.compat.v1.add_to_collection("important_ops", op)

print("deep neural network construction: complete!!!")

with tf.compat.v1.Session() as sess:
    init = tf.compat.v1.global_variables_initializer()
    init.run()

    merged_summary = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter(log_dir + '/train', sess.graph)
    validation_writer = tf.compat.v1.summary.FileWriter(log_dir + '/validation', sess.graph)
    saver = tf.compat.v1.train.Saver()

    print('\nTraining phase initiated.')
    for epoch in range(n_epochs):
        n_batches = train_size // batch_size
        for iteration in range(n_batches):
            randidx = np.random.randint(int(train_size), size=batch_size)
            x_batch = x_train.iloc[randidx, :]
            y_batch = y_train.iloc[randidx]

            sess.run(
                     [train_op, extra_update_ops],
                     feed_dict={training: True,
                                       x: x_batch,
                                       y: y_batch}
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
            run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            run_metadata = tf.compat.v1.RunMetadata()
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

    saver.save(sess, os.path.join(log_dir, 'deep_learning.ckpt'))
    train_writer.close()
    validation_writer.close()

    # Evaluating final accuracy of the model
    accuracy_train = sess.run(accuracy, feed_dict={x: x_train,
                                                   y: y_train})

    accuracy_val = sess.run(accuracy, feed_dict={x: x_val,
                                                 y: y_val})

    accuracy_test = sess.run(accuracy, feed_dict={x: x_test,
                                                  y: y_test})

    print('\nfinal train accuracy of the model:', np.around(accuracy_train, d))
    print('final validation accuracy of the model:', np.around(accuracy_val, d))
    print('final test accuracy of the model:', np.around(accuracy_test, d))

    # calculate confusion matrix and make the plots
    y_pred = tf.argmax(input=logits.eval(feed_dict={x: x_val}), axis=1)
    y_prediction = sess.run(y_pred)

    con_mat = tf.math.confusion_matrix(
                                  labels=y_val,
                                  predictions=y_pred,
                                  num_classes=n_outputs,
                                  dtype=tf.int32,
                                  name="confusion_matrix"
    )

    cm_1 = sess.run(con_mat)
    cm_2 = np.around(cm_1.astype('float')/cm_1.sum(axis=1)[:, np.newaxis], d)
    print("\nconfusion matrix: print...")
    print(cm_1)
    print("\nnormalized confusion matrix: print...")
    print(cm_2)
    print("\nprecision, recall, f1-score: print...")
    print(classification_report(y_val, y_prediction, digits=d))

    # plot confusion matrix
    ax_2 = sn.heatmap(cm_2, annot=True, annot_kws={"size": 16}, cmap="Blues", linewidths=.5)
    ax_2.axhline(y=0, color='k', linewidth=3)
    ax_2.axhline(y=2, color='k', linewidth=3)
    ax_2.axvline(x=0, color='k', linewidth=3)
    ax_2.axvline(x=2, color='k', linewidth=3)
    ax_2.set_aspect('equal')
    plt.savefig(os.path.join(result_dir, 'confusion_matrix.png'), format='png', dpi=600)
    plt.tight_layout()
    plt.show()
    print("plotting confusion matrix_2: complete!")

    # Compute ROC curve and AUC for each class
    # fpr = dict()
    # tpr = dict()
    # threshold = dict()
    # roc_auc = dict()
    # for i in range(n_outputs):
    #     fpr[i], tpr[i], threshold[i] = roc_curve(y_val[:, i], y_prediction[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     print(tpr[i])
    #     print(fpr[i])
    #     print(roc_auc[i])
    #     ax = fig.add_subplot(5, 1, i)
    #     print()
    #     plt.title('Receiver Operating Characteristic')
    #     plt.plot(fpr[i], tpr[i], 'b', label='AUC = %0.2f' % roc_auc[i])
    #     plt.legend(loc='lower right')
    #     plt.plot([0, 1], [0, 1], 'r--')
    #     plt.xlim([0, 1])
    #     plt.ylim([0, 1])
    #     plt.ylabel('True Positive Rate')
    #     plt.xlabel('False Positive Rate')
    #     plt.show()

    save_path = saver.save(sess, log_dir)

sess.close()
# file_writer.close()
print('session close!')
print("Run the command line:\n"\
      "--> tensorboard --logdir="\
      "\nThen open http://localhost:6006/ into your web browser")
print("deep neural network classification: complete!")



# ##########################################
#   print('Evaluating ROC curve')
#   #predictions = []
#   labels = mnist.test.labels
#   threshold_num = 100
#   thresholds = []
#   fpr_list = [] #false positive rate
#   tpr_list = [] #true positive rate
#   summt = tf.Summary()
#   pred = tf.nn.softmax(y_conv)
#   predictions = sess.run(pred,
#                          feed_dict={x: mnist.test.images,
#                                           y_: mnist.test.labels,
#                                           keep_prob: 1.0})

#   for i in range(len(labels)):
#       threshold_step = 1. / threshold_num
#       for t in range(threshold_num+1):
#           th = 1 - threshold_num * t
#           fp = 0
#           tp = 0
#           tn = 0
#           for j in range(len(labels)):
#               for k in range(10):
#                   if not labels[j][k]:
#                       if predictions[j][k] >= t:
#                           fp += 1
#                       else:
#                           tn += 1
#                   elif predictions[j][k].any() >= t:
#                       tp += 1
#           fpr = fp / float(fp + tn)
#           tpr = tp / float(len(labels))
#           fpr_list.append(fpr)
#           tpr_list.append(tpr)
#           thresholds.append(th)
#   #auc = tf.metrics.auc(labels, predictions, thresholds)
#   summt.value.add(tag = 'ROC', simple_value = tpr)
#   roc_writer.add_summary(summt, fpr * 100)
#   roc_writer.flush()
#
#
# # tensorboard projector
# metadata = os.path.join(log_dir, 'metadata.tsv')
#
# mnist = input_data.read_data_sets('MNIST_data')
# images = tf.Variable(mnist.test.images, name='images')
#
# with open(metadata, 'w') as metadata_file:
#     for row in mnist.test.labels:
#         metadata_file.write('%d\n' % row)
#
# with tf.Session() as sess:
#     saver = tf.train.Saver([images])
#
#     sess.run(images.initializer)
#     saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))
#
#     config = projector.ProjectorConfig()
#     # One can add multiple embeddings.
#     embedding = config.embeddings.add()
#     embedding.tensor_name = images.name
#     # Link this tensor to its metadata file (e.g. labels).
#     embedding.metadata_path = metadata
#     # Saves a config file that TensorBoard will read during startup.
#     projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)













# Calculate batch normalization for the inputs of the second hidden layer
    # The BN algorithm uses 'exponential decay' to compute the running averages.
    # 'momentum' parameter is used for this purpose.
    # Given a value v then the running average v.hat is upated as follow:
    #   v.hat <- v.hat * momentum + v * (1 - momentum)
    # Normally, momentum value is greater or equal to 0.9 (very close to 1)
    # Larger datasets, smaller mini-batches -> bigger value of momentum




