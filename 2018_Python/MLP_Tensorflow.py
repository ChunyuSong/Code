import pandas as pd
import glob2 as glob
from sklearn.svm import SVC
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sn
import string
from matplotlib.pylab import *
import os
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from functools import partial


print("Deep Neural Network for PCa grade classification: start...")

project_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification'
results_dir = r'\\10.39.42.102\temp\Prostate_Cancer_Project_Shanghai\PCa_Machine_Learning\PCa_Benign_Classification'

# load data from excel files
print("loading data: start...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("data loading: complete!")
print("training set size:", len(X_train))
print("test set size:", len(X_test))

# training a SVM classifier
print("training a MLP classifier: start...")

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10
X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
training = tf.placeholder_with_default(False, shape=(), name='training')
hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = tf.layers.batch_normalization(hidden1, training=training, momentum=0.9)
bn1_act = tf.nn.elu(bn1)
hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
bn2 = tf.layers.batch_normalization(hidden2, training=training, momentum=0.9)
bn2_act = tf.nn.elu(bn2)
logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = tf.layers.batch_normalization(logits_before_bn)


my_batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = my_batch_norm_layer(hidden1)
bn1_act = tf.nn.elu(bn1)
hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
bn2 = my_batch_norm_layer(hidden2)
bn2_act = tf.nn.elu(bn2)
logits_before_bn = tf.layers.dense(bn2_act, n_outputs, name="outputs")
logits = my_batch_norm_layer(logits_before_bn)

my_dense_layer = partial(tf.layers.dense,
                         activation=tf.nn.relu,
                         kernel_regularizer=tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope("dnn"):
     hidden1 = my_dense_layer(X, n_hidden1, name="hidden1")
     hidden2 = my_dense_layer(hidden1, n_hidden2, name="hidden2")
     logits = my_dense_layer(hidden2, n_outputs, activation=None, name="outputs")


training = tf.placeholder_with_default(False, shape=(), name='training')
dropout_rate = 0.5 # == 1 - keep_prob
X_drop = tf.layers.dropout(X, dropout_rate, training=training)
with tf.name_scope("dnn"):
     hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu, name="hidden1")
     hidden1_drop = tf.layers.dropout(hidden1, dropout_rate, training=training)
     hidden2 = tf.layers.dense(hidden1_drop, n_hidden2, activation=tf.nn.relu, name="hidden2")
     hidden2_drop = tf.layers.dropout(hidden2, dropout_rate, training=training)
     logits = tf.layers.dense(hidden2_drop, n_outputs, name="outputs")

from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
           'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
          'b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'b2': tf.Variable(tf.random_normal([n_hidden_2])),
          'out': tf.Variable(tf.random_normal([n_classes]))
}

# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    """ Neural Network.
    A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
    implementation with TensorFlow. This example is using the MNIST database
    of handwritten digits (http://yann.lecun.com/exdb/mnist/).
    This example is using TensorFlow layers, see 'neural_network_raw' example for
    a raw implementation with variables.
    Links:
        [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
    Author: Aymeric Damien
    Project: https://github.com/aymericdamien/TensorFlow-Examples/
    """

    from __future__ import print_function

    # Import MNIST data
    from tensorflow.examples.tutorials.mnist import input_data

    mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

    import tensorflow as tf

    # Parameters
    learning_rate = 0.1
    num_steps = 1000
    batch_size = 128
    display_step = 100

    # Network Parameters
    n_hidden_1 = 256  # 1st layer number of neurons
    n_hidden_2 = 256  # 2nd layer number of neurons
    num_input = 784  # MNIST data input (img shape: 28*28)
    num_classes = 10  # MNIST total classes (0-9 digits)


    # Define the neural network
    def neural_net(x_dict):
        # TF Estimator input is a dict, in case of multiple inputs
        x = x_dict['images']
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.layers.dense(x, n_hidden_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.layers.dense(layer_1, n_hidden_2)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.layers.dense(layer_2, num_classes)
        return out_layer


    # Define the model function (following TF Estimator Template)
    def model_fn(features, labels, mode):
        # Build the neural network
        logits = neural_net(features)

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
        pred_probas = tf.nn.softmax(logits)

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)

        # Define loss and optimizer
        loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss_op,
            train_op=train_op,
            eval_metric_ops={'accuracy': acc_op})

        return estim_specs


    # Build the Estimator
    model = tf.estimator.Estimator(model_fn)

    # Define the input function for training
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.train.images}, y=mnist.train.labels,
        batch_size=batch_size, num_epochs=None, shuffle=True)
    # Train the Model
    model.train(input_fn, steps=num_steps)

    # Evaluate the Model
    # Define the input function for evaluating
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': mnist.test.images}, y=mnist.test.labels,
        batch_size=batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = model.evaluate(input_fn)

    print("Testing Accuracy:", e['accuracy'])

    import tensorflow as tf
    import pandas as pd
    from sklearn.cross_validation import train_test_split

    FILE_PATH = '~/Desktop/bank-add/bank_equalized.csv'  # Path to .csv dataset
    raw_data = pd.read_csv(FILE_PATH)  # Open raw .csv

    print("Raw data loaded successfully...\n")
    # ------------------------------------------------------------------------------
    # Variables

    Y_LABEL = 'y'  # Name of the variable to be predicted
    KEYS = [i for i in raw_data.keys().tolist() if i != Y_LABEL]  # Name of predictors
    N_INSTANCES = raw_data.shape[0]  # Number of instances
    N_INPUT = raw_data.shape[1] - 1  # Input size
    N_CLASSES = raw_data[Y_LABEL].unique().shape[0]  # Number of classes (output size)
    TEST_SIZE = 0.1  # Test set size (% of dataset)
    TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))  # Train size
    LEARNING_RATE = 0.001  # Learning rate
    TRAINING_EPOCHS = 400  # Number of epochs
    BATCH_SIZE = 100  # Batch size
    DISPLAY_STEP = 20  # Display progress each x epochs
    HIDDEN_SIZE = 200  # Number of hidden neurons 256
    ACTIVATION_FUNCTION_OUT = tf.nn.tanh  # Last layer act fct
    STDDEV = 0.1  # Standard deviation (for weights random init)
    RANDOM_STATE = 100  # Random state for train_test_split

    print("Variables loaded successfully...\n")
    print("Number of predictors \t%s" % (N_INPUT))
    print("Number of classes \t%s" % (N_CLASSES))
    print("Number of instances \t%s" % (N_INSTANCES))
    print("\n")
    print("Metrics displayed:\tPrecision\n")
    # ------------------------------------------------------------------------------
    # Loading data

    # Load data
    data = raw_data[KEYS].get_values()  # X data
    labels = raw_data[Y_LABEL].get_values()  # y data

    # One hot encoding for labels
    labels_ = np.zeros((N_INSTANCES, N_CLASSES))
    labels_[np.arange(N_INSTANCES), labels] = 1

    # Train-test split
    data_train, data_test, labels_train, labels_test = train_test_split(data,
                                                                        labels_,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=RANDOM_STATE)

    print("Data loaded and splitted successfully...\n")
    # ------------------------------------------------------------------------------
    # Neural net construction

    # Net params
    n_input = N_INPUT  # input n labels
    n_hidden_1 = HIDDEN_SIZE  # 1st layer
    n_hidden_2 = HIDDEN_SIZE  # 2nd layer
    n_hidden_3 = HIDDEN_SIZE  # 3rd layer
    n_hidden_4 = HIDDEN_SIZE  # 4th layer
    n_classes = N_CLASSES  # output m classes

    # Tf placeholders
    X = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    dropout_keep_prob = tf.placeholder(tf.float32)


    def mlp(_X, _weights, _biases, dropout_keep_prob):
        layer1 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])), dropout_keep_prob)
        layer2 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer1, _weights['h2']), _biases['b2'])), dropout_keep_prob)
        layer3 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer2, _weights['h3']), _biases['b3'])), dropout_keep_prob)
        layer4 = tf.nn.dropout(tf.nn.tanh(tf.add(tf.matmul(layer3, _weights['h4']), _biases['b4'])), dropout_keep_prob)
        out = ACTIVATION_FUNCTION_OUT(tf.add(tf.matmul(layer4, _weights['out']), _biases['out']))
        return out


    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], stddev=STDDEV)),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=STDDEV)),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], stddev=STDDEV)),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], stddev=STDDEV)),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], stddev=STDDEV)),
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Build model
    pred = mlp(X, weights, biases, dropout_keep_prob)

    # Loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))  # softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Net built successfully...\n")
    print("Starting training...\n")
    # ------------------------------------------------------------------------------
    # Training

    # Initialize variables
    init_all = tf.initialize_all_variables()

    # Launch session
    sess = tf.Session()
    sess.run(init_all)

    # Training loop
    for epoch in range(TRAINING_EPOCHS):
        avg_cost = 0.
        total_batch = int(data_train.shape[0] / BATCH_SIZE)
        # Loop over all batches
        for i in range(total_batch):
            randidx = np.random.randint(int(TRAIN_SIZE), size=BATCH_SIZE)
            batch_xs = data_train[randidx, :]
            batch_ys = labels_train[randidx, :]
            # Fit using batched data
            sess.run(optimizer, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 0.9})
            # Calculate average cost
            avg_cost += sess.run(cost, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 1.}) / total_batch
        # Display progress
        if epoch % DISPLAY_STEP == 0:
            print("Epoch: %03d/%03d cost: %.9f" % (epoch, TRAINING_EPOCHS, avg_cost))
            train_acc = sess.run(accuracy, feed_dict={X: batch_xs, y: batch_ys, dropout_keep_prob: 1.})
            print("Training accuracy: %.3f" % (train_acc))

    print("End of training.\n")
    print("Testing...\n")
    # ------------------------------------------------------------------------------
    # Testing

    test_acc = sess.run(accuracy, feed_dict={X: data_test, y: labels_test, dropout_keep_prob: 1.})
    print("Test accuracy: %.3f" % (test_acc))

    sess.close()
    print("Session closed!")