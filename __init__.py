from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
import time

execute_start_time = time.time()

x_training_data = np.loadtxt("binary_training_input_data.txt", dtype=float, delimiter=" ")
y_training_data = np.loadtxt("binary_training_output_data.txt", dtype=float, delimiter=" ")
tau_array = np.array([0.0])


# Network Parameters
input_node_amount = x_training_data.shape[1]
hidden_node_amount = 1
output_node_amount = 1
learning_rate_eta = 0.01

# Parameters
data_size = x_training_data.shape[0]
big_number = 25

# placeholders
x_placeholder = tf.placeholder(tf.float64)
y_placeholder = tf.placeholder(tf.float64)
tau_placeholder = tf.placeholder(tf.float64)

# network architecture
hidden_weights = tf.Variable(tf.random_normal([input_node_amount, hidden_node_amount], dtype=tf.float64))
hidden_thresholds = tf.Variable(tf.random_normal([hidden_node_amount], dtype=tf.float64))
tau_in_each_hidden_node = np.tile(tau_array, (hidden_node_amount, 1))
output_weights = tf.Variable(tf.random_normal([hidden_node_amount, output_node_amount], dtype=tf.float64))
output_threshold = tf.Variable(tf.random_normal([output_node_amount], dtype=tf.float64))

hidden_layer_before_tanh = tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds)
# exp = tf.exp(-1 * tf.matmul(hidden_layer_before_tanh, tf.pow(2.0, tau_placeholder)))
# hidden_layer = tf.divide(1.0 - exp, 1.0 + exp)
hidden_layer = tf.tanh(tf.multiply(hidden_layer_before_tanh, tf.pow(tf.constant(2.0, dtype=tf.float64), tau_placeholder)))
output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

# learning goal & optimizer
average_squared_residual = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_squared_residual)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

previous_alpha = tf.double.max
previous_beta = tf.double.min

for k in range(1, data_size+1):
    print('-----stage: '+str(k)+'-----')
    # take first k training case
    current_stage_x_training_data = x_training_data[:k]
    current_stage_y_training_data = y_training_data[:k]

    # calculate alpha & beta in condition L
    predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data, tau_placeholder: tau_in_each_hidden_node})
    alpha = tf.double.max
    beta = tf.double.min
    # for node in tf.get_default_graph().as_graph_def().node:
    #     print(node.name)
    print(predict_y)
    it = np.nditer(predict_y, flags=['f_index'])
    while not it.finished:
        if current_stage_y_training_data[it.index] == 1:
            if it[0] < alpha:
                alpha = it[0]
        if current_stage_y_training_data[it.index] == -1:
            if it[0] > beta:
                beta = it[0]
        it.iternext()
    print('alpha= '+str(alpha))
    print('beta= '+str(beta))
    if k == 2:
        previous_alpha = alpha
        previous_beta = beta

    if alpha > beta:
        print('new training case is familiar to us, no further learning effort involved.')
    else:
        # cram it first
        print('start cramming')
        # calculate relevant parameters
        hidden_node_amount += 1
        current_hidden_weights, current_hidden_thresholds, current_output_weights, current_output_threshold, = sess.run([hidden_weights, hidden_thresholds, output_weights, output_threshold], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data, tau_placeholder: tau_in_each_hidden_node})
        print('current hidden weights:')
        print(current_hidden_weights)
        print('current hidden thresholds:')
        print(current_hidden_thresholds)
        print('current output weights:')
        print(current_output_weights)
        print('current output threshold:')
        print(current_output_threshold)
        new_hidden_node_neuron_weights = current_stage_x_training_data[k-1]
        print('new hidden weights:')
        print(new_hidden_node_neuron_weights)
        new_hidden_node_threshold = 1 - input_node_amount
        print('new hidden thresholds:')
        print(new_hidden_node_threshold)
        if current_stage_y_training_data[k-1] == 1:
            new_output_node_neuron_weight = previous_beta - predict_y[0][k-1]
        if current_stage_y_training_data[k-1] == -1:
            new_output_node_neuron_weight = predict_y[0][k-1] - previous_alpha
        print('predict value of most recent training case: '+str(predict_y[0][k-1]))
        print('previous alpha: '+str(previous_alpha))
        print('previous beta: '+str(previous_beta))
        print('new output weight: '+str(new_output_node_neuron_weight))

        # combine weights & thresholds
        new_hidden_weights = np.append(current_hidden_weights, new_hidden_node_neuron_weights.reshape(input_node_amount, 1), axis=1)
        new_hidden_thresholds = np.append(current_hidden_thresholds, new_hidden_node_threshold)
        new_output_weights = np.append(current_output_weights, new_output_node_neuron_weight).reshape(hidden_node_amount,1)
        tau_array = np.append(tau_array, big_number)
        # print(new_hidden_weights)
        # print(new_hidden_thresholds)
        # print(new_output_weights)
        # print(tau_array)

        # rebuild neuron network architecture
        hidden_weights = tf.Variable(new_hidden_weights, dtype=tf.float64)
        hidden_thresholds = tf.Variable(new_hidden_thresholds, dtype=tf.float64)
        # tau_in_each_hidden_node = np.tile(tau_array, (hidden_node_amount, 1))
        tau_in_each_hidden_node = tau_array
        output_weights = tf.Variable(new_output_weights, dtype=tf.float64)
        output_threshold = tf.Variable(current_output_threshold, dtype=tf.float64)

        # update weights & threshold in sess
        # update_operation = tf.variables_initializer([hidden_weights, hidden_threshold, output_weights])
        # sess.run(update_operation)

        hidden_layer_before_tanh = tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds)
        hidden_layer = tf.tanh(tf.multiply(hidden_layer_before_tanh, tf.pow(tf.constant(2.0, dtype=tf.float64), tau_placeholder)))
        output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

        average_squared_residual = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
        train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_squared_residual)

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        previous_alpha = alpha
        previous_beta = beta
        # sess.run(train, feed_dict={x_placeholder: x_training_data, y_placeholder: y_training_data})

    # have some trouble in calculate exp, skip softening process

    # pruning
    # if hidden_node_amount > 1:



