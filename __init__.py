import tensorflow as tf
import numpy as np
import time

execute_start_time = time.time()

x_training_data = np.loadtxt("binary_training_input_data.txt", dtype=float, delimiter=" ")
y_training_data = np.loadtxt("binary_training_output_data.txt", dtype=float, delimiter=" ")
tau_in_each_hidden_node = np.array([1.0])

# Network Parameters
input_node_amount = x_training_data.shape[1]
hidden_node_amount = 1
output_node_amount = 1
learning_rate_eta = 0.01

# Parameters
data_size = x_training_data.shape[0]
big_number = 15

# placeholders
x_placeholder = tf.placeholder(tf.float64)
y_placeholder = tf.placeholder(tf.float64)
tau_placeholder = tf.placeholder(tf.float64)

# network architecture
output_threshold = tf.Variable(tf.zeros([output_node_amount], dtype=tf.float64))
output_weights = tf.Variable(tf.ones([hidden_node_amount, output_node_amount], dtype=tf.float64))
tau_in_each_hidden_node = np.tile(tau_in_each_hidden_node, (hidden_node_amount, 1))
hidden_thresholds = tf.Variable(tf.zeros([hidden_node_amount], dtype=tf.float64))
hidden_weights = tf.Variable(tf.ones([input_node_amount, hidden_node_amount], dtype=tf.float64))

hidden_layer_before_tanh = tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds)
hidden_layer = tf.tanh(
    tf.multiply(hidden_layer_before_tanh, tf.pow(tf.constant(2.0, dtype=tf.float64), tau_placeholder)))
output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

# learning goal & optimizer
average_squared_residual = tf.reduce_mean(tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_squared_residual)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for k in range(1, data_size + 1):
    print('-----stage: ' + str(k) + '-----')
    # take first k training case
    current_stage_x_training_data = x_training_data[:k]
    current_stage_y_training_data = y_training_data[:k]

    # calculate alpha & beta in condition L
    predict_y = sess.run([output_layer],
                         {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data,
                          tau_placeholder: tau_in_each_hidden_node})
    alpha = tf.double.max
    beta = tf.double.min
    # for node in tf.get_default_graph().as_graph_def().node:
    #     print(node.name)
    print(predict_y)
    it = np.nditer(predict_y, flags=['f_index'])
    i = 1
    while not it.finished:
        if current_stage_y_training_data[it.index] == 1:
            if it[0] < alpha:
                alpha = it[0]
                if not i == k:
                    min_predict_value_in_class_one_of_previous_stage_training_case = alpha
        if current_stage_y_training_data[it.index] == -1:
            if it[0] > beta:
                beta = it[0]
                if not i == k:
                    max_predict_value_in_class_two_of_previous_stage_training_case = beta
        i += 1
        it.iternext()
    print('alpha= ' + str(alpha))
    print('beta= ' + str(beta))

    if alpha > beta:
        print('new training case is familiar to us, no further learning effort involved.')
    else:
        # cram it first
        print('start cramming')
        # calculate relevant parameters
        hidden_node_amount += 1
        current_hidden_weights, current_hidden_thresholds, current_output_weights, current_output_threshold, = sess.run(
            [hidden_weights, hidden_thresholds, output_weights, output_threshold],
            {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data,
             tau_placeholder: tau_in_each_hidden_node})
        print('current hidden weights:')
        print(current_hidden_weights)
        print('current hidden thresholds:')
        print(current_hidden_thresholds)
        print('current output weights:')
        print(current_output_weights)
        print('current output threshold:')
        print(current_output_threshold)
        new_hidden_node_neuron_weights = current_stage_x_training_data[k - 1]
        print('new hidden weights:')
        print(new_hidden_node_neuron_weights)
        new_hidden_node_threshold = 1 - input_node_amount
        print('new hidden thresholds:')
        print(new_hidden_node_threshold)
        if current_stage_y_training_data[k - 1] == 1:
            new_output_node_neuron_weight = predict_y[0][
                                                k - 1] - max_predict_value_in_class_two_of_previous_stage_training_case
        if current_stage_y_training_data[k - 1] == -1:
            new_output_node_neuron_weight = min_predict_value_in_class_one_of_previous_stage_training_case - \
                                            predict_y[0][k - 1]
        print('predict value of most recent training case: ' + str(predict_y[0][k - 1]))
        print('new output weight:')
        print(new_output_node_neuron_weight)

        # combine weights & thresholds
        new_hidden_weights = np.append(current_hidden_weights,
                                       new_hidden_node_neuron_weights.reshape(input_node_amount, 1), axis=1)
        new_hidden_thresholds = np.append(current_hidden_thresholds, new_hidden_node_threshold)
        new_output_weights = np.append(current_output_weights, new_output_node_neuron_weight).reshape(
            hidden_node_amount, 1)
        tau_in_each_hidden_node = np.append(tau_in_each_hidden_node, big_number)

        # create new graph & session
        with tf.Graph().as_default():  # Create a new graph, and make it the default.
            # placeholders
            x_placeholder = tf.placeholder(tf.float64)
            y_placeholder = tf.placeholder(tf.float64)
            tau_placeholder = tf.placeholder(tf.float64)

            # network architecture
            output_threshold = tf.Variable(current_output_threshold)
            output_weights = tf.Variable(new_output_weights)
            hidden_thresholds = tf.Variable(new_hidden_thresholds)
            hidden_weights = tf.Variable(new_hidden_weights)

            hidden_layer_before_tanh = tf.add(tf.matmul(x_placeholder, hidden_weights), hidden_thresholds)
            hidden_layer = tf.tanh(
                tf.multiply(hidden_layer_before_tanh, tf.pow(tf.constant(2.0, dtype=tf.float64), tau_placeholder)))
            output_layer = tf.add(tf.matmul(hidden_layer, output_weights), output_threshold)

            # learning goal & optimizer
            average_squared_residual = tf.reduce_mean(
                tf.reduce_sum(tf.square(y_placeholder - output_layer), reduction_indices=[1]))
            train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(average_squared_residual)

            saver = tf.train.Saver()

            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            # softening
            # save variables
            saver.save(sess, r"C:\Users\Lee Chia Lun\PycharmProjects\autoencoder\softening_learning\model.ckpt")

            # change tau value of newest hidden node
            newest_hidden_node_tau_value = tau_in_each_hidden_node[hidden_node_amount-1]
            print(newest_hidden_node_tau_value)
            while newest_hidden_node_tau_value > 1:
                newest_hidden_node_tau_value -= 1
                tau_in_each_hidden_node[hidden_node_amount-1] = newest_hidden_node_tau_value
                print('tau array:')
                print(tau_in_each_hidden_node)
                softening_success = False

                for i in range(1000):
                    # forward pass
                    predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data,
                                                          y_placeholder: current_stage_y_training_data,
                                                          tau_placeholder: tau_in_each_hidden_node})
                    # print(predict_y)

                    # check condition L
                    alpha = tf.double.max
                    beta = tf.double.min
                    it = np.nditer(predict_y, flags=['f_index'])
                    while not it.finished:
                        if current_stage_y_training_data[it.index] == 1:
                            if it[0] < alpha:
                                alpha = it[0]
                        if current_stage_y_training_data[it.index] == -1:
                            if it[0] > beta:
                                beta = it[0]
                        it.iternext()

                    # print(alpha)
                    # print(beta)
                    if alpha > beta:
                        print('softening success, gradient descent trained {0} times, #{1} tau value decrease by 1, current tau value: {2}'.format(i, hidden_node_amount, newest_hidden_node_tau_value))
                        softening_success = True
                        saver.save(sess, r"C:\Users\Lee Chia Lun\PycharmProjects\autoencoder\softening_learning\model.ckpt")
                        break
                    else:
                        sess.run(train, feed_dict={x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data, tau_placeholder: tau_in_each_hidden_node})

                if not softening_success:
                    print('softening failed, gradient descent trained {0} times, restore #{1} tau value , restore tau value: {2}'.format(
                            i, hidden_node_amount, newest_hidden_node_tau_value+1))
                    tau_in_each_hidden_node[hidden_node_amount - 1] = newest_hidden_node_tau_value+1
                    saver.restore(sess, r"C:\Users\Lee Chia Lun\PycharmProjects\autoencoder\softening_learning\model.ckpt")
                    break

            # PRUNING
            """
            嚴重問題: session.run 會執行整個定義好的 graph,
            意思是如果同個session, 每次欲計算不同的部份數值而執行 session.run()
            預設graph內定義的所有 tensor 都會被再執行, 包括 optimizer, 可能進而影響定義的 tf.Variable
            可能解決方案: 將 Graph 與 Session 分開定義
            Reference: https://danijar.com/what-is-a-tensorflow-session/
            """
            if hidden_node_amount > 1:  # equals to the number of hidden nodes
                # then try pruning from the begining hidden node
                for remove_index in range(hidden_node_amount):
                    exam_hidden_weights = tf.concat(
                        [
                            tf.slice(hidden_weights, [0, 0], [input_node_amount, remove_index]),
                            tf.slice(hidden_weights, [0, remove_index + 1],
                                     [input_node_amount, hidden_node_amount - remove_index - 1])
                        ],
                        1
                    )
                    """
                    hidden_weights_val, exam_hidden_weights_val, train_val = sess.run([hidden_weights, exam_hidden_weights, train])
                    print(hidden_weights.get_shape())
                    print(exam_hidden_weights.get_shape())
                    print(hidden_weights_val)
                    print(exam_hidden_weights_val)  # 此二者比較可確認 移除 input->hidden weight 的正確性
                    print(train_val)
                    """
                    exam_hidden_thresholds = tf.concat(
                        [
                            tf.slice(hidden_thresholds, [0], [remove_index]),
                            tf.slice(hidden_thresholds, [remove_index + 1], [hidden_node_amount - remove_index - 1])
                        ],
                        0
                    )
                    """
                    hidden_thresholds_val, exam_hidden_thresholds_val = sess.run(
                        [hidden_thresholds, exam_hidden_thresholds])
                    print(hidden_thresholds.get_shape())
                    print(exam_hidden_thresholds.get_shape())
                    print(hidden_thresholds_val)
                    print(exam_hidden_thresholds_val)  # 此二者比較可確認 移除 input->hidden thresholds 的正確性
                    """
                    exam_tau_array = np.delete(tau_in_each_hidden_node, remove_index)
                    """
                    print(tau_in_each_hidden_node)
                    print(exam_tau_array)  # 此二者比較可確認 移除 input->hidden tau 的正確性
                    """
                    exam_output_weights = tf.concat(
                        [
                            tf.slice(output_weights, [0, 0], [remove_index, output_node_amount]),
                            tf.slice(output_weights, [remove_index + 1, 0],
                                     [hidden_node_amount - remove_index - 1, output_node_amount])
                        ], 0)
                    """
                    output_weights_val, exam_output_weights_val = sess.run([output_weights, exam_output_weights])
                    print(output_weights.get_shape())
                    print(exam_output_weights.get_shape())
                    print(output_weights_val)
                    print(exam_output_weights_val)  # 此二者比較可確認 移除 hidden->output weight 的正確性
                    """

                    # build exam Graph
                    exam_hidden_layer_before_tanh = tf.add(
                        tf.matmul(
                            x_placeholder,
                            exam_hidden_weights),
                        exam_hidden_thresholds
                    )
                    exam_hidden_layer = tf.tanh(
                        tf.multiply(exam_hidden_layer_before_tanh,
                                    tf.pow(
                                        tf.constant(2.0, dtype=tf.float64),
                                        tau_placeholder)
                                    )
                    )
                    exam_output_layer = tf.add(tf.matmul(exam_hidden_layer, exam_output_weights), output_threshold)
                    exam_y, exam_hidden_weights_val, exam_hidden_thresholds_val, exam_output_weights_val, output_threshold_val = sess.run(
                        [  # fetch value of target tensors
                            exam_output_layer,
                            exam_hidden_weights,
                            exam_hidden_thresholds,
                            exam_output_weights,
                            output_threshold
                        ],
                        {  # input to placeholder
                            x_placeholder: current_stage_x_training_data,
                            y_placeholder: current_stage_y_training_data,
                            tau_placeholder: exam_tau_array
                        }
                    )
                    # check condition L
                    exam_alpha = tf.double.max
                    exam_beta = tf.double.min

                    it = np.nditer(exam_y, flags=['f_index'])
                    while not it.finished:
                        if current_stage_y_training_data[it.index] == 1:
                            if it[0] < exam_alpha:
                                exam_alpha = it[0]
                        if current_stage_y_training_data[it.index] == -1:
                            if it[0] > exam_beta:
                                exam_beta = it[0]
                        it.iternext()
                    print('-' * 5 + "exam hidden node #{}".format(remove_index) + '-' * 5)
                    print(list(zip(exam_y, current_stage_y_training_data)))
                    print('exam_alpha= ' + str(exam_alpha))
                    print('exam_beta= ' + str(exam_beta))
                    if alpha <= beta:
                        print('pruning current hidden node #{0} will violate condition L'.format(remove_index))
                        print('-' * 10)
                        continue
                    else:
                        print("pruning current hidden node #{0} won't violate condition L".format(remove_index))
                        print("!!!!! REMOVE hidden node #{0} !!!!!".format(remove_index))
                        hidden_node_amount -= 1
                        """
                        print('-' * 5 + 'after pruning' + '-' * 5)
                        print(exam_hidden_weights_val)
                        print(exam_hidden_thresholds_val)
                        print(exam_output_weights_val)
                        print(output_threshold_val)
                        print(exam_tau_array)
                        print('-' * 10)
                        """
                        # set new hidden node value
                        tau_in_each_hidden_node = exam_tau_array
                        current_hidden_weights = exam_hidden_weights_val
                        current_hidden_thresholds = exam_hidden_thresholds_val
                        current_output_weights = exam_output_weights_val
                        current_output_threshold = output_threshold_val

                        assign_new_hidden_weight = tf.assign(hidden_weights, exam_hidden_weights_val,
                                                             validate_shape=False)
                        assign_new_hidden_threshold = tf.assign(hidden_thresholds, exam_hidden_thresholds_val,
                                                                validate_shape=False)
                        assign_new_output_weights = tf.assign(output_weights, exam_output_weights_val,
                                                              validate_shape=False)
                        # tf.assign(output_threshold, output_threshold_val, validate_shape=False)
                        sess.run([assign_new_hidden_weight, assign_new_hidden_threshold, assign_new_output_weights])
                        break

