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

# saver
saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

tool_graph = tf.Graph()
with tool_graph.as_default():
    tool_alpha = tf.placeholder(tf.float64)
    tool_beta = tf.placeholder(tf.float64)
    min_alpha = tf.reduce_min(tool_alpha)
    max_beta = tf.reduce_max(tool_beta)
    tool_init = tf.global_variables_initializer()
tool_sess = tf.Session(graph=tool_graph)
tool_sess.run([tool_init])

for k in range(1, data_size + 1):
    print('-----stage: ' + str(k) + '-----')
    # take first k training case
    current_stage_x_training_data = x_training_data[:k]
    current_stage_y_training_data = y_training_data[:k]

    # thinking
    saver.save(sess, r"C:\Users\Lee Chia Lun\PycharmProjects\autoencoder\softening_learning\model.ckpt")
    for i in range(1000):
        sess.run(train, feed_dict={x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data, tau_placeholder: tau_in_each_hidden_node})
        current_average_squared_residual = sess.run([average_squared_residual], {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data, tau_placeholder: tau_in_each_hidden_node})
        # print(current_average_squared_residual)
        if current_average_squared_residual[0] < 0.1:
            print('after thinking, average squared residual: ')
            print(current_average_squared_residual)
            break
        if i == 999:
            print('thinking failed, current average squared residual:')
            print(current_average_squared_residual)
            # if needed, restore weight before cramming
            # print('restore weights.')
            # saver.restore(sess, r"C:\Users\Lee Chia Lun\PycharmProjects\autoencoder\softening_learning\model.ckpt")

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
                    class_1_output = []
                    class_2_output = []
                    it = np.nditer(predict_y, flags=['f_index'])
                    while not it.finished:
                        if current_stage_y_training_data[it.index] == 1:
                            class_1_output.append(it[0])
                        else:  # if current_stage_y_training_data[it.index] == -1:
                            class_2_output.append(it[0])
                        it.iternext()
                    alpha, beta = tool_sess.run(
                        [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})

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
            if hidden_node_amount > 1:  # equals to the number of hidden nodes
                # get current weights & thresholds
                current_hidden_weights = hidden_weights.eval(sess)
                current_hidden_thresholds = hidden_thresholds.eval(sess)
                current_output_weights = output_weights.eval(sess)
                current_output_threshold = output_threshold.eval(sess)
                """
                print('#' * 10)
                print(current_hidden_weights)
                print(current_hidden_thresholds)
                print(current_output_weights)
                print(current_output_threshold)
                print('#' * 10)
                """

                # then try pruning from the begining hidden node
                for remove_index in range(hidden_node_amount):
                    # 算出欲檢驗的結構之 weight 和 threshold
                    exam_hidden_weights = np.concatenate(
                        (current_hidden_weights[..., :remove_index], current_hidden_weights[..., remove_index + 1:]),
                        axis=1)
                    exam_hidden_thresholds = np.concatenate(
                        (current_hidden_thresholds[:remove_index], current_hidden_thresholds[remove_index + 1:]),
                        axis=0)
                    exam_tau = np.delete(tau_in_each_hidden_node, remove_index)
                    exam_output_weights = np.concatenate(
                        (current_output_weights[:remove_index], current_output_weights[remove_index + 1:]), axis=0)

                    # 建立測試 pruning 可行性的 Graph
                    exam_graph = tf.Graph()
                    with exam_graph.as_default():
                        # placeholders
                        exam_x_holder = tf.placeholder(tf.float64)
                        exam_y_holder = tf.placeholder(tf.float64)
                        exam_tau_holder = tf.placeholder(tf.float64)

                        # exam variables
                        exam_hidden_weights_var = tf.Variable(exam_hidden_weights)
                        exam_hidden_thresholds_var = tf.Variable(exam_hidden_thresholds)
                        exam_output_weights_var = tf.Variable(exam_output_weights)
                        exam_output_threshold_var = tf.Variable(current_output_threshold)

                        # exam tensors
                        exam_hidden_layer_before_tanh = tf.add(tf.matmul(exam_x_holder, exam_hidden_weights_var), exam_hidden_thresholds_var)
                        exam_hidden_layer = tf.tanh(
                            tf.multiply(exam_hidden_layer_before_tanh,
                                        tf.pow(tf.constant(2.0, dtype=tf.float64), exam_tau_holder)))
                        exam_output_layer = tf.add(tf.matmul(exam_hidden_layer, exam_output_weights_var), exam_output_threshold_var)

                        # exam goal & optimizer
                        exam_average_squared_residual = tf.reduce_mean(
                            tf.reduce_sum(tf.square(exam_y_holder - exam_output_layer), reduction_indices=[1]))
                        exam_train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(exam_average_squared_residual)
                        exam_init = tf.global_variables_initializer()

                        # saver
                        exam_saver = tf.train.Saver()

                    # 使用 exam Session 執行 exam Graph
                    exam_sess = tf.Session(graph=exam_graph)
                    exam_sess.run(exam_init)
                    exam_h_w_val, exam_h_t_val, exam_o_w_val, exam_o_t_val, exam_predict_y = exam_sess.run(
                        [
                            exam_hidden_weights_var,
                            exam_hidden_thresholds_var,
                            exam_output_weights_var,
                            exam_output_threshold_var,
                            exam_output_layer
                        ],
                        {
                            exam_x_holder: current_stage_x_training_data,
                            exam_y_holder: current_stage_y_training_data,
                            exam_tau_holder: exam_tau
                        }
                    )

                    # check if exam_alpha & exam_beta match condition L
                    exam_alpha = tf.double.max
                    exam_beta = tf.double.min
                    class_1_output = []
                    class_2_output = []

                    it = np.nditer(exam_predict_y, flags=['f_index'])
                    while not it.finished:
                        if current_stage_y_training_data[it.index] == 1:
                            class_1_output.append(it[0])
                        else:  # if current_stage_y_training_data[it.index] == -1:
                            class_2_output.append(it[0])
                        it.iternext()
                    exam_alpha, exam_beta = tool_sess.run(
                        [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})
                    print('*' * 5 + "exam removing hidden node #{}".format(remove_index) + '*' * 5)
                    print("***** exam #{0} variables *****".format(remove_index))
                    print(exam_h_w_val)
                    print(exam_h_t_val)
                    print(exam_o_w_val)
                    print(exam_o_t_val)
                    print("***** exam #{0} predict_y *****".format(remove_index))
                    print(exam_predict_y)
                    print("***** exam #{0} alpha & beta *****".format(remove_index))
                    print(list(zip(exam_predict_y, current_stage_y_training_data)))
                    print('exam_alpha= ' + str(exam_alpha))
                    print('exam_beta= ' + str(exam_beta))
                    if exam_alpha <= exam_beta:
                        print('pruning current hidden node #{0} will violate condition L'.format(remove_index))
                        print('*' * 10)
                        continue
                    else:
                        print("pruning current hidden node #{0} won't violate condition L".format(remove_index))
                        print("!!!!! REMOVE hidden node #{0} !!!!!".format(remove_index))
                        print('*' * 10)
                        hidden_node_amount -= 1
                        # 直接使用測試成功的 Session 和 Graph 取代舊的
                        sess = exam_sess
                        # 更換 placeholders 操作指標
                        x_placeholder = exam_x_holder
                        y_placeholder = exam_y_holder
                        tau_placeholder = exam_tau_holder
                        # 更換 variables 操作指標
                        hidden_weights = exam_hidden_weights_var
                        hidden_thresholds = exam_hidden_thresholds_var
                        output_weights = exam_output_weights_var
                        output_threshold = exam_output_threshold_var
                        output_layer = exam_output_layer
                        # 更換其他操作指標
                        hidden_layer_before_tanh = exam_hidden_layer_before_tanh
                        hidden_layer = exam_hidden_layer
                        output_layer = exam_output_layer
                        average_squared_residual = exam_average_squared_residual
                        train = exam_train
                        saver = exam_saver
                        # modify constant
                        tau_in_each_hidden_node = exam_tau
                        break
