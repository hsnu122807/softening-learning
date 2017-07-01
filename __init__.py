import tensorflow as tf
import numpy as np
import time
import random
import os


dir_path = os.path.dirname(os.path.realpath(__file__))
execute_start_time = time.time()

# file_input = "binary_training_input_data"
# file_output = "binary_training_output_data"
data_amount = '100'
file_input = "tensorflow_binary_input_" + data_amount
file_output = "tensorflow_binary_output_" + data_amount
x_training_data = np.loadtxt(file_input + ".txt", dtype=float, delimiter=" ")
y_training_data = np.loadtxt(file_output + ".txt", dtype=float, delimiter=" ")
tau_in_each_hidden_node = np.array([1.0])

# Network Parameters
input_node_amount = x_training_data.shape[1]
hidden_node_amount = 1
output_node_amount = 1
learning_rate_eta = 0.001

# Parameters
thinking_times = 5000
data_size = x_training_data.shape[0]
big_number = 15

# counters
thinking_times_count = 0
cramming_times_count = 0
softening_thinking_times_count = 0
pruning_success_times_count = 0

# placeholders
x_placeholder = tf.placeholder(tf.float64)
y_placeholder = tf.placeholder(tf.float64)
tau_placeholder = tf.placeholder(tf.float64)

# network architecture
output_threshold = tf.Variable(tf.random_normal([output_node_amount], dtype=tf.float64))
output_weights = tf.Variable(tf.random_normal([hidden_node_amount, output_node_amount], dtype=tf.float64))
tau_in_each_hidden_node = np.tile(tau_in_each_hidden_node, (hidden_node_amount, 1))
hidden_thresholds = tf.Variable(tf.random_normal([hidden_node_amount], dtype=tf.float64))
hidden_weights = tf.Variable(tf.random_normal([input_node_amount, hidden_node_amount], dtype=tf.float64))

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
    '''
    !!!!!!!!!!!重要!!!!!!!!!!!
    tsaih(1993)的softening learning在cramming的時候必須確保除去最新一筆資料以外的所有資料已經符合condition L
    否則cramming 無效
    因為cramming這個方法只針對硬背一筆資料而已
    
    也因此不能挑residual最小的幾筆資料
    因為可能不只需要調整一筆資料來滿足condition L
    
    應該問蔡老師在outlier detection的時候
    硬背的部分該怎麼做到
    會先確保除了最新的一筆(硬背目標)以外都滿足in envelope嗎?
    '''
    # if k == 1:
    #     # get a random training case
    #     r = random.randint(0, data_size-1)
    #     current_stage_x_training_data = x_training_data[r-1:r]
    #     current_stage_y_training_data = y_training_data[r-1:r]
    # else:
    #     # pick k data of smallest residual
    #     predict_y = sess.run([output_layer],
    #                          {x_placeholder: x_training_data,
    #                           y_placeholder: y_training_data,
    #                           tau_placeholder: tau_in_each_hidden_node})
    #     squared_residuals = np.square(predict_y[0] - y_training_data.reshape((-1, 1)))
    #     # print(squared_residuals)
    #     concat_residual_and_x = np.concatenate((squared_residuals, x_training_data), axis=1)
    #     concat_residual_and_y = np.concatenate((squared_residuals, y_training_data.reshape((data_size, output_node_amount))), axis=1)
    #     sort_concat_x = concat_residual_and_x[np.argsort(concat_residual_and_x[:, 0])]
    #     sort_concat_y = concat_residual_and_y[np.argsort(concat_residual_and_y[:, 0])]
    #     x_training_data_sort_by_residual = np.delete(sort_concat_x, 0, axis=1)
    #     y_training_data_sort_by_residual = np.delete(sort_concat_y, 0, axis=1)
    #     # print(sort_concat_x)
    #     # print(sort_concat_y)
    #     # print(x_training_data_sort_by_residual)
    #     # print(y_training_data_sort_by_residual)
    #     current_stage_x_training_data = x_training_data_sort_by_residual[:k]
    #     current_stage_y_training_data = y_training_data_sort_by_residual[:k]
    #     # print(current_stage_x_training_data)
    #     # print(current_stage_y_training_data)

    # take first k training case
    current_stage_x_training_data = x_training_data[:k]
    current_stage_y_training_data = y_training_data[:k]

    # calculate alpha & beta in condition L
    predict_y = sess.run([output_layer],
                         {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data,
                          tau_placeholder: tau_in_each_hidden_node})

    # for node in tf.get_default_graph().as_graph_def().node:
    #     print(node.name)
    print(predict_y)
    # check condition L
    class_1_output = [tf.double.max]
    class_2_output = [tf.double.min]
    it = np.nditer(predict_y, flags=['f_index'])
    while not it.finished:
        if current_stage_y_training_data[it.index] == 1:
            class_1_output.append(it[0])
        else:  # if current_stage_y_training_data[it.index] == -1:
            class_2_output.append(it[0])
        it.iternext()
    alpha, beta = tool_sess.run(
        [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})
    print('alpha = {0}   beta = {1}'.format(alpha, beta))
    class_1_output = [tf.double.max]
    class_2_output = [tf.double.min]
    it = np.nditer(predict_y, flags=['f_index'])
    while not it.finished:
        if current_stage_y_training_data[it.index] == 1:
            class_1_output.append(it[0])
        else:  # if current_stage_y_training_data[it.index] == -1:
            class_2_output.append(it[0])
        it.iternext()
    if current_stage_y_training_data[k - 1] == 1:  # new training case is class 1
        class_1_output = class_1_output[:-1]
    else:
        class_2_output = class_2_output[:-1]
    min_predict_value_in_class_one_of_previous_stage_training_case, max_predict_value_in_class_two_of_previous_stage_training_case = tool_sess.run(
        [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})
    print(min_predict_value_in_class_one_of_previous_stage_training_case)
    print(max_predict_value_in_class_two_of_previous_stage_training_case)

    if alpha > beta:
        print('new training case is familiar to us, no further learning effort involved.')
    else:
        # thinking
        print('---start thinking---')
        thinking_failed = False
        saver.save(sess, r"{0}/model.ckpt".format(dir_path))
        for stage in range(thinking_times):
            sess.run(train, feed_dict={x_placeholder: current_stage_x_training_data,
                                       y_placeholder: current_stage_y_training_data,
                                       tau_placeholder: tau_in_each_hidden_node})
            thinking_times_count += 1

            predict_y = sess.run([output_layer],
                                 {x_placeholder: current_stage_x_training_data,
                                  y_placeholder: current_stage_y_training_data,
                                  tau_placeholder: tau_in_each_hidden_node})
            # check condition L
            class_1_output = [tf.double.max]
            class_2_output = [tf.double.min]
            it = np.nditer(predict_y, flags=['f_index'])
            while not it.finished:
                if current_stage_y_training_data[it.index] == 1:
                    class_1_output.append(it[0])
                else:  # if current_stage_y_training_data[it.index] == -1:
                    class_2_output.append(it[0])
                it.iternext()
            alpha, beta = tool_sess.run(
                [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})
            print('thinking {0} times, current alpha = {1}, current beta = {2}'.format((stage+1),alpha,beta))
            if alpha > beta:
                print('thinking success!!!')
                break
            else:
                if stage == (thinking_times-1):
                    thinking_failed = True
                    print('thinking failed: after {0} times training, alpha still smaller than beta.'.format((stage + 1)))
                    # MUST restore before cramming(因為調權重可能會讓先前的資料違反condition L)
                    print('restore weights.')
                    saver.restore(sess, r"{0}/model.ckpt".format(dir_path))

        if thinking_failed:
            # cram it first
            print('start cramming')
            cramming_times_count += 1
            # calculate relevant parameters
            hidden_node_amount += 1
            current_hidden_weights, current_hidden_thresholds, current_output_weights, current_output_threshold, = sess.run(
                [hidden_weights, hidden_thresholds, output_weights, output_threshold],
                {x_placeholder: current_stage_x_training_data, y_placeholder: current_stage_y_training_data,
                 tau_placeholder: tau_in_each_hidden_node})
            predict_y = sess.run([output_layer],
                                 {x_placeholder: current_stage_x_training_data,
                                  y_placeholder: current_stage_y_training_data,
                                  tau_placeholder: tau_in_each_hidden_node})
            # print('current hidden weights:')
            # print(current_hidden_weights)
            # print('current hidden thresholds:')
            # print(current_hidden_thresholds)
            # print('current output weights:')
            # print(current_output_weights)
            # print('current output threshold:')
            # print(current_output_threshold)
            # calculate new hidden weight
            new_hidden_node_neuron_weights = current_stage_x_training_data[k - 1]
            # print('new hidden weights:')
            # print(new_hidden_node_neuron_weights)
            # calculate new hidden threshold
            new_hidden_node_threshold = 1 - input_node_amount
            # print('new hidden thresholds:')
            # print(new_hidden_node_threshold)
            # calculate new output weight
            if current_stage_y_training_data[k - 1] == 1:
                # 錯的原因已發現 因為上一個版本thinking完 在model restore之後沒有重新取predict y的值 導致運算出錯
                # new_output_node_neuron_weight = predict_y[0][k - 1] - max_predict_value_in_class_two_of_previous_stage_training_case  # 估計真的是錯的
                new_output_node_neuron_weight = max_predict_value_in_class_two_of_previous_stage_training_case - predict_y[0][k - 1]
            if current_stage_y_training_data[k - 1] == -1:
                # new_output_node_neuron_weight = min_predict_value_in_class_one_of_previous_stage_training_case - predict_y[0][k - 1]
                new_output_node_neuron_weight = predict_y[0][k - 1] - min_predict_value_in_class_one_of_previous_stage_training_case
            # print('predict value of most recent training case: ' + str(predict_y[0][k - 1]))
            # print('new output weight:')
            # print(new_output_node_neuron_weight)

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

                # save current alpha & beta
                predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data,
                                                      y_placeholder: current_stage_y_training_data,
                                                      tau_placeholder: tau_in_each_hidden_node})
                print(predict_y)

                # check condition L
                class_1_output = [tf.double.max]
                class_2_output = [tf.double.min]
                it = np.nditer(predict_y, flags=['f_index'])
                while not it.finished:
                    if current_stage_y_training_data[it.index] == 1:
                        class_1_output.append(it[0])
                    else:  # if current_stage_y_training_data[it.index] == -1:
                        class_2_output.append(it[0])
                    it.iternext()
                alpha, beta = tool_sess.run(
                    [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})
                print('after cramming, alpha = {0}, beta = {1}'.format(alpha, beta))

                # softening
                # save variables
                saver.save(sess, r"{0}/model.ckpt".format(dir_path))

                # change tau value of newest hidden node
                newest_hidden_node_tau_value = tau_in_each_hidden_node[hidden_node_amount - 1]
                print(newest_hidden_node_tau_value)
                while newest_hidden_node_tau_value > 1:
                    newest_hidden_node_tau_value -= 1
                    tau_in_each_hidden_node[hidden_node_amount - 1] = newest_hidden_node_tau_value
                    print('tau array:')
                    print(tau_in_each_hidden_node)
                    softening_success = False

                    for times in range(thinking_times):
                        # forward pass
                        predict_y = sess.run([output_layer], {x_placeholder: current_stage_x_training_data,
                                                              y_placeholder: current_stage_y_training_data,
                                                              tau_placeholder: tau_in_each_hidden_node})
                        # print(predict_y)

                        # check condition L
                        class_1_output = [tf.double.max]
                        class_2_output = [tf.double.min]
                        it = np.nditer(predict_y, flags=['f_index'])
                        while not it.finished:
                            if current_stage_y_training_data[it.index] == 1:
                                class_1_output.append(it[0])
                            else:  # if current_stage_y_training_data[it.index] == -1:
                                class_2_output.append(it[0])
                            it.iternext()
                        test_alpha, test_beta = tool_sess.run(
                            [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})

                        # print(alpha)
                        # print(beta)
                        if test_alpha > test_beta:
                            alpha = test_alpha
                            beta = test_beta
                            print(
                                'softening success, gradient descent trained {0} times, #{1} tau value decrease by 1, current tau value: {2}'.format(
                                    times, hidden_node_amount, newest_hidden_node_tau_value))
                            softening_success = True
                            saver.save(sess, r"{0}/model.ckpt".format(dir_path))
                            break
                        else:
                            sess.run(train, feed_dict={x_placeholder: current_stage_x_training_data,
                                                       y_placeholder: current_stage_y_training_data,
                                                       tau_placeholder: tau_in_each_hidden_node})
                            softening_thinking_times_count += 1

                    if not softening_success:
                        print(
                            'softening failed, gradient descent trained {0} times, restore #{1} tau value , restore tau value: {2}'.format(
                                times, hidden_node_amount, newest_hidden_node_tau_value + 1))
                        tau_in_each_hidden_node[hidden_node_amount - 1] = newest_hidden_node_tau_value + 1
                        saver.restore(sess, r"{0}/model.ckpt".format(dir_path))
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
                            (
                            current_hidden_weights[..., :remove_index], current_hidden_weights[..., remove_index + 1:]),
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
                            exam_hidden_layer_before_tanh = tf.add(tf.matmul(exam_x_holder, exam_hidden_weights_var),
                                                                   exam_hidden_thresholds_var)
                            exam_hidden_layer = tf.tanh(
                                tf.multiply(exam_hidden_layer_before_tanh,
                                            tf.pow(tf.constant(2.0, dtype=tf.float64), exam_tau_holder)))
                            exam_output_layer = tf.add(tf.matmul(exam_hidden_layer, exam_output_weights_var),
                                                       exam_output_threshold_var)

                            # exam goal & optimizer
                            exam_average_squared_residual = tf.reduce_mean(
                                tf.reduce_sum(tf.square(exam_y_holder - exam_output_layer), reduction_indices=[1]))
                            exam_train = tf.train.GradientDescentOptimizer(learning_rate_eta).minimize(
                                exam_average_squared_residual)
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
                        class_1_output = [tf.double.max]
                        class_2_output = [tf.double.min]

                        it = np.nditer(exam_predict_y, flags=['f_index'])
                        while not it.finished:
                            if current_stage_y_training_data[it.index] == 1:
                                class_1_output.append(it[0])
                            else:  # if current_stage_y_training_data[it.index] == -1:
                                class_2_output.append(it[0])
                            it.iternext()
                        exam_alpha, exam_beta = tool_sess.run(
                            [min_alpha, max_beta], {tool_alpha: class_1_output, tool_beta: class_2_output})
                        # print('*' * 5 + "exam removing hidden node #{}".format(remove_index) + '*' * 5)
                        # print("***** exam #{0} variables *****".format(remove_index))
                        # print(exam_h_w_val)
                        # print(exam_h_t_val)
                        # print(exam_o_w_val)
                        # print(exam_o_t_val)
                        # print("***** exam #{0} predict_y *****".format(remove_index))
                        # print(exam_predict_y)
                        print("***** exam #{0} alpha & beta *****".format(remove_index))
                        # print(list(zip(exam_predict_y, current_stage_y_training_data)))
                        print('exam_alpha= ' + str(exam_alpha))
                        print('exam_beta= ' + str(exam_beta))
                        if exam_alpha <= exam_beta:
                            print('pruning current hidden node #{0} will violate condition L'.format(remove_index))
                            print('*' * 10)
                            continue
                        else:
                            alpha = exam_alpha
                            beta = exam_beta
                            pruning_success_times_count += 1
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
    # if k == data_size:
    #     new_path = r"{0}/".format(dir_path) + file_output
    #     if not os.path.exists(new_path):
    #         os.makedirs(new_path)
    #
    #     file = open(new_path + r"\_training_detail.txt", 'w')
    #     file.writelines("learning_rate: " + str(learning_rate_eta) + "\n")
    #     file.writelines("input_node_amount: " + str(input_node_amount) + "\n")
    #     file.writelines("hidden_node_amount: " + str(hidden_node_amount) + "\n")
    #     file.writelines("output_node_amount: " + str(output_node_amount) + "\n")
    #     file.writelines("training_data_amount: " + str(data_size) + "\n")
    #     file.writelines("alpha(class 1 min value): " + str(alpha) + "\n")
    #     file.writelines("beta(class 2 max value): " + str(beta) + "\n")
    #     file.writelines("thinking_times_count: " + str(thinking_times_count) + "\n")
    #     file.writelines("cramming_times_count: " + str(cramming_times_count) + "\n")
    #     file.writelines("softening_thinking_times_count: " + str(softening_thinking_times_count) + "\n")
    #     file.writelines("pruning_success_times_count: " + str(pruning_success_times_count) + "\n")
    #     file.writelines(
    #         "total execution time: " + str(time.time() - execute_start_time) + " seconds" + "\n")
    #     file.close()
    #     curr_hidden_neuron_weight = sess.run([hidden_weights], {x_placeholder: x_training_data,
    #                                                             y_placeholder: y_training_data,
    #                                                             tau_placeholder: tau_in_each_hidden_node})
    #     np.savetxt(new_path + r"\hidden_neuron_weight.txt", curr_hidden_neuron_weight)
    #     curr_hidden_threshold = sess.run([hidden_thresholds],
    #                                      {x_placeholder: x_training_data, y_placeholder: y_training_data,
    #                                       tau_placeholder: tau_in_each_hidden_node})
    #     np.savetxt(new_path + r"\hidden_threshold.txt", curr_hidden_threshold)
    #     curr_output_neuron_weight = sess.run([output_weights], {x_placeholder: x_training_data,
    #                                                             y_placeholder: y_training_data,
    #                                                             tau_placeholder: tau_in_each_hidden_node})
    #     np.savetxt(new_path + r"\output_neuron_weight.txt", curr_output_neuron_weight)
    #     curr_output_threshold = sess.run([output_threshold],
    #                                      {x_placeholder: x_training_data, y_placeholder: y_training_data,
    #                                       tau_placeholder: tau_in_each_hidden_node})
    #     np.savetxt(new_path + r"\output_threshold.txt", curr_output_threshold)
    #
    #     curr_average_loss = sess.run([average_squared_residual],
    #                                  {x_placeholder: x_training_data, y_placeholder: y_training_data,
    #                                   tau_placeholder: tau_in_each_hidden_node})
    #     file.writelines("average_loss_of_the_model: " + str(curr_average_loss) + "\n")
    #
    #     np.savetxt(new_path + r"\tau_in_each_hidden_node.txt", tau_in_each_hidden_node)
    #
    #     print("--- execution time: %s seconds ---" % (time.time() - execute_start_time))

    if k == data_size:
        new_path = r"{0}/".format(dir_path) + file_output
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        curr_hidden_neuron_weight, curr_hidden_threshold, curr_output_neuron_weight, curr_output_threshold, curr_average_loss, curr_output = sess.run(
            [hidden_weights, hidden_thresholds,
             output_weights, output_threshold, average_squared_residual,
             output_layer],
            {x_placeholder: x_training_data, y_placeholder: y_training_data, tau_placeholder: tau_in_each_hidden_node})
        np.savetxt(new_path + r"\hidden_neuron_weight.txt", curr_hidden_neuron_weight)
        np.savetxt(new_path + r"\hidden_threshold.txt", curr_hidden_threshold)
        np.savetxt(new_path + r"\output_neuron_weight.txt", curr_output_neuron_weight)
        np.savetxt(new_path + r"\output_threshold.txt", curr_output_threshold)
        np.savetxt(new_path + r"\tau_in_each_hidden_node.txt", tau_in_each_hidden_node)
        file = open(new_path + r"\_training_detail.txt", 'w')
        file.writelines("learning_rate: " + str(learning_rate_eta) + "\n")
        file.writelines("input_node_amount: " + str(input_node_amount) + "\n")
        file.writelines("hidden_node_amount: " + str(hidden_node_amount) + "\n")
        file.writelines("output_node_amount: " + str(output_node_amount) + "\n")
        file.writelines("training_data_amount: " + str(data_size) + "\n")
        file.writelines("average_loss_of_the_model: " + str(curr_average_loss) + "\n")
        file.writelines("alpha(class 1 min value): " + str(alpha) + "\n")
        file.writelines("beta(class 2 max value): " + str(beta) + "\n")
        file.writelines("thinking_times_count: " + str(thinking_times_count) + "\n")
        file.writelines("cramming_times_count: " + str(cramming_times_count) + "\n")
        file.writelines("softening_thinking_times_count: " + str(softening_thinking_times_count) + "\n")
        file.writelines("pruning_success_times_count: " + str(pruning_success_times_count) + "\n")
        file.writelines("total execution time: " + str(time.time() - execute_start_time) + " seconds" + "\n")
        file.close()
        print("--- execution time: %s seconds ---" % (time.time() - execute_start_time))