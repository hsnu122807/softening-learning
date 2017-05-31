# softening-learning

替換variable tensor的方式終於找到了(使用tf.assign)

固定了初始權重和偏移量

加減hidden node可以往兩個方向思考:

1.切換不同的graph

2.維持一個大型矩陣當hidden weight，並mask掉目前不參與運算的部分
