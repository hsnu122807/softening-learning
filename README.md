# softening-learning

現在big_number設25也沒問題

替換variable tensor的方式終於找到了(使用tf.assign)

固定了初始權重和偏移量

log出許多運算的過程，發現cram的機制好像無法讓當前的training case符合condition L
