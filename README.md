# softening-learning

現在big_number設25也沒問題

暫時找不到可以替換variable tensor的方法，目前採用重建session的方式

固定了初始權重和偏移量

log出許多運算的過程，發現cram的機制好像無法讓當前的training case符合condition L
