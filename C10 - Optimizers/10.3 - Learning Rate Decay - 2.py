starting_learning_rate = 1.0
learning_rate_decay = 0.1
step = 20

learning_rate = starting_learning_rate * \
    (1 / (1 + (learning_rate_decay * step)))

print(learning_rate)
