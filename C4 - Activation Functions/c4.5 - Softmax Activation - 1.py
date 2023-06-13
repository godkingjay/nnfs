layer_outputs = [4.8, 1.21, 2.385]

E = 2.71828182846

exp_values = []

for output in layer_outputs:
    exp_values.append(E ** output)

print(exp_values)
