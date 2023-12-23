import numpy as np
import neurolab as nl

target = [[-1, 1, -1, -1, 1, -1, -1, 1, -1],
          [1, 1, 1, 1, -1, 1, 1, -1, 1],
          [1, -1, 1, 1, 1, 1, 1, -1, 1],
          [1, 1, 1, 1, -1, -1, 1, -1, -1],
          [-1, -1, -1, -1, 1, -1, -1, -1, -1]]

input = [[-1, -1, 1, 1, 1, 1, 1, -1, 1],
         [-1, -1, 1, -1, 1, -1, -1, -1, -1],
         [-1, -1, -1, -1, 1, -1, -1, 1, -1]]

net = nl.net.newhem(target)

output = net.sim(target)
print("Навчання на тренувальних зразках")
print(np.argmax(output, axis=0))

output = net.sim([input[0]])
print("Виводи в рекурентному циклі:")
print(np.array(net.layers[1].outs))

output = net.sim(input)
print("Виводи в тренувальному зразку:")
print(output)
