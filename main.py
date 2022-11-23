import numpy
import matplotlib.pyplot as plt
from NeuralNetwork import NeuralNetwork

i_nodes = 3
h_nodes = 3
o_nodes = 3
l_rate = 0.4

neural_network = NeuralNetwork(i_nodes, h_nodes, o_nodes, l_rate)
print(neural_network.query([1.0, 0.5, -1.5]))

i = 1
stop_index = 0
while i < 10000 and stop_index < 0.95:
    neural_network.train([0.6, 0.2, 0.54], [1, 0, 0])
    a = neural_network.query([1.0, 0.5, -1.5])
    stop_index = a[0,0]
    print("Iteration # ", i)
    print(a)
    i+= 1
    pass


# plt.imshow(a, interpolation="nearest")
# plt.show()
