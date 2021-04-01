import numpy as np
# sorry about the lack of comments, im still figuring out what exactly is going on

class Network:
    def __init__(self, size_input, size_output):
        self.size_hidden = int(2 / 3 * size_input + size_output)
        self.size_input = size_input
        self.size_output = size_output

        # the 2 transformation matrices from the input to the output of the network given a random starting point
        self.transform_in_hid = np.random.rand(self.size_hidden, self.size_input)
        self.bias_in_hid = np.random.rand(self.size_hidden, 1)
        self.transform_hid_out = np.random.rand(self.size_output, self.size_hidden)
        self.bias_hid_out = np.random.rand(self.size_output, 1)

        print("Neural network initialised.")

    def learn(self, dataset):
        pass

    def backpropagate_layer(self, differences, transformation_matrix, neurons_previous):
        pass

    @staticmethod
    def error_output_layer(expected_output_layer, actual_output_layer):
        return (expected_output_layer - actual_output_layer) * Network.activate_prime(actual_output_layer)

    def error_hidden_layer(self, error_output_layer, actual_hidden_layer):
        pass

    def predict(self, input_layer):
        print(input_layer)
        hidden_layer_unactivated = np.matmul(self.transform_in_hid, input_layer) + self.bias_in_hid
        hidden_layer = Network.activate(hidden_layer_unactivated)
        print("Propagated to hidden layer")
        output_layer_unactivated = np.matmul(self.transform_hid_out, hidden_layer) + self.bias_hid_out
        output_layer = Network.activate(output_layer_unactivated)
        print("Propagated to output layer")
        return output_layer, hidden_layer

    @staticmethod
    def cost(outputs, expected):
        cost = 0
        for index, output in enumerate(outputs):
            cost += (output-expected[index])**2
        return cost

    @staticmethod
    def activate(value):
        return 1/(1+np.exp(-1*value))

    @staticmethod
    def activate_prime(value_of_sigmoid):
        return value_of_sigmoid * (1 - value_of_sigmoid)
