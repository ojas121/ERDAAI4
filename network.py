import numpy as np


# some equations from https://sudeepraja.github.io/Neural/
class Network:
    def __init__(self, size_input, size_output, learning_rate, in_hid=[], hid_out=[]):
        # Set hidden layer size based on rule of thumb
        print("Starting initialisation")
        # self.size_hidden = int(2 / 3 * size_input + size_output)
        self.size_hidden = 1000
        self.size_input = size_input
        self.size_output = size_output

        # The 2 transformation matrices from the input to the output of the network given a random starting point
        # We do not include bias!
        if len(in_hid) == 0:
            print("Starting weight matrix 1")
            self.transform_in_hid = np.array(np.random.rand(self.size_hidden, self.size_input), dtype="half")
            print(self.transform_in_hid.shape)
        else:
            self.transform_in_hid = in_hid
        if len(hid_out) == 0:
            print("Starting weight matrix 2")
            self.transform_hid_out = np.array(np.random.rand(self.size_output, self.size_hidden), dtype="half")
            print(self.transform_hid_out.shape)
        else:
            self.transform_hid_out = hid_out

        # Arbitrary learning rate
        self.learning_rate = learning_rate

        print("Neural network initialised.")

    def learn(self, input_layer, hidden_layer, output_layer, expected_output):
        # Calculate the delta terms for both output and hidden layer
        error_output = Network.error_output_layer(expected_output, output_layer)
        error_hidden = self.error_hidden_layer(error_output, hidden_layer)

        # Derivative of cost function with respect to weights of the hidden layer
        # Delta matrix times the transpose of the input layer
        transpose_in = np.atleast_2d(input_layer)
        err_hid_2d = np.transpose(np.atleast_2d(error_hidden))
        derivative_hidden = np.matmul(err_hid_2d, transpose_in)
        # New weights = old weights - learning rate * old weights * cost function derivative (element wise mult.)
        self.transform_in_hid = self.transform_in_hid - np.multiply(self.learning_rate, np.multiply(self.transform_in_hid, derivative_hidden))

        # Same thing, just now for the weights for the output layer


        transpose_hid = np.atleast_2d(hidden_layer)
        err_out_2d = np.transpose(np.atleast_2d(error_output))
        derivative_output = np.matmul(err_out_2d, transpose_hid)
        self.transform_hid_out = self.transform_hid_out - np.multiply(self.learning_rate, np.multiply(self.transform_hid_out, derivative_output))
        print("I've learned something!")

    @staticmethod
    def error_output_layer(expected_output_layer, actual_output_layer):
        # d_output = (output_layer - expected) * f'(output_layer_before_activation)
        # element wise multiplication
        # since f' depends purely on the output of sigmoid, and not directly on the unactivated layer,
        # we just give it the actual output layer after activation so as to not calculate sigmoid unnecessarily
        return np.multiply((expected_output_layer - actual_output_layer), Network.activate_prime(actual_output_layer))

    def error_hidden_layer(self, error_output_layer, actual_hidden_layer):
        # error = (hidden layer weights transpose X output layer error) * derivative of activation function for unactivated hidden layer
        # X for matrix multiplication, and *  for element wise multiplication
        transformed = np.transpose(self.transform_hid_out[:, :])
        delta_part_1 = np.matmul(transformed, error_output_layer)
        delta_part_2 = Network.activate_prime(actual_hidden_layer)

        # element-wise multiplication
        return np.multiply(delta_part_1, delta_part_2)

    def predict(self, input_layer):
        input_layer = np.append(input_layer, [[]]) # 1 more element for bias; does not depend on neurons, so we use 1
        # calcuate unactivated hidden layer by matrix multiplication
        hidden_layer_unactivated = np.matmul(self.transform_in_hid, input_layer)
        # use sigmoid function to get actual neuron activation
        hidden_layer = Network.activate(hidden_layer_unactivated)
        # make a copy of hidden layer to output before we add the extra element for bias
        hidden_layer_copy = hidden_layer[:]

        # Now, same thing, but for the output layer
        hidden_layer = np.append(hidden_layer, [[]]) # 1 more element for bias
        output_layer_unactivated = np.matmul(self.transform_hid_out, hidden_layer)
        output_layer = Network.activate(output_layer_unactivated)
        return hidden_layer_copy, output_layer

    @staticmethod
    def cost(outputs, expected):
        # Not used at all, but just to show what the cost function is
        cost = 0
        for index, output in enumerate(outputs):
            cost += (output - expected[index]) ** 2
        return cost

    @staticmethod
    def activate(value):
        # Sigmoid function for activation
        return 1 / (1 + np.exp(-1 * value))

    @staticmethod
    def activate_prime(value_of_sigmoid):
        # Derivative of sigmoid depends purely on the output of the sigmoid, so we can just input the activated values
        return np.multiply(value_of_sigmoid, (1 - value_of_sigmoid))

    def get_weights(self):
        return self.transform_in_hid, self.transform_hid_out

    def set_weights(self, in_hid, hid_out):
        self.transform_in_hid = in_hid
        self.transform_hid_out = hid_out