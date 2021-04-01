from network import Network
import numpy as np

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

image_network = Network(2, 3)
output, hidden = image_network.predict(np.transpose(np.matrix([5, 7])))

print(output)
print(hidden)
# all image inputs must be arrays with each elemt represeting % in a single pixel (grayscale)
