from network import Network
import numpy as np
import cv2

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# change to actual size
num_pixels = 39900
num_aircraft = 100

image_network = Network(num_pixels, num_aircraft, 0.7)
images = []  # IMAGES HERE! PLEASE IN THE SAME ORDER AS IN THE GOOGLE DOC!!
# all image inputs must be arrays with each element representing % in a single pixel (grayscale)

for i in range(1, num_aircraft + 1):
    name = str(i)
    if i < 10:
        name = "0" + name
    img = cv2.imread("pics_processed/" + name + ".jpg")
    pixels = []

    for j in range(img.shape[0]):  # traverses through height of the image
        for k in range(img.shape[1]):  # traverses through width of the image
            pixels.append(img[i][j][2])
    pixels = np.transpose(pixels)
    print(max(pixels))
    images.append(pixels)

aircraft = [0] * 10 + [1] * 7 + [2] * 9 + [3] * 6 + [4] * 4 + [5] + [4] + [6] + [4] + [2] + [4] * 2 + [7] + [5] * 2 + [
    6] + [8] * 2 + [9] * 3 + [10] * 3 + [11] * 2 + [12] * 3 + [13] + [14] * 3 + [8] * 2 + [15] * 5 + [16] * 3 + [
    17] * 3 + [18] * 2 + [19] * 2 + [20] * 4 + [21] * 3 + [22] * 3 + [23] * 4 + [24] * 4 + [25]  # actual aircraft

# we do 3 rounds of learning!
for index, image in enumerate(images):
    print("Picture index", index)
    hid, out = image_network.predict(image)
    expected_output = np.zeros(num_aircraft)
    expected_output[aircraft[index]] = 1
    image_network.learn(image, hid, out, expected_output)


in_hid, hid_out = image_network.get_weights()
np.savetxt("in_hid.txt", in_hid)
np.savetxt("hid_out.txt", hid_out)

planes = {
    '0': "B737",
    '1': "A320",
    '2': "B777",
    '3': "A380",
    '4': "B747",
    '5': "An-255",
    '6': "Airbus Beluga",
    '7': "B767",
    '8': "CRJ",
    '9': "E175",
    '10': "Cessna Skyhawk",
    '11': "Cessna Citation",
    '12': "Cessna Skymaster",
    '13': "Piper M600",
    '14': "A220",
    '15': "A319",
    '16': "Bell 407",
    '17': "Airbus H120",
    '18': "Embraer Phemom",
    '19': "HA420",
    '20': "DC10",
    '21': "Aquila A210",
    '22': "An-148",
    '23': "ATR 42",
    '24': "Dash-8",
    '25': "Beechcraft B200GT",
}
