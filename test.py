import cv2
num_aircraft = 100
for i in range(1, num_aircraft + 1):
    name = str(i)
    if i < 10:
        name = "0" + name
    img = cv2.imread("pics_processed/" + name + ".jpg")
    pixels = []
    for j in range(img.shape[0]):  # traverses through height of the image
        for k in range(img.shape[1]):  # traverses through width of the image
            pixels.append(img[i][j])
    #print("Image", i, "has", len(pixels), "pixels")
    print(len(pixels), i)