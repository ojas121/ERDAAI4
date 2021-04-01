import os
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

path = "C:\\Users\\ojasm\\PycharmProjects\\ERDAAI4\\traindata"

#Load the images as paths
def loadImages(path):
    files = os.listdir(path)

    for i in range(len(files)):
        f = os.path.join(path, files[i])
        files[i] = f

    return files

#Display one image
def display_one(a, title1 = "original"):
    plt.imshow(a)
    plt.title(title1)
    plt.xticks([])
    plt.yticks([])
    plt.show()

#display 2 images
def display(a, b, title1 = "original", title2 = "Edited"):
    plt.subplot(121)
    plt.imshow(a)
    plt.title(title1)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(122)
    plt.imshow(b)
    plt.title(title2)
    plt.xticks([])
    plt.yticks([])

    plt.show()

def showHist(img):
    flat_list = []
    for i in img:
        for j in range(len(i)):
            for k in range(len(i[j])):
                flat_list.append(i[j][k])

    hist, bins = np.histogram(flat_list, 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    plt.plot(cdf_normalized, color = 'b')
    plt.hist(flat_list,256,[0,256], color = 'r')
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()







#IMPORTANT PART
#preprocessing
def processing(data):
    scale = 2
    imgind = 3
    


    #LOADING IMAGE
    #loading img
    img = []
    for i in data:
        simg = cv2.imread(i,cv2.IMREAD_UNCHANGED)
        simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
        img.append(simg)
    
    print("original size", img[imgind].shape)

    orSize = img[imgind].shape



    #RESIZE IMAGE, LATER USED TO CREATE AN IMAGE OF THE SAME SIZE TO BE USED AS IMPUT IN THE NEURAL NETWORK
    #resize
    height = int(orSize[0] / scale)
    width = int(orSize[1] / scale)

    newsize = (height, width)

    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i],dim,interpolation=cv2.INTER_LINEAR)
        res_img.append(res)
    
    print("resized", res_img[imgind].shape)
    original = res_img[imgind]
    #display_one(original)



    #REMOVE THE NOISE IN THE PICTURE BY BLURRING IT, NOT USED ATM
    no_noise = []
    for i in range(len(res_img)):
        blur = cv2.GaussianBlur(res_img[i], (5,5),0)
        no_noise.append(blur)
    
    #res_img = no_noise
    


    #EQUALIZE THE HISTOGRAM TO INCREASE THE CONTRAST OF THE COLORS
    showHist(res_img[imgind])
    img_yuv = cv2.cvtColor(res_img[imgind], cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    res_img[imgind] = img_output
    #showHist(img_output)
    #display_one(img_output)
    
    
    
    #PEFFORM AN EDGE DETECTION ALGORITHM, IN OPENCV IT IS CALLED CANNY. TO DO THIS, THE IMAGE MUST BE CONVERTED TO GRAYSCALE
    #GRAYSCALE
    gray = cv2.cvtColor(res_img[imgind], cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 9)

    #EDGE DETECTION
    edges = cv2.Canny(gray, 100, 250)
    
    #THIS IS THE PART THAT GENERATES THE PICTURE ON THE LEFT, WE WILL PROBABLY NOT USE IT.
    ret, thresh = cv2.threshold(gray, 200,300,0)
    contours, hierarchie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contourimg = cv2.drawContours(res_img[imgind], contours, -1, (0,255,0),3)
    
    #DISPLAY THE 2 IMAGES GENERATED, THE RIGHT ONE WILL BE USED FOR THE AIRCRAFT TYPE NETWORK
    display(contourimg, edges)




    #THE PART OF CODE THAT CUTS THE AIRPLANE OUT OF THE SKY. THE GRABCUT FUNCTION FROM OPENCV IS USED. IT STILL MUST BE PROGRAMMED THE DETERMINE THE RECT COORDINATES AUTOMATICALLY.
    mask = np.zeros(newsize,np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    #The 3 rectangles I found for the 3 test images I had, still need to make it automatic.
    
    rect = (int(0 / scale),int(75 / scale),int(480 / scale),int(215 / scale))                  #For index 2
    #rect = (int(400 / scale),int(390 / scale),int(1200 / scale),int(450 / scale))              #for index 1
    #rect = (int(30 / scale),int(240 / scale),int(1120 / scale),int(335 / scale))               #for index 0
    cv2.grabCut(res_img[imgind],mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    newImg = res_img[imgind] * mask2[:,:,np.newaxis]


    #DISPLAY THE CUT OUT AIRCRAFT. THE COLORS MUST BE REPLACED BY THEIR ORIGINAL COLORS, THEN THIS IMG CAN BE USED FOR THE CARRIER RECOGNITION
    display_one(newImg)







    #STUFF THAT DOENT MATTER ANYMORE


    
    #display(original,image, "original", "blurred")

 #  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
 #  ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

 #  #display(original, thresh, "original", "segmented")

 #  kernel = np.ones((3,3), np.uint8)
 #  opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
 #  sure_bg = cv2.dilate(opening,kernel, iterations=3)
 #  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2, 5)
 #  ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(),255,0)
 #  sure_fg = np.uint8(sure_fg)
 #  unknown = cv2.subtract(sure_bg,sure_fg)

 #  #display(original, sure_bg, "original", "segmented background")

 #  ret, markers = cv2.connectedComponents(sure_fg)

 #  markers = markers + 1
 #  markers[unknown==255] = 0
 #  markers = cv2.watershed(image, markers)
 #  image[markers == -1] = [255,0,0]

 #  display(image, markers, "original", "marked")



traindata = loadImages(path)
processing(traindata)