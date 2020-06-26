import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
import pickle
import os

###############################################################################
##############################Setting##########################################
threshold = 0.65
###############################################################################
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

#####IMPORT MODEL##############################################################
pickle_in = open("../Sudoku_project/Model/model_trained.p","rb")
model = pickle.load(pickle_in)


image = cv2.imread('../Sudoku_project/Model/2.jpg')
image = cv2.resize(image,(32,32))
image = preProcessing(image)
image = image.reshape(1,32,32,1)
classIndex = int(model.predict_classes(image))
print("Class index is",classIndex)
# cv2.imshow('Image',image)
# cv2.waitKey(0)