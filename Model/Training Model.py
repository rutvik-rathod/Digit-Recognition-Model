import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
import os
# pip
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.optimizers import Adam
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle


################################################################
######################### Setting ##############################
path = '../Sudoku_project/Data Set'
testRatio = 0.15
valRatio = 0.1
imageDimension = (32,32)

batchSizeVal = 32
epochsVal = 10
stepsPerEpoch = 2000
################################################################
################################################################

images=[]
classNo=[]

mylist = os.listdir(path)
print(len(mylist))
noOfClasses = len(mylist)

print('Loading the Data Set.....')
for x in range (0,noOfClasses):
    myPiclist = os.listdir(path+"/"+str(x))
    for y in myPiclist:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(imageDimension[0],imageDimension[1]))
        images.append(curImg)
        classNo.append(x)
    print(x,end = " ")
print(" ")
print("Number of Classes Detected =",len(classNo))

images = np.asarray(images)
classNo = np.asarray(classNo)

print(images.shape)
print(classNo.shape)


X_train,X_test,y_train,y_test = train_test_split(images,classNo,test_size = testRatio)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size = valRatio)
print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

# print(len(np.where(y_train == 0)[0]))

numOfSamples = []
for x in range(0,noOfClasses):
    numOfSamples.append(len(np.where(y_test == x)[0]))

plt.figure(figsize = (10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title("Class ID")
plt.show()


def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[30])
# img = cv2.resize(img,(300,300))
# cv2.imshow("Processing",img)
# cv2.waitKey(0)

X_train = np.asarray(list(map(preProcessing,X_train)))
X_test = np.asarray(list(map(preProcessing,X_test)))
X_val = np.asarray(list(map(preProcessing,X_val)))


X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range = 0.1,
                             height_shift_range = 0.1,
                             zoom_range = 0.1,
                             shear_range = 0.1,
                             rotation_range=10)

dataGen.fit(X_train)

y_train = to_categorical(y_train,noOfClasses)
y_test = to_categorical(y_test,noOfClasses)
y_val = to_categorical(y_val,noOfClasses)


def myModel():
    noOfFilters = 50
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 500
    model = Sequential()

    model.add((Conv2D(noOfFilters,
                      sizeOfFilter1,
                      input_shape = (imageDimension[0],imageDimension[1],1),
                      activation = 'relu')))
    model.add((Conv2D(noOfFilters,sizeOfFilter1,activation = 'relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation = 'relu')))
    model.add((Conv2D(noOfFilters//2,sizeOfFilter2,activation = 'relu')))
    model.add(MaxPooling2D(pool_size = sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation = 'softmax'))
    
    
    model.compile( Adam(lr=0.001),
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
    
    return model

model = myModel()

print(model.summary())


history = model.fit_generator(dataGen.flow(X_train,y_train,batch_size=batchSizeVal),
                              steps_per_epoch = stepsPerEpoch,
                              epochs= epochsVal,
                              validation_data=(X_val,y_val),
                              shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend('Training','Validation')
plt.title('LOSS')
plt.show()

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend('Training','Validation')
plt.title('ACCURACY')
plt.show()

score = model.evaluate(X_test,y_test,verbose = 0)
print('Test Score =',score[0])
print('Test Accuracy =',score[1])


pickle_out = open("../Sudoku_project/Model/model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()