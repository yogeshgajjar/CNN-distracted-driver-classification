import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
# from skimage.feature import hog
# from skimage import exposure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import sys 
# from keras.utils import to_categorical

# working_directory = '/home/yogesh/Git/CNN-distracted-driver-classification-'
# list_image = []
# for image in glob.glob(working_directory + '/imgs/train/c1/*.jpg'):
#     img = cv2.imread(image) 
#     list_image.append(img) 

# new = np.array(list_image)

# print(new.shape) 

def readImages(directory):
    trainImages, trainLabels, testImages = [], [], []
    for i in range(10):
        for image in glob.glob(directory + '/imgs/train/c'+str(i)+'/*.jpg'):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
            resized_img = cv2.resize(img, (64,64))
            trainImages.append(resized_img) 
            trainLabels.append(i)

    count = 0
    for image in glob.glob(directory + '/imgs/test/*.jpg'):
        if count != 500:
            img = cv2.imread(image ,cv2.IMREAD_GRAYSCALE) 
            resized_img = cv2.resize(img, (64,64))
            testImages.append(resized_img)
            count += 1
        else:
            exit 
        
    return np.array(trainImages), np.array(trainLabels), np.array(testImages)

def testTrainSplit(feature, label):
    X_tr, X_te, y_tr, y_te = train_test_split(feature, label, train_size = 0.8, shuffle = True) 

    return X_tr, X_te, y_tr, y_te

def main():
    working_directory = '/home/yogesh/Git/CNN-distracted-driver-classification-'

    trainImages, trainLabels, testImages = readImages(working_directory)
    # np.save('c5_to_c9.npy', train_images_1, allow_pickle=True) 
    print("train images: ", trainImages.shape)
    print("train labels: ", len(trainLabels))
    print("test images: ", testImages.shape)

    X_train, X_test, y_train, y_test = testTrainSplit(trainImages, trainLabels)
    print("X_train: ", X_train.shape)
    print("X_test: ", X_test.shape)
    print("y_train: ", y_train.shape)
    print("y_test: ", y_test.shape)
    np.save('X_train.npy', X_train, allow_pickle=True)
    np.save('X_test.npy', X_test, allow_pickle=True)
    np.save('y_train.npy', y_train, allow_pickle=True)
    np.save('y_test.npy', y_test, allow_pickle=True)
    np.save('testImages.npy', testImages, allow_pickle=True)


if __name__ == "__main__":
    main()

