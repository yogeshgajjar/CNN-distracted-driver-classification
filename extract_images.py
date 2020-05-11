import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import sys 

def readImages(directory, num_class, count_test):
    trainImages, trainLabels, testImages = [], [], []
    for i in range(num_class):
        for image in glob.glob(directory + '/imgs/train/c'+str(i)+'/*.jpg'):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE) 
            resized_img = cv2.resize(img, (64,64))
            trainImages.append(resized_img) 
            trainLabels.append(i)

    count = 0
    for image in glob.glob(directory + '/imgs/test/*.jpg'):
        if count != count_test:
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
    working_directory = sys.argv[1]
    no_of_class, test_image_count = 10, 500
    trainImages, trainLabels, testImages = readImages(working_directory, no_of_class, test_image_count)

    X_train, X_test, y_train, y_test = testTrainSplit(trainImages, trainLabels)
    np.save('X_train.npy', X_train, allow_pickle=True)
    np.save('X_test.npy', X_test, allow_pickle=True)
    np.save('y_train.npy', y_train, allow_pickle=True)
    np.save('y_test.npy', y_test, allow_pickle=True)
    np.save('testImages.npy', testImages, allow_pickle=True)


if __name__ == "__main__":
    main()

