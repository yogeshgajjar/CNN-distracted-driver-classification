import keras
import cv2
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from matplotlib import pyplot as plt
from keras.callbacks import CSVLogger
from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import math 
from PIL import Image
from keras.models import model_from_json
import os
import time 
import sys 

def loadNumpyfile(root_path):
    X_train = np.load(root_path+'X_train.npy', allow_pickle=True)
    X_train = X_train.reshape(X_train.shape[0], 64,64,1)
    X_test = np.load(root_path+'X_test.npy', allow_pickle=True)
    X_test = X_test.reshape(X_test.shape[0],64,64,1)
    y_train = np.load(root_path+'y_train.npy', allow_pickle=True)
    y_test = np.load(root_path+'y_test.npy', allow_pickle=True)
    test_images = np.load(root_path+'testImages.npy', allow_pickle=True)

    return X_train, X_test, y_train, y_test, test_images

def preProcessing(totalLabels, X_train, X_test, y_train, y_test, test_images):
    y_train = keras.utils.to_categorical(y_train, totalLabels)
    y_test = keras.utils.to_categorical(y_test, totalLabels)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    test_images = test_images.astype('float32')
    X_train /= 255
    X_test /= 255
    test_images /= 255

    labels = {'c0': 'Safe driving', 
                'c1': 'Texting - right', 
                'c2': 'Talking on the phone - right', 
                'c3': 'Texting - left', 
                'c4': 'Talking on the phone - left', 
                'c5': 'Operating the radio', 
                'c6': 'Drinking', 
                'c7': 'Reaching behind', 
                'c8': 'Hair and makeup', 
                'c9': 'Talking to passenger'}

    return X_train, X_test, y_train, y_test, test_images, labels

def myCNNArchitecture(weights=None):
    mycnn = Sequential()

    mycnn.add(Conv2D(32, (3, 3), activation='relu', padding = 'same', input_shape=(64,64,1)))
    mycnn.add(BatchNormalization())
    mycnn.add(Conv2D(32, (3,3), activation='relu', padding='same'))
    mycnn.add(BatchNormalization())
    mycnn.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    mycnn.add(Dropout(0.5))

    mycnn.add(Conv2D(64, (3, 3), activation='relu', padding = 'same'))
    mycnn.add(BatchNormalization())
    mycnn.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    mycnn.add(BatchNormalization())
    mycnn.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    mycnn.add(Dropout(0.5))

    mycnn.add(Conv2D(128, (3, 3), activation='relu', padding = 'same'))
    mycnn.add(BatchNormalization())
    mycnn.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    mycnn.add(BatchNormalization())
    mycnn.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    mycnn.add(Dropout(0.5))

    mycnn.add(Flatten())
    mycnn.add(Dense(512,activation='relu'))
    mycnn.add(BatchNormalization())
    mycnn.add(Dropout(0.5))
    mycnn.add(Dense(128,activation='relu'))
    mycnn.add(BatchNormalization())
    mycnn.add(Dropout(0.25))
    mycnn.add(Dense(10,activation='softmax'))

    return mycnn

def modelDetails(totalEpochs, batchSize, weightDecay, learningRate, key):
    """
    Provides the summary of the hyper-parameters used for training. 
    """
    print("******* DETAILS OF THE MODEL *********\n")
    print("** Hyper - parameters detail **\n")
    print("Epochs : ", totalEpochs)
    print("Batch Size: ", batchSize)
    print("Weight Decay: ", weightDecay)
    print("Learning rate: ", learningRate)
    print("Optimizer: ", key,"\n")


def saveModel(model):
    '''
    Saves the model after training. 
    '''

    newmodel = model.to_json()
    with open("myModel.json", "w") as json_file:
        json_file.write(newmodel)

    model.save_weights("myModel.h5")
    print("Saved model to disk")


def loadModel(json_file='myModel.json', h5_model='myModel.h5'):
    """
    Loads the model 
    """

    json_file = open(json_file, 'r')
    myloaded_model = json_file.read()
    json_file.close()
    finalModel = model_from_json(myloaded_model)
    finalModel.load_weights(h5_model)
    print("Loaded model from disk")
    
    return finalModel

def train(model, learningRate, weightDecay, X_train, y_train, X_test, y_test, totalEpochs, batchSize):

    optim = SGD(lr=learningRate, decay=weightDecay , momentum=0.9, nesterov=True)
    model.compile(optimizer=optim,loss='categorical_crossentropy', metrics=['accuracy'])

    tf.keras.callbacks.CSVLogger("log.csv", separator=',', append=False)
    csv_logger = CSVLogger('training.csv')

    mycnnHistory = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=totalEpochs, batch_size=batchSize, verbose = 1, callbacks=[csv_logger])

    return mycnnHistory, optim


def test(final, Xte, yte, batchSize, optim):
    """
    Test on the test data and displays the final accuracy 
    """

    final.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
    score = final.evaluate(Xte, yte, verbose=0)
    # print("%s: %.2f%%" % (final.metrics_names[1], score[1]*100))

    score, acc = final.evaluate(Xte, yte, batch_size=batchSize)

    print("******** FINAL TEST ACCURACY **********")
    print('Test score:', score)
    print('Test accuracy: ', acc*100, '%')

def plotTestClass(model, test_image, image_number, label, bs, color_type=1):

    img = cv2.resize(test_image[image_number],(64,64))
    plt.imshow(img, cmap='gray')

    img_final = img.reshape(-1,64,64,color_type)

    y_prediction = model.predict(img_final, batch_size=bs, verbose=1)
    print('Y prediction: {}'.format(y_prediction),'\n')
    print('Predicted: {}'.format(label.get('c{}'.format(np.argmax(y_prediction)))))
    
    plt.show()

def dataVisualize(hist, dropout, lr, key, wd, bs, xavier='None'):
    plt.figure(1, figsize=(12,8))
    plt.plot(hist.history['accuracy'], color='red',marker='o', linewidth=3)
    plt.plot(hist.history['val_accuracy'],color='blue', marker='x', linewidth=3)
    plt.title("Epoch Accuracy Plot \n Dropout: "+str(dropout)+"| Learning Rate: "+str(lr)+"| Optimizers: "+key+
                " | Weight Decay:"+str(wd)+"| Batch Size: "+str(bs)+"| Filter Weight(xavier_normal): "+str(xavier))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train Accuracy', 'Test Accuracy'], loc='upper left')
    plt.savefig("epoch-accuracy.png")
    plt.show()

    plt.figure(2, figsize=(12,8))
    plt.plot(hist.history['loss'], color='red', marker='o', linewidth=3)
    plt.plot(hist.history['val_loss'], color='blue', marker='x', linewidth=3)
    plt.title("Epoch Loss Plot \n Dropout: "+str(dropout)+"| Learning Rate: "+str(lr)+"| Optimizers: "+key+
                " | Weight Decay: "+str(wd)+"| Batch Size: "+str(bs)+"| Filter Weight(xavier_normal): "+str(xavier))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train loss', 'Test loss'], loc='upper right')
    plt.savefig("epoch-loss.png")
    plt.show()

    

def main():
    root_path = sys.argv[1]
    choice = sys.argv[2]

    totalLabels, batchSize, totalEpochs, learningRate, dropout, key = 10, 8, 1, 0.001, 0.5, "SGD"
    start_time = time.time()

    weightDecay = learningRate/totalEpochs

    X_train, X_test, y_train, y_test, test_images = loadNumpyfile(root_path)
    X_train, X_test, y_train, y_test, test_images, labels = preProcessing(totalLabels, X_train, X_test, y_train, y_test, test_images)

    optimizer = SGD(lr=learningRate, decay=weightDecay , momentum=0.9, nesterov=True)

    mycnn = myCNNArchitecture()

    if choice == 'train':
        modelDetails(totalEpochs, batchSize, weightDecay, learningRate, key)

        print("---- Training starts ----\n ")

        history, optimizer = train(mycnn, learningRate, weightDecay, X_train, y_train, X_test, y_test, totalEpochs, batchSize)

        print("---- Training ends ----\n")
        print("Average time taken by each epoch for training is: ", round(time.time() - start_time, 2)/totalEpochs, 's')
        print("Total time taken for training 50,000 images: ", (round(time.time() - start_time, 2))/60,'m\n')

        dataVisualize(history, dropout, learningRate, key, weightDecay, batchSize)
        saveModel(mycnn)
        final = loadModel()

    if choice == 'test':
        final = loadModel('myModel_load.json', 'myModel_load.h5')
        plotTestClass(final, test_images, 101, labels, batchSize)


    test(final, X_test, y_test, batchSize, optimizer)

if __name__ == "__main__":
    main()



    

