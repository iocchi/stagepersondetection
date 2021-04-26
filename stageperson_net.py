# -*- coding: utf-8 -*-
"""
    StagePerson_Net: model for detecting people from camera images in Stage simulator

    Luca Iocchi 2021 - iocchi@diag.uniroma1.it

"""

import os, sys
import argparse

import numpy as np

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
                         Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras import optimizers
from keras.models import load_model

import matplotlib.pyplot as plt


print("Tensorflow version %s" %tf.__version__)
print("Keras version %s" %keras.__version__)


datadir = 'dataset'
modelsdir = 'models'

train_generator = None
test_generator = None
classnames = ['blue', 'green', 'none', 'red', 'yellow']

default_server_port = 9250

"""
Load data from folders
"""
def loadData():
    global train_generator, test_generator, classnames

    trainingset = datadir + '/train/'
    testset = datadir + '/test/'

    batch_size = 32

    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,\
        zoom_range=0.1,\
        #rotation_range=10,\
        width_shift_range=0.1,\
        height_shift_range=0.1,\
        horizontal_flip=True,\
        vertical_flip=False)

    train_generator = train_datagen.flow_from_directory(
        directory=trainingset,
        target_size=(118, 224),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True)

    test_datagen = ImageDataGenerator(
        rescale = 1. / 255)

    test_generator = test_datagen.flow_from_directory(
        directory=testset,
        target_size=(118, 224),
        color_mode="rgb",
        batch_size=batch_size//2,
        class_mode="categorical",
        shuffle=False
    )

    num_samples = train_generator.n
    num_classes = train_generator.num_classes
    input_shape = train_generator.image_shape

    classnames = [k for k,v in train_generator.class_indices.items()]

    print("Image input %s" %str(input_shape))
    print("Classes: %r" %classnames)

    print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
    print('Loaded %d test samples from %d classes.' %(test_generator.n,test_generator.num_classes))

    return input_shape, num_classes 
    



"""

StagePersonNet model

"""

def StagePersonNet(input_shape, num_classes, regl2 = 0.001, lr=0.001):

    model = Sequential()

    # C1 Convolutional Layer 
    model.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11),\
                     strides=(2,4), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Flatten
    model.add(Flatten())

    flatten_shape = (input_shape[0]*input_shape[1]*input_shape[2],)
    
    # D1 Dense Layer
    model.add(Dense(1000, kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    #model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(100,kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Dropout
    #model.add(Dropout(0.4))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile

    adam = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model


"""
Load a trained model
"""


def loadModel(modelname):
    global modelsdir
    filename = os.path.join(modelsdir, '%s.h5' %modelname)
    try:
        model = load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:    
        print("\nModel file %s not found!!!\n" %modelname)
        model = None
    return model


"""
Training
"""

def trainModel(model, epochs = 30):

    global train_generator, test_generator

    steps_per_epoch=train_generator.n//train_generator.batch_size+1
    val_steps=test_generator.n//test_generator.batch_size+1
    try:
        history = model.fit(train_generator, epochs=epochs, verbose=1,\
                        steps_per_epoch=steps_per_epoch,\
                        validation_data=test_generator,\
                        validation_steps=val_steps)
    except KeyboardInterrupt:
        pass


"""
Save the model
"""

def saveModel(model, modelname):
    global modelsdir
    models_dir = datadir + '/models/'
    filename = os.path.join(modelsdir, '%s.h5' %modelname)
    model.save(filename)
    print("\nModel saved successfully on file %s\n" %filename)


"""
Evaluate the model (accuracy on test set)
"""

def evalModel(model):

    global train_generator, test_generator

    testset = datadir + '/test/'

    test_datagen = ImageDataGenerator(
        rescale = 1. / 255)

    test_generator = test_datagen.flow_from_directory(
        directory=testset,
        target_size=(118, 224),
        color_mode="rgb",
        batch_size=1,
        class_mode="categorical",
        shuffle=False
    )

    print('Loaded %d test samples from %d classes.' %(test_generator.n,test_generator.num_classes))

    val_steps=test_generator.n
    loss, acc = model.evaluate(test_generator,verbose=1,steps=val_steps)
    print('Test loss: %f' %loss)
    print('Test accuracy: %f' %acc)


"""
Full train procedure
"""
def doTrain(modelname):

    input_shape, num_classes = loadData()

    model = StagePersonNet(input_shape,num_classes)

    adam = optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    trainModel(model,epochs=30)

    for lr in [0.0005, 0.0004, 0.0003, 0.0002, 0.0001]:
        adam = optimizers.Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        trainModel(model,epochs=10)

    saveModel(model, modelname)



"""
Load an image and return input data for the network
"""
def inputImage(imagefile):
    img = load_img(imagefile, target_size=(118, 224), color_mode="rgb")
    arr = img_to_array(img) / 255
    inp = np.array([arr])  # Convert single image to a batch.
    return inp

"""
Predict class of an image
"""
def predictImage(model, imagefile):
    global classnames
    inp = inputImage(imagefile)
    pr = model.predict(inp)
    return (np.max(pr), classnames[np.argmax(pr)])


def doTest(modelname):
    model = loadModel(modelname)
    evalModel(model)


def startServer(port):
    print("Starting stagepersondetection server on port %d", port)
    # TODO

def testimages():

    p1 = inputImage('test/none/20210425-170843-photo.jpg')
    p2 = inputImage('test/yellow/20210425-171333-photo.jpg')
    p3 = inputImage('test/blue/20210425-170928-photo.jpg')
    p4 = inputImage('test/green/20210425-171007-photo.jpg')
    p5 = inputImage('test/red/20210425-220733-photo.jpg')

    c1 = model.predict(p1)
    c2 = model.predict(p2)
    c3 = model.predict(p3)
    c4 = model.predict(p4)
    c5 = model.predict(p5)

    print("none:   %.3f  %s" %(np.max(c1), classnames[np.argmax(c1)]))
    print("yellow: %.3f  %s" %(np.max(c2), classnames[np.argmax(c2)]))
    print("blue:   %.3f  %s" %(np.max(c3), classnames[np.argmax(c3)]))
    print("green:  %.3f  %s" %(np.max(c4), classnames[np.argmax(c4)]))
    print("red:    %.3f  %s" %(np.max(c5), classnames[np.argmax(c5)]))


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-train", type=str, default=None,
                        help="Train and save model")
    parser.add_argument("-test", type=str, default=None,
                        help="Test saved model")
    parser.add_argument("-predict", type=str, default=None,
                        help="Image file to predict")
    parser.add_argument('--server', default = False, action ='store_true', 
                        help='Start in server mode')
    parser.add_argument('-server_port', type=int, default=default_server_port, 
                        help='server port')

    args = parser.parse_args()

    if (args.train != None):
        doTrain(args.train)
    elif (args.test != None):
        doTest(args.test)
    elif (args.predict != None):
        model = loadModel('stageperson5_v3')
        (p,c) = predictImage(model,args.predict)
        print("Predicted: %s, prob: %.3f" %(c,p))
    elif (args.server):
        startServer(args.server_port)
    else:
        print("No action specified. Use '-h' flag to see options available.")

# python stageperson_net -test stageperson5_v3

