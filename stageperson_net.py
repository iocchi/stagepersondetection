# -*- coding: utf-8 -*-
"""
    StagePerson_Net: model for detecting people from camera images in Stage simulator

    Luca Iocchi 2021 - iocchi@diag.uniroma1.it

"""

import os, sys, time, socket
import threading
import argparse

import numpy as np

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten,\
                         Conv2D, MaxPooling2D, Input
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
height = 120
width = 160

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
        target_size=(height, width),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True)

    test_datagen = ImageDataGenerator(
        rescale = 1. / 255)

    test_generator = test_datagen.flow_from_directory(
        directory=testset,
        target_size=(height, width),
        color_mode="rgb",
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    num_samples = train_generator.n
    num_classes = train_generator.num_classes
    input_shape = (height, width, 3)


    classnames = [k for k,v in train_generator.class_indices.items()]

    print("Image input %s" %str(input_shape))
    print("Classes: %r" %classnames)

    print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
    print('Loaded %d test samples from %d classes.' %(test_generator.n,test_generator.num_classes))

    return input_shape, num_classes




"""

StagePersonNet model

"""

def StagePersonNet_v1(input_shape, num_classes, regl2 = 0.001, lr=0.001):

    from keras.layers.experimental.preprocessing import Resizing,Rescaling

    model = Sequential()

    model.add(Input(shape=(120,160,3)))
    model.add(Resizing(height, width, interpolation="bilinear", name=None))
    model.add(Rescaling(scale=1./255.))

    # C1 Convolutional Layer 
    model.add(Conv2D(filters=96, kernel_size=(11,11),\
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

    # D1 Dense Layer
    model.add(Dense(1000, kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(100,kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile

    adam = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.summary()

    return model

"""
  Simple model (868 k parameters)
"""
def StagePersonNet(input_shape, num_classes, regl2 = 0.001, lr=0.001):

    model = Sequential()

    model.add(Input(shape=input_shape))

    # C1 Convolutional Layer 
    model.add(Conv2D(filters=96, kernel_size=(9,9),\
                     strides=(3,4), padding='same'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(BatchNormalization())

    # C2 Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(5,5), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    # Pooling
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # C4 Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))

    # Flatten
    model.add(Flatten())
   
    # D1 Dense Layer
    model.add(Dense(256, kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # D2 Dense Layer
    model.add(Dense(100,kernel_regularizer=regularizers.l2(regl2)))
    model.add(Activation('relu'))
    # Batch Normalisation
    model.add(BatchNormalization())

    # Output Layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # Compile

    adam = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.summary()

    return model



"""
Load a trained model
"""


def loadModel(modelname):
    try:
        model = load_model(modelname)
        print("\nModel loaded successfully from file %s\n" %modelname)
    except OSError as e:
        print(e)
        print("\n!!! Model %s not loaded !!!\n" %modelname)
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
    model.save(modelname)
    print("\nModel saved successfully on file %s\n" %modelname)


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
        target_size=(height, width),
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

    adam = optimizers.Adam(lr=5e-4)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    trainModel(model,epochs=30)

    for lr in [3e-4, 1e-4, 1e-5]:
        adam = optimizers.Adam(lr=lr)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        trainModel(model,epochs=10)

    saveModel(model, modelname)


def moreTrain(modelname, lr=0.0001):

    input_shape, num_classes = loadData()

    model = loadModel(modelname)

    adam = optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    trainModel(model,epochs=10)

    saveModel(model, modelname)



"""
Load an image and return input data for the network
"""
def inputImage(imagefile):
    try:
        img = load_img(imagefile, color_mode="rgb", target_size=(height, width))
        arr = img_to_array(img) / 255
        inp = np.array([arr])  # Convert single image to a batch.
        return inp
    except:
        return None

"""
Predict class of an image
"""
def predictImage(model, imagefile):
    global classnames
    inp = inputImage(imagefile)
    if inp is not None:
        pr = model.predict(inp)
        return (np.max(pr), classnames[np.argmax(pr)])
    else:
        return (0, 'error')


class ModelServer(threading.Thread):

    def __init__(self, port, model):
        threading.Thread.__init__(self)

        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.settimeout(3) # timeout when listening (exit with CTRL+C)
        # Bind the socket to the port
        server_address = ('', port)
        self.sock.bind(server_address)
        self.sock.listen(1)

        self.model = model

        print("Model Server running on port %d" %port)

        self.dorun = True # server running
        self.connection = None  # connection object

    def stop(self):
        self.dorun = False

    def connect(self):
        connected = False
        while (self.dorun and not connected):
            try:
                # print 'Waiting for a connection ...'
                # Wait for a connection
                self.connection, client_address = self.sock.accept()
                self.connection.settimeout(3)
                connected = True
                print('Connection from %s' %str(client_address))
            except:
                pass #print("Listen again ...")

    # buf may contain a first chunk of data
    def recvall(self, count, chunk):
        buf = chunk
        count -= len(buf)
        while count>0:
            newbuf = self.connection.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf


    def run(self):

        imgsize = -1
        res = 'none 0.0'
        while (self.dorun):
            self.connect()  # wait for connection
            try:
                # Receive data
                while (self.dorun):
                    try:
                        data = self.connection.recv(256)
                        data = data.strip()
                    except socket.timeout:
                        data = "***"
                    except Exception as e:
                        print(e)
                        data = None

                    buf = b''
                    if (type(data)!=str):
                        k = data.find(b'\n')
                        if (k<0):
                            data = data.decode('utf-8')
                        elif (len(data)>k+1):
                            buf = data[k+2:]
                            data = data[0:k].decode('utf-8')


                    if (data!=None and data!="" and data!="***"):
                        self.received = data
                        print('Received: %s' %data)
                        v = data.split(' ')
                        if v[0]=='REQ':
                            self.connection.send('ACK\n\r'.encode('UTF-8'))
                        elif v[0]=='GETRESULT':
                            ressend = (res+'\n\r').encode('UTF-8')
                            self.connection.send(ressend)
                        elif v[0]=='EVAL' and len(v)>1:
                            print('Eval image [%s]' %v[1])
                            (p,c) = predictImage(self.model,v[1])
                            print("Predicted: %s, prob: %.3f" %(c,p))
                            res = "%s %.3f" %(c,p)
                            ressend = (res+'\n\r').encode('UTF-8')
                            self.connection.send(ressend)
                        elif v[0]=='RGB' and len(v)>=3:
                            imgwidth = int(v[1])
                            imgheight = int(v[2])
                            imgsize = imgwidth*imgheight*3
                            print("RGB image size: %d" %imgsize)
                            buf = self.recvall(imgsize, buf)
                            if buf is not None:
                                print("Image received size: %d " %(len(buf)))
                                a = np.fromstring(buf, dtype='uint8')
                                a = a.reshape((imgheight,imgwidth,3))
                                a = a / 255.0
                                inp = np.array([a])
                                pr = model.predict(inp)
                                (p,c) = (np.max(pr), classnames[np.argmax(pr)])
                                print("Predicted: %s, prob: %.3f" %(c,p))
                                res = "%s %.3f" %(c,p)
                                ressend = (res+'\n\r').encode('UTF-8')
                                self.connection.send(ressend)
                        else:
                            print('Received: %s' %data)

                    elif (data == None or data==""):
                        break
            finally:
                print('Connection closed.')
                # Clean up the connection
                if (self.connection != None):
                    self.connection.close()
                    self.connection = None

    # wait for Keyboard interrupt
    def spin(self):
        while (self.dorun):
            try:
                time.sleep(120)
            except KeyboardInterrupt:
                print("Exit")
                self.dorun = False


"""
Start prediction server
"""
def startServer(port, model):
    print("Starting stagepersondetection server on port %d" %port)
    print("Send string message 'EVAL <imagefile>'")
    print("Send RGB image 'RGB <width> <height>' followed by data buffer")
    mserver = ModelServer(port, model)
    mserver.start()
    mserver.spin() 
    mserver.stop()




def clean(modelname):
    global modelsdir
    r = modelname
    if r[0:6]!=modelsdir:
      r = modelsdir + '/' + r
    if r[-3:]!='.h5':
      r = r + '.h5'
    return r


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-modelname", type=str, default=None,
                        help="Model name to load/save")
    parser.add_argument("--train", default = False, action ='store_true',
                        help="Train and save the model")
    parser.add_argument("--moretrain", default = False, action ='store_true',
                        help="Keep training")
    parser.add_argument("--test", default = False, action ='store_true',
                        help="Test the model")
    parser.add_argument("-predict", type=str, default=None,
                        help="Image file to predict")
    parser.add_argument('--server', default = False, action ='store_true', 
                        help='Start in server mode')
    parser.add_argument('-server_port', type=int, default=default_server_port, 
                        help='server port (default: %d)' %default_server_port)

    args = parser.parse_args()

    if (args.modelname == None):
        print("Please specify a model name and an operation to perform.")
        sys.exit(1)

    modelname = clean(args.modelname)

    if (args.train):
        doTrain(modelname)
    elif (args.test):
        model = loadModel(modelname)
        evalModel(model)
    elif (args.moretrain):
        moreTrain(modelname)
    elif (args.predict != None):
        model = loadModel(modelname)
        (p,c) = predictImage(model,args.predict)
        print("Predicted: %s, prob: %.3f" %(c,p))
    elif (args.server):
        model = loadModel(modelname)
        startServer(args.server_port, model)
    else:
        print("Please specify a model name and an operation to perform.")
        sys.exit(1)


