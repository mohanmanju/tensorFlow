import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import *
from keras.callbacks import TensorBoard
import numpy as np
import math
import os
from time import time


dim = 28*28
y_train = np.zeros(10)



class mnist:

    def __init__(self):
        self.x_train = []
        self.y_train = []
        #self.y_train.append([0,0,0,0,0,0,0,0,0,0])

    def build_model(self):

        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3),strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(Conv2D(64, (3, 3),strides=(1, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(64, (3, 3),strides=(1, 1), activation='relu'))
        self.model.add(Conv2D(64, (3, 3),strides=(1, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
        self.model.add(Dropout(0.25))

        self.model.add(Conv2D(64, (3, 3),strides=(1, 1), activation='relu'))
        #self.model.add(Conv2D(64, (3, 3),strides=(1, 1), activation='relu'))
        #self.model.add(MaxPooling2D(pool_size=(2, 2),padding='valid'))
        self.model.add(Dropout(0.25))


        self.model.add(Flatten())
        self.model.add(Dense(1000, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(10, activation='linear'))


    def compile_model(self):

        self.model.compile(loss='mse',optimizer='adam',metrics=['acc'])


    def train(self):
        tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        self.model.fit(self.x_train,self.y_train,epochs=10,callbacks=[tensorboard])


    def read_data(self):
        count = 0
        with open('../mnist/training') as data:
            for img in data:
                targetClass,inputData = img.split(".")
                targetClass = int(targetClass)
                self.y_train.append([0,0,0,0,0,0,0,0,0,0])
                self.y_train[-1][targetClass] = 1



                inputData = [int(dat) for dat in inputData.split(",")]
                count = 0
                temprow = []
                dat = []
                for i,x in enumerate(inputData):
                    temp = [x]
                    temprow.append(temp)
                    if((i+1) % 28==0):
                        dat.append(temprow)
                        temprow = []
                self.x_train.append(dat)
                #self.y_train = np.asarray(self.y_train)

                #self.y_train = np.asarray(list(self.y_train))
                #print(self.y_train)
                #break
        self.train()

        y_test = []
        x_test = []
        with open('../mnist/testing') as data:
            for img in data:
                targetClass,inputData = img.split(".")
                targetClass = int(targetClass)
                y_test.append([0,0,0,0,0,0,0,0,0,0])
                y_test[-1][targetClass] = 1



                inputData = [int(dat) for dat in inputData.split(",")]
                count = 0
                temprow = []
                dat = []
                for i,x in enumerate(inputData):
                    temp = [x]
                    temprow.append(temp)
                    if((i+1) % 28==0):
                        dat.append(temprow)
                        temprow = []

                x_test.append(dat)
                #self.y_train = np.asarray(self.y_train)

                #self.y_train = np.asarray(list(self.y_train))
                #print(self.y_train)
                #break
        self.test(x_test,y_test)


    def test(self,x_test,y_test):
        score = self.model.evaluate(x_test, y_test)

        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))




digits = mnist()
digits.build_model()
digits.compile_model()
digits.read_data()

digits.model.save('model.h5')