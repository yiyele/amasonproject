# -*- coding: utf-8 -*-
'''
Created on 2017年6月22日

@author: USER
'''
from collections import OrderedDict
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten,Activation,Input,GlobalAveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D,ZeroPadding2D
from tensorflow.contrib.keras.api.keras.layers import concatenate 
from keras.models import Model,Sequential
from keras.layers import merge
from keras import backend as K


__all__ = ['AlexNet', 'alexnet']

class AlexNet:
    def __init__(self, num_classes=1000):
        self.num_classes = num_classes
    def feature(self,x):
        x = ZeroPadding2D(padding=(2, 2))(x)                               #自己根据下面的conv2d中的padding为2来判断，这里补padding=2
        x = Conv2D(64, kernel_size=(11,11), strides=(4,4), padding='valid')(x)    #nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        x = Activation('relu')(x)                                           #nn.ReLU(inplace=True),
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)                 #nn.MaxPool2d(kernel_size=3, stride=2),
        x = Conv2D(192, kernel_size=(5,5), padding='same')(x)               #nn.Conv2d(64, 192, kernel_size=5, padding=2),
        x = Activation('relu')(x)                                           #nn.ReLU(inplace=True),
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)               #nn.MaxPool2d(kernel_size=3, stride=2),
        x = Conv2D(384, kernel_size=(3,3), padding='same')(x)               #nn.Conv2d(192, 384, kernel_size=3, padding=1),
        x = Activation('relu')(x)                                           #nn.ReLU(inplace=True),
        x = Conv2D(256, kernel_size=(3,3), padding='same')(x)               #nn.Conv2d(384, 256, kernel_size=3, padding=1),
        x = Activation('relu')(x)#nn.ReLU(inplace=True),
        x = Conv2D(256, kernel_size=(3,3), padding='same')(x)             #nn.Conv2d(256, 256, kernel_size=3, padding=1),
        x = Activation('relu')(x)#nn.ReLU(inplace=True),
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)#             nn.MaxPool2d(kernel_size=3, stride=2),
        return x
    
    def classifier(self,x):
        x = Dropout(0.5)(x)#             nn.Dropout(),
        x = Dense(4096)(x)#nn.Linear(256 * 6 * 6, 4096),
        x = Activation('relu')(x)#nn.ReLU(inplace=True),
        x = Dropout(0.5)(x)#nn.Dropout(),
        x = Dense(4096)(x)#nn.Linear(4096, 4096),
        x = Activation('relu')(x)#nn.ReLU(inplace=True),
        x = Dense(self.num_classes)(x)#nn.Linear(4096, num_classes),
        return x    

    def forward(self, x):
        x = self.feature(x)
        x = Flatten()(x)#x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

def alexnet(x,**kwargs):
    model = AlexNet(**kwargs)
    x = model.forward(x)
    return x
