# -*- coding: utf-8 -*-
'''
Created on 2017年6月20日
Funtion：构建一个squeeznet模型的keras库(有待测试)
@author: USER
'''
import math
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten,Activation,Input,GlobalAveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import concatenate
from keras.models import Model,Sequential
from keras import backend as K

__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']

class Fire:
    def __init__(self,x,squeeze_planes,expand1x1_planes, expand3x3_planes):
        self.squeeze = Conv2D(squeeze_planes,(1,1), padding='valid')          #self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = Activation('relu')                          #self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = Conv2D(expand1x1_planes,(1,1), padding='valid')      #self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,kernel_size=1)
        self.expand1x1_activation = Activation('relu')                        #self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = Conv2D(expand3x3_planes,(3,3), padding='same')       #self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,kernel_size=3, padding=1)
        self.expand3x3_activation = Activation('relu')                        #self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self,x):
        layer_list = []
        layer_list.append(self.squeeze)                                       #x = self.squeeze_activation(self.squeeze(x))
        layer_list.append(self.squeeze_activation)
        concatenate()                                                         #torch.cat([self.expand1x1_activation(self.expand1x1(x)),self.expand3x3_activation(self.expand3x3(x))], 1)
        return layer_list
    
class Fire1:
    def __init__(self,squeeze_planes,expand1x1_planes,expand3x3_planes):
        self.squeeze = Conv2D(squeeze_planes,(1,1), padding='valid')          #self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = Activation('relu')                          #self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = Conv2D(expand1x1_planes,(1,1), padding='valid')      #self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,kernel_size=1)
        self.expand1x1_activation = Activation('relu')                        #self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = Conv2D(expand3x3_planes,(3,3), padding='same')       #self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,kernel_size=3, padding=1)
        self.expand3x3_activation = Activation('relu')                        #self.expand3x3_activation = nn.ReLU(inplace=True)
    
    def forward(self,x):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        #layer_list = []
        x = self.squeeze_activation(self.squeeze(x))                          #x = self.squeeze_activation(self.squeeze(x))
        #layer_list.append(self.squeeze_activation)
                                                                 
        return concatenate([self.expand1x1_activation(self.expand1x1(x)),self.expand3x3_activation(self.expand3x3(x))],axis=channel_axis)
                                                                              #torch.cat([self.expand1x1_activation(self.expand1x1(x)),self.expand3x3_activation(self.expand3x3(x))], 1)

def Fire_modele(x,squeeze=16, expand=64):
   x = Conv2D(squeeze, (1, 1), padding='valid')(x)
   x = Activation('relu')(x)

   left = Conv2D(expand, (1, 1), padding='valid')(x)
   left = Activation('relu')(left)

   right = Conv2D(expand, (3, 3), padding='same')(x)
   right = Activation('relu')(right)

   x = concatenate([left, right], axis=3)
   return x

class SqueezeNet:
    def __init__(self,version=1.0,num_classes=1000):
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        self.version = version
        
    def features(self,x):
        if self.version == 1.0:
            x = Conv2D(96,(7,7), padding='valid',strides=(2, 2))(x)              #Conv2d(3, 96, kernel_size=7, stride=2),
            x = Activation('relu')(x)                                            #ReLU(inplace=True),
            x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)                   #MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            x = Fire1(16, 64, 64).forward(x)                                             #Fire(96, 16, 64, 64),
            x = Fire1(16, 64, 64).forward(x)                                             #Fire(128, 16, 64, 64),
            x = Fire1(32, 128, 128).forward(x)                                           #Fire(128, 32, 128, 128),
            x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)                   #MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            x = Fire1(32, 128, 128).forward(x)                                           #Fire(256, 32, 128, 128),
            x = Fire1(48, 192, 192).forward(x)                                           #Fire(256, 48, 192, 192),
            x = Fire1(48, 192, 192).forward(x)                                           #Fire(384, 48, 192, 192),
            x = Fire1(64, 256, 256).forward(x)                                           #Fire(384, 64, 256, 256),
            x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)                  #MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            x = Fire1(64, 256, 256).forward(x)                                            #Fire(512, 64, 256, 256),
        else:
             x = Conv2D(64,(3,3), padding='valid',strides=(2, 2))(x)                #nn.Conv2d(3, 64, kernel_size=3, stride=2),
             x = Activation('relu')(x)                                              #nn.ReLU(inplace=True),
             x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)                    #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
             x = Fire1(16, 64, 64).forward(x)                                           #Fire(64, 16, 64, 64),
             x = Fire1(16, 64, 64).forward(x)                                           #Fire(128, 16, 64, 64),
             x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)                    #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
             x = Fire1(32, 128, 128).forward(x)                                        #Fire(128, 32, 128, 128),
             x = Fire1(32, 128, 128).forward(x)                                         #Fire(256, 32, 128, 128),
             x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)                    #nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
             x = Fire1(48, 192, 192).forward(x)                                         #Fire(256, 48, 192, 192),
             x = Fire1(48, 192, 192).forward(x)                                        #Fire(384, 48, 192, 192),
             x = Fire1(64, 256, 256).forward(x)                                        #Fire(384, 64, 256, 256),
             x = Fire1(64, 256, 256).forward(x)                                          #Fire(512, 64, 256, 256),
        return x
    
    def classifier(self,x):
        x = Dropout(0.5)(x)
        x = Conv2D(self.num_classes, kernel_size=(1,1),padding='valid')(x)
        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        return x
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if m is final_conv:
#                     init.normal(m.weight.data, mean=0.0, std=0.01)
#                 else:
#                     init.kaiming_uniform(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)                                                  
        return x                                                              #x.view(x.size(0), self.num_classes)

def squeezenet1_0(x,**kwargs):
    model = SqueezeNet(1,kwargs['num_classes'])
    x = model.forward(x)
    return x

def squeezenet1_1(x, **kwargs):
    model = SqueezeNet(1.1,kwargs['num_classes'])
    x = model.forward(x)
    return x