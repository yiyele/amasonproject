# -*- coding: utf-8 -*-
'''
Created on 2017年6月19日
Funtion：构建一个vgg模型的keras库(有待测试)，默认的输入img_size(224,224)
@author: USER
'''
from keras.layers import Conv2D, MaxPooling2D, Input,Dropout,Flatten, Dense,Activation
from keras.models import Model,Sequential
from keras.layers.normalization import BatchNormalization
import math


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

def vgg11(**kwargs):
    model = VGG(make_layers(cfg['A']), kwargs['num_classes'])
    return model.forward()

def vgg11_bn(**kwargs):
    model = VGG(make_layers(cfg['A'], batch_norm=True), kwargs['num_classes'])
    return model.forward()

def vgg13(**kwargs):
    model = VGG(make_layers(cfg['B']), kwargs['num_classes'])
    return model.forward()

def vgg13_bn(**kwargs):
    model = VGG(make_layers(cfg['B'], batch_norm=True), kwargs['num_classes'])
    return model.forward()

def vgg16(**kwargs):
    model = VGG(make_layers(cfg['D']), kwargs['num_classes'])
    return model.forward()

def vgg16_bn(**kwargs):
    model = VGG(make_layers(cfg['D'], batch_norm=True), kwargs['num_classes'])
    return model.forward()

def vgg19(**kwargs):
    model = VGG(make_layers(cfg['E']), kwargs['num_classes'])
    return model.forward()

def vgg19_bn(**kwargs):
    model = VGG(make_layers(cfg['E'], batch_norm=True), kwargs['num_classes'])
    return model.forward()

class VGG:
    def __init__(self, features, num_classes=1000):
        self.features = features
        self.layer_list = []
        
        self.classifier = [
          "Dense(4096)" ,                                     #nn.Linear(512 * 7 * 7, 4096),
          "Activation('relu')",                               #nn.ReLU(True),
          "Dropout(0.5)" ,                                    #nn.Dropout(),  
          "Dense(4096, activation='relu')",                   #nn.Linear(4096, 4096),  
          "Activation('relu')",                               #nn.ReLU(True),
          "Dropout(0.5)" ,                                    #nn.Dropout(),  
          "Dense({})".format(num_classes)                     #nn.Linear(4096, num_classes),
        ]
#        self._initialize_weights()

    def forward(self):
        #self.layer_list = ["BatchNormalization(input_shape=(*img_size, img_channels))"]
        #model.add(Dense(32, input_shape=(784,)))
        self.layer_list.extend(self.features)
        self.layer_list.append("Flatten()")                       #x = x.view(x.size(0), -1)
        self.layer_list.extend(self.classifier)
        return self.layer_list
    
    def getLayerList(self):
        return self.layer_list
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, Conv2D):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, BatchNormalization):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()
def make_layers(cfg, batch_norm=False):
    layers = []
#    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += ["MaxPooling2D(pool_size=(2,2), strides=(2,2))"]                            # layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = "Conv2D({},(3,3), padding='same')".format(v)                                 #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, "BatchNormalization()","Activation('relu')"]                   #layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d,"Activation('relu')"]                                           #layers += [conv2d, nn.ReLU(inplace=True)]
#            in_channels = v
    return layers

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}