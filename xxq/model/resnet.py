# -*- coding: utf-8 -*-
'''
Created on 2017年6月21日

@author: USER
'''
import math
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten,Activation,Input,GlobalAveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import concatenate 
from keras.models import Model,Sequential
from keras.layers import merge

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

def conv3x3(out_planes, stride=1):
#     return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1, bias=False)
    return Conv2D(out_planes,kernel_size=(3,3),padding='same',strides=(stride,stride),use_bias=False)

class BasicBlock:
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
#         self.conv1 = conv3x3(planes, stride)
#         self.bn1 = BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
        self.conv1 = conv3x3(planes, stride) 
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu') 
        self.conv2 = conv3x3(planes)
        self.bn2 = BatchNormalization()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            for i,value in enumerate(self.downsample):
                x = eval(value)(x)
        out = merge([out,x],mode='sum')
        out = self.relu(out)
        return out

class Bottleneck:
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
        self.conv1 = Conv2D(planes,kernel_size=(1,1),use_bias=False) 
        self.bn1 = BatchNormalization() 
        self.conv2 = Conv2D(planes,kernel_size=(3,3),strides=(stride,stride),padding='same',use_bias=False)
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(planes * 4, kernel_size=(1,1), use_bias=False)
        self.bn3 = BatchNormalization()
        self.relu = Activation('relu')
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            for i,value in enumerate(self.downsample):
                x = eval(value)(x)

        out = merge([out,x],mode='sum')
        out = self.relu(out)
        return out

class ResNet:
    def __init__(self, block, layers, num_classes=1000):
#         self.inplanes = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
#         self.avgpool = nn.AvgPool2d(7)
#         self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.inplanes = 64
        self.conv1 = Conv2D(64,kernel_size=(7,7),strides=(2,2),padding='same',use_bias=False)
        self.bn1 = BatchNormalization()
        self.relu = Activation('relu')
        self.maxpool = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding='same')
        self.block = block
        self.layers = layers
#         self.layer1 = self._make_layer(block, 64, layers[0])
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AveragePooling2D(pool_size=(7,7))
        self.fc = Dense(num_classes)
        
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

    def _make_layer(self,x, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = [
                'Conv2D({},kernel_size=(1,1), strides=({},{}), use_bias=False)'.format(planes * block.expansion,stride,stride),
                'BatchNormalization()'
            ]
        #layers = []
        x = block(self.inplanes, planes, stride, downsample).forward(x)     #layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            x = block(self.inplanes, planes).forward(x)                #layers.append(block(self.inplanes, planes))
        return x                            #Sequential(layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
        x = self._make_layer(x,self.block, 64, self.layers[0])
        x = self._make_layer(x,self.block, 128, self.layers[1], stride=2)
        x = self._make_layer(x,self.block, 256, self.layers[2], stride=2)
        x = self._make_layer(x,self.block, 512, self.layers[3], stride=2)

        x = self.avgpool(x)
        x = Flatten()(x)  #x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(x,**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], kwargs['num_classes'])
    x = model.forward(x)
    return x

def resnet34(x,**kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], kwargs['num_classes'])
    x = model.forward(x)
    return x

def resnet50(x,**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], kwargs['num_classes'])
    x = model.forward(x)
    return x

def resnet101(x,**kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], kwargs['num_classes'])
    x = model.forward(x)
    return x

def resnet152(x,**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], kwargs['num_classes'])
    x = model.forward(x)
    return x