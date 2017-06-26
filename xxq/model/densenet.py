# -*- coding: utf-8 -*-
'''
Created on 2017年6月21日

@author: USER
'''
from collections import OrderedDict
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten,Activation,Input,GlobalAveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import concatenate 
from keras.models import Model,Sequential
from keras.layers import merge
from keras import backend as K

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

def densenet121(x,**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),num_classes=kwargs['num_classes'])
    x = model.forward(x)
    return x

def densenet169(x,**kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),num_classes=kwargs['num_classes'])
    x = model.forward(x)
    return x

def densenet201(x, **kwargs):
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),num_classes=kwargs['num_classes'])
    x = model.forward(x)
    return x

def densenet161(x,**kwargs):
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),num_classes=kwargs['num_classes'])
    x = model.forward(x)
    return x

class _DenseLayer:
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        self.BN1 = BatchNormalization()        #self.add_module('norm.1', BatchNormalization()),
        self.relu = Activation('relu')          #self.add_module('relu.1', Activation('relu')),
        self.conv = Conv2D(bn_size *growth_rate, kernel_size=(1,1), strides=(1,1), use_bias=False)      #self.add_module('conv.1', Conv2D(bn_size *growth_rate, kernel_size=(1,1), strides=(1,1), use_bias=False)),
        self.BN2 = BatchNormalization()        #self.add_module('norm.2', BatchNormalization()),
        self.relu2 = Activation('relu')           #self.add_module('relu.2', Activation('relu')),
        self.conv2 = Conv2D(growth_rate,kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)   #self.add_module('conv.2', Conv2D(growth_rate,kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        x = self.BN1(x) 
        x = self.relu(x)
        x = self.conv(x)
        x = self.BN2(x)
        x = self.relu2(x)
        new_features = self.conv2(x)                     #new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = Dropout(self.drop_rate)(new_features)     #new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return concatenate([x, new_features], axis=channel_axis)     #torch.cat([x, new_features], 1)


class _DenseBlock:
    def __init__(self, x,num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        for i in range(num_layers):
            x = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate).forward(x)  #layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            #self.add_module('denselayer%d' % (i + 1), layer)

class _Transition:
    def __init__(self,num_input_features, num_output_features):
        self.BN1 = BatchNormalization()                                                          #self.add_module('norm', BatchNormalization())
        self.relu = Activation('relu')                                                           #self.add_module('relu', Activation('relu'))
        self.conv = Conv2D(num_output_features,kernel_size=(1,1), strides=(1,1), use_bias=False)  #self.add_module('conv', Conv2D(num_output_features,kernel_size=(1,1), strides=(1,1), use_bias=False))
        self.avp = AveragePooling2D(pool_size=(2,2), strides=(2,2))                              #self.add_module('pool', AveragePooling2D(pool_size=(2,2), strides=(2,2)))
    
    def forward(self,x):
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv(x)
        x = self.avp(x)
        return x
    
class DenseNet:
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        self.features = [
            Conv2D(num_init_features, kernel_size=(7,7), strides=(2,2), padding='same', use_bias=False),  #('conv0', Conv2D(num_init_features, kernel_size=(7,7), strides=(2,2), padding='same', use_bias=False)),
            BatchNormalization(),                                                                         #('norm0', BatchNormalization()),
            Activation('relu'),                                                                           #('relu0', Activation('relu')),
            MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')                                 #('pool0', MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))
        ]
        self.num_init_features = num_init_features
        self.block_config = block_config
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate
        # Each denseblock
#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = num_features // 2
 
        # Final batch norm
        self.BN_last = BatchNormalization()   #self.features.add_module('norm5', BatchNormalization())
        # Linear layer
        self.classifier = Dense(num_classes)                  #self.classifier = nn.Linear(num_features, num_classes)
    
    def feature(self,x):
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            _DenseBlock(x,num_layers=num_layers, num_input_features=num_features,bn_size=self.bn_size, growth_rate=self.growth_rate, drop_rate=self.drop_rate)
            #self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * self.growth_rate
            if i != len(self.block_config) - 1:
                x = _Transition(num_input_features=num_features, num_output_features=num_features // 2).forward(x)
                #self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
        return x

    def forward(self, x):
        for value in self.features:
            x = value(x)
        features = self.feature(x)
        features = self.BN_last(features)
        out = Activation('relu')(features)                    #out = F.relu(features, inplace=True)
        out = AveragePooling2D(pool_size=(7,7))(out)          #out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = Flatten()(out)
        out = self.classifier(out)
        return out