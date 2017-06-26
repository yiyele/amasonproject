# -*- coding: utf-8 -*-
'''
Created on 2017年6月21日
incepion网络模型实现，默认的输入img_size(299,299)
@author: USER
'''

from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten,Activation,Input,GlobalAveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,AveragePooling2D
from tensorflow.contrib.keras.api.keras.layers import concatenate 
from keras.models import Model,Sequential
from keras.layers import merge
from keras import backend as K

__all__ = ['Inception3', 'inception_v3']

def inception_v3(x,**kwargs):
    return Inception3(**kwargs).forward(x)

class Inception3:
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False):
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = BasicConv2d(3, 32, kernel_size=(3,3), strides=(2,2))
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=(3,3))
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=(3,3), padding='same')
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80, kernel_size=(1,1))
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=(3,3))
        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)
        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = InceptionAux(768, num_classes)
        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)
        self.fc = Dense(num_classes)

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 import scipy.stats as stats
#                 stddev = m.stddev if hasattr(m, 'stddev') else 0.1
#                 X = stats.truncnorm(-2, 2, scale=stddev)
#                 values = torch.Tensor(X.rvs(m.weight.data.numel()))
#                 m.weight.data.copy_(values)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3.forward(x)                        #x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3.forward(x)                         #x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3.forward(x)                       #x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)        #x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1.forward(x)                 #x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3.forward(x)                 #x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)  #x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b.forward(x)                            #x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c.forward(x)                  #x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d.forward(x)                 #x = self.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.Mixed_6a.forward(x)                    #x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b.forward(x)                    #x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c.forward(x)                 #x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d.forward(x)                #x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e.forward(x)          #x = self.Mixed_6e(x)
        # 17 x 17 x 768
#         if self.aux_logits:
#             aux = self.AuxLogits.forward(x)                  #aux = self.AuxLogits(x)
        # 17 x 17 x 768
        x = self.Mixed_7a.forward(x)        #x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b.forward(x)             #x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c.forward(x)             #x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = AveragePooling2D(pool_size=(8,8))(x)         #x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        x = Dropout(0.5)(x)    #x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = Flatten()(x)        #x = x.view(x.size(0), -1)
        # 2048
        x = self.fc(x)
        # 1000 (num_classes)
#         if self.aux_logits:
#             return x, aux
        return x


class InceptionA:
    def __init__(self, in_channels, pool_features):
        self.branch1x1 = BasicConv2d(in_channels, 64, kernel_size=(1,1))

        self.branch5x5_1 = BasicConv2d(in_channels, 48, kernel_size=(1,1))
        self.branch5x5_2 = BasicConv2d(48, 64, kernel_size=(5,5), padding='same')

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=(1,1))
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=(3,3), padding='same')
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=(3,3), padding='same')

        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=(1,1))

    def forward(self, x):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        branch1x1 = self.branch1x1.forward(x)

        branch5x5 = self.branch5x5_1.forward(x)
        branch5x5 = self.branch5x5_2.forward(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1.forward(x)
        branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3.forward(branch3x3dbl)

        branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)   #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool.forward(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return concatenate(outputs, axis=channel_axis)                   #torch.cat(outputs, 1)


class InceptionB:
    def __init__(self, in_channels):
        self.branch3x3 = BasicConv2d(in_channels, 384, kernel_size=(3,3), strides=(2,2))

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 64, kernel_size=(1,1))
        self.branch3x3dbl_2 = BasicConv2d(64, 96, kernel_size=(3,3), padding='same')
        self.branch3x3dbl_3 = BasicConv2d(96, 96, kernel_size=(3,3), strides=(2,2))

    def forward(self, x):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        branch3x3 = self.branch3x3.forward(x)

        branch3x3dbl = self.branch3x3dbl_1.forward(x)
        branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3.forward(branch3x3dbl)

        branch_pool = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)     #branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return concatenate(outputs, axis=channel_axis)               #torch.cat(outputs, 1)


class InceptionC:
    def __init__(self, in_channels, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, 192, kernel_size=(1,1))

        c7 = channels_7x7
        self.branch7x7_1 = BasicConv2d(in_channels, c7, kernel_size=(1,1))
        self.branch7x7_2 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding='same')
        self.branch7x7_3 = BasicConv2d(c7, 192, kernel_size=(7, 1), padding='same')

        self.branch7x7dbl_1 = BasicConv2d(in_channels, c7, kernel_size=(1,1))
        self.branch7x7dbl_2 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding='same')
        self.branch7x7dbl_3 = BasicConv2d(c7, c7, kernel_size=(1, 7), padding='same')
        self.branch7x7dbl_4 = BasicConv2d(c7, c7, kernel_size=(7, 1), padding='same')
        self.branch7x7dbl_5 = BasicConv2d(c7, 192, kernel_size=(1, 7), padding='same')

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=(1,1))

    def forward(self, x):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        branch1x1 = self.branch1x1.forward(x)

        branch7x7 = self.branch7x7_1.forward(x)
        branch7x7 = self.branch7x7_2.forward(branch7x7)
        branch7x7 = self.branch7x7_3.forward(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1.forward(x)
        branch7x7dbl = self.branch7x7dbl_2.forward(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3.forward(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4.forward(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5.forward(branch7x7dbl)

        branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)          #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool.forward(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return concatenate(outputs, axis=channel_axis)          #torch.cat(outputs, 1)


class InceptionD:
    def __init__(self, in_channels):
        self.branch3x3_1 = BasicConv2d(in_channels, 192, kernel_size=(1,1))
        self.branch3x3_2 = BasicConv2d(192, 320, kernel_size=(3,3), strides=(2,2))

        self.branch7x7x3_1 = BasicConv2d(in_channels, 192, kernel_size=(1,1))
        self.branch7x7x3_2 = BasicConv2d(192, 192, kernel_size=(1, 7), padding='same')
        self.branch7x7x3_3 = BasicConv2d(192, 192, kernel_size=(7, 1), padding='same')
        self.branch7x7x3_4 = BasicConv2d(192, 192, kernel_size=(3,3), strides=(2,2))

    def forward(self, x):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        branch3x3 = self.branch3x3_1.forward(x)
        branch3x3 = self.branch3x3_2.forward(branch3x3)

        branch7x7x3 = self.branch7x7x3_1.forward(x)
        branch7x7x3 = self.branch7x7x3_2.forward(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3.forward(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4.forward(branch7x7x3)

        branch_pool = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)    #branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return concatenate(outputs, axis=channel_axis)        #torch.cat(outputs, 1)


class InceptionE:
    def __init__(self, in_channels):
        self.branch1x1 = BasicConv2d(in_channels, 320, kernel_size=(1,1))

        self.branch3x3_1 = BasicConv2d(in_channels, 384, kernel_size=(1,1))
        self.branch3x3_2a = BasicConv2d(384, 384, kernel_size=(1, 3), padding='same')
        self.branch3x3_2b = BasicConv2d(384, 384, kernel_size=(3, 1), padding='same')

        self.branch3x3dbl_1 = BasicConv2d(in_channels, 448, kernel_size=(1,1))
        self.branch3x3dbl_2 = BasicConv2d(448, 384, kernel_size=(3,3), padding='same')
        self.branch3x3dbl_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding='same')
        self.branch3x3dbl_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding='same')

        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=(1,1))

    def forward(self, x):
        if K.image_data_format() == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        branch1x1 = self.branch1x1.forward(x)

        branch3x3 = self.branch3x3_1.forward(x)
        branch3x3 = [
            self.branch3x3_2a.forward(branch3x3),
            self.branch3x3_2b.forward(branch3x3),
        ]
        branch3x3 = concatenate(branch3x3, axis=channel_axis)          #branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1.forward(x)
        branch3x3dbl = self.branch3x3dbl_2.forward(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a.forward(branch3x3dbl),
            self.branch3x3dbl_3b.forward(branch3x3dbl),
        ]
        branch3x3dbl = concatenate(branch3x3dbl, axis=channel_axis)           #branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = AveragePooling2D(pool_size=(3,3), strides=(1,1), padding='same')(x)       #branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool.forward(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return concatenate(outputs, axis=channel_axis) #torch.cat(outputs, 1)


class InceptionAux:
    def __init__(self, in_channels, num_classes):
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=(1,1))
        self.conv1 = BasicConv2d(128, 768, kernel_size=(5,5))
        self.conv1.stddev = 0.01
        self.fc = Dense(num_classes)    #self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # 17 x 17 x 768
        x = AveragePooling2D(pool_size=(5,5), strides=(3,3))(x)    #x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # 5 x 5 x 768
        x = self.conv0.forward(x)              #x = self.conv0(x)
        # 5 x 5 x 128
        x = self.conv1.forward(x)    #x = self.conv1(x)
        # 1 x 1 x 768
        x = Flatten()(x)     #x = x.view(x.size(0), -1)
        # 768
        x = self.fc(x)
        # 1000
        return x


class BasicConv2d:
    def __init__(self, in_channels, out_channels, **kwargs):
        self.conv = Conv2D(out_channels, use_bias=False, **kwargs)        #self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = BatchNormalization()                                    #self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return Activation('relu')(x)                                      #F.relu(x, inplace=True)