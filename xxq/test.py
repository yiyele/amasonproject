'''
Created on 2017年6月13日

@author: USER
'''

from itertools import chain
import pandas as pd
from tensorflow.contrib.keras.api.keras.models import Sequential,Model
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Input

import xxq.model.vgg as vgg
import xxq.model.squeezenet as squeezenet
import xxq.model.resnet as resnet
import xxq.model.inception as inception
import xxq.model.densenet as densenet
import xxq.model.alexnet as alexnet
from keras.utils import plot_model
from keras import backend as K

import numpy as np
#from keras.utils.visualize_util import plot

def main():
#     input_shape = Input(shape=(224,224, 3))
#     x = resnet.resnet18(input_shape,num_classes=1000)
#     model = Model(inputs=input_shape,outputs=x)
#     print (K.image_data_format())
    
#     input_shape = Input(shape=(299,299, 3))
#     x = inception.inception_v3(input_shape,num_classes=1000, aux_logits=True, transform_input=False)
#     model = Model(inputs=input_shape,outputs=x)
    
#     input_shape = Input(shape=(299,299, 3))
#     x = densenet.densenet161(input_shape,num_classes=1000)
#     model = Model(inputs=input_shape,outputs=x)

#     input_shape = Input(shape=(299,299, 3))
#     x = alexnet.alexnet(input_shape,num_classes=1000)
#     model = Model(inputs=input_shape,outputs=x)

    []    
    return

if __name__ == '__main__':
    main()
