# -*- coding: utf-8 -*-
'''
Created on 2017年6月16日

@author: USER
'''

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from sklearn.utils import shuffle
import numpy as np

import h5py

#使用keras预训练的模型
def write_gap(self,MODEL,image_size,lambda_func=None):
    width = image_size[0]
    height = image_size[0]
    input_tensor = Input((height,width,3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    
    base_model = MODEL(input_tensor=x,weights='imagenet',include_top=False)
    model = Model(base_model.input,GlobalAveragePooling2D()(base_model.output))
    
    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("./data/train-jpg",image_size,shuffle=False,batch_size=16)
    test_generator = gen.flow_from_directory("./data/test-jpg",image_size,shuffle=False,batch_size=16,class_model=None)
    test_generator_addtion = gen.flow_from_directory("./data/test-jpg-additional",image_size,shuffle=False,batch_size=16,class_model=None)
    test_filenames = test_generator.filenames
    test_addtion_filenames = test_generator_addtion.filenames
    
    train = model.predict_generator(train_generator,train_generator.nb_sample)
    test = model.predict_generator(test_generator,test_generator.nb_sample)
    test_addi = model.predict_generator(test_generator_addtion,test_generator_addtion.nb_sample)
    with h5py.File("gap_%s.h5"%MODEL.func_name) as h:
        h.create_dataset("train",data=train)
        h.create_dataset("test",data=test)
        h.create_dataset("test_addition",data=test_addi)
        h.create_dataset("label",data=train_generator.classes)
    x_test_filename = np.hstack((test_filenames, test_addtion_filenames))
    return x_test_filename

def run_modelOfPretrain(output_size,batch_size):
    np.random.seed(2017)
    x_train = []
    x_test = []
    x_test_addi = []
    
    for filename in ["gap_ResNet50.h5","gap_Xception.h5","gap_InceptionV3.h5"]:
        with h5py.File(filename,'r') as h:
            x_train.append(np.array(h['train']))
            x_test.append(np.array[h['test']])
            x_test_addi.append(np.array[h['test_addition']])
            y_train = np.array(h['label'])
    
    x_train = np.concatenate(x_train,axis=1)
    x_test = np.concatenate(x_test,axis=1)
    x_test_addi = np.concatenate(x_test_addi,axis=1)
    
    #x_train,y_train = shuffle(x_train,y_train)

    input_tensor = Input(x_train.shape[1:])
    x = Dropout(0.5)(input_tensor)
    x = Dense(output_size,activation='sigmoid')(x)
    
    model = Model(input_tensor,x)
    
    model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
    
    y_pre = model.predict(x_test, batch_size=batch_size, verbose=1)
    y_pre = y_pre.clip(min=0.005,max=0.995)
    
    y_pre_addi = model.predict(x_test_addi, batch_size=batch_size, verbose=1)
    y_pre_addi = y_pre_addi.clip(min=0.005,max=0.995)
    
    predictions = np.vstack((y_pre, y_pre_addi))
    
