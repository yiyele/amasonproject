# -*- coding: utf-8 -*-
'''
Created on 2017年6月13日

@author: USER
'''

import xxq.Reader as Reader
import xxq.DataPreprocessing as DataPreprocess
import xxq.DataAnalysis as DataAnalysis
import gc
from tensorflow.contrib.keras.api.keras.callbacks import ModelCheckpoint
from xxq.Models import AmazonKerasClassifier
import pandas as pd
import numpy as np
import xxq.Models as models
from keras.applications import *
import xxq.model.vgg as vgg
import xxq.model.squeezenet as squeezenet

from itertools import chain

def main():
    #img_resize = (64, 64)
    img_resize = (64, 64) 
    validation_split_size = 0.2 
    batch_size = 128
    imageInfo = pd.read_csv('../data/train_v2.csv/train_v2.csv')
    
    labels_list = list(chain.from_iterable([tags.split(" ") for tags in imageInfo['tags'].values]))
    labels_set = set(labels_list)
    print ("总共有{}个标签，分别是{}".format(len(labels_set),labels_set))
    
    x_train, y_train, y_map = DataPreprocess.preprocess_train_data('../data/train-jpg', '../data/train_v2.csv/train_v2.csv', img_resize)
    # Free up all available memory space after this heavy operation
    gc.collect();
    print("x_train shape: {}".format(x_train.shape))
    print("y_train shape: {}".format(y_train.shape))
    
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    
    classifier = AmazonKerasClassifier()
    classifier.add_conv_layer(img_resize)
    classifier.add_flatten_layer()
    classifier.add_ann_layer(len(y_map))
    #classifier.vgg(16,img_size=img_resize,img_channels=3,output_size=len(y_map))
    #classifier.squeezenet(img_size=img_resize,img_channels=3,output_size=len(y_map))
    #classifier.resnet(1,img_size=img_resize,img_channels=3,output_size=len(y_map))
    #classifier.densenet(121,img_size=img_resize,img_channels=3,output_size=len(y_map))
    classifier.alexnet(img_size=img_resize,img_channels=3,output_size=len(y_map))
    
    #训练模型
    train_losses, val_losses = [], []
    epochs_arr = [1]#[20, 5, 5]
    learn_rates = [0.001]#[0.001, 0.0001, 0.00001]
    for learn_rate, epochs in zip(learn_rates, epochs_arr):
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model(x_train, y_train, learn_rate, epochs, batch_size, validation_split_size=validation_split_size, train_callbacks=[checkpoint])
        train_losses += tmp_train_losses
        val_losses += tmp_val_losses
    
    #加载训练数据
    classifier.load_weights("weights.best.hdf5")
    print("Weights loaded")
    print(fbeta_score)
    
    result_threshold_list_final = classifier.setBestThreshold()
    
    del x_train, y_train
    gc.collect()
    #预测
    x_test, x_test_filename = DataPreprocess.preprocess_test_data('../data/test-jpg', img_resize)
    # Predict the labels of our x_test images
    predictions = classifier.predict(x_test)
    
    del x_test
    gc.collect()
    
    x_test, x_test_filename_additional = DataPreprocess.preprocess_test_data('../data/test-jpg-additional', img_resize)
    new_predictions = classifier.predict(x_test)
    
    del x_test
    gc.collect()
    predictions = np.vstack((predictions, new_predictions))
    x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
    print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                             x_test_filename.shape,predictions[0]))

    thresholds = [0.2] * len(labels_set) 
    predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)
    
    tags_list = [None] * len(predicted_labels)
    for i, tags in enumerate(predicted_labels):
        tags_list[i] = ' '.join(map(str, tags))

    print ("tags_list:".format(tags_list))
    print ('x_test_filename'.format(x_test_filename))  
    print (':')
    print (tags_list)
    print (':')
    print (x_test_filename)

    final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]
    final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
    final_df.to_csv('./submission_file.csv', index=False)
    classifier.close()
    
#使用生成器生成训练数据

def main2():
    #img_resize = (64, 64)
    img_resize = (64, 64) 
    validation_split_size = 0.2 
    batch_size = 128
    imageInfo = pd.read_csv('../data/train_v2.csv/train_v2.csv')
    
    labels_list = list(chain.from_iterable([tags.split(" ") for tags in imageInfo['tags'].values]))
    labels_set = set(labels_list)
    print ("总共有{}个标签，分别是{}".format(len(labels_set),labels_set))
    
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    
    classifier = AmazonKerasClassifier()
    classifier.setTrainFilePath('../data/train_v2.csv/train_v2.csv')
    classifier.setValidFilePath('../data/valid-v2.csv/valid_v2.csv')
    classifier.setTrainImgFilePath('../data/train-jpg')
    classifier.setValidImgFilePath('../data/valid-jpg')
    classifier.setTestImgFilePath('../data/test-jpg')
    classifier.setTestAdditionImgFilePath('../data/test-jpg-additional')
#     classifier.add_conv_layer(img_resize)
#     classifier.add_flatten_layer()
#     classifier.add_ann_layer(len(y_map))
    #classifier.vgg(16,img_size=img_resize,img_channels=3,output_size=len(y_map))
    #classifier.squeezenet(img_size=img_resize,img_channels=3,output_size=len(y_map))
    #classifier.resnet(1,img_size=img_resize,img_channels=3,output_size=len(y_map))
    #classifier.densenet(121,img_size=img_resize,img_channels=3,output_size=len(y_map))
    classifier.alexnet(img_size=img_resize,img_channels=3,output_size=len(labels_set))
    
    #训练模型
    train_losses, val_losses = [], []
    epochs_arr = [1]#[20, 5, 5]
    learn_rates = [0.001]#[0.001, 0.0001, 0.00001]
    for learn_rate, epochs in zip(learn_rates, epochs_arr):
        tmp_train_losses, tmp_val_losses, fbeta_score = classifier.train_model_generator(classifier.generate_trainOrValid_img_from_file(classifier.getTrainImgFilePath(),classifier.getTrainFilePath(),img_resize=img_resize), classifier.generate_trainOrValid_img_from_file(classifier.getValidImgFilePath(),classifier.getValidFilePath(),img_resize=img_resize), learn_rate, epochs, steps=32383, validation_steps=8096, train_callbacks=[checkpoint])
        train_losses += tmp_train_losses
        val_losses += tmp_val_losses
    
    y_map = classifier.getYMap()
    #加载训练数据
    classifier.load_weights("weights.best.hdf5")
    print("Weights loaded")
    print(fbeta_score)
    
    result_threshold_list_final = classifier.setBestThreshold()
    
    gc.collect()
    #预测
    predictions = classifier.predict_generator(classifier.generate_test_img_from_file(classifier.getTestImgFilePath(),img_resize=img_resize),40669)
    x_test_filename = classifier.getTestImgNameList()
    gc.collect()
    
    new_predictions = classifier.predict_generator(classifier.generate_test_img_from_file(classifier.getTestImgFilePath(),img_resize=img_resize),20522)
    x_test_filename_additional = classifier.getTestImgNameList()
    gc.collect()
    predictions = np.vstack((predictions, new_predictions))
    x_test_filename = np.hstack((x_test_filename, x_test_filename_additional))
    print("Predictions shape: {}\nFiles name shape: {}\n1st predictions entry:\n{}".format(predictions.shape, 
                             x_test_filename.shape,predictions[0]))

    thresholds = [0.2] * len(labels_set) 
    predicted_labels = classifier.map_predictions(predictions, y_map, thresholds)
    
    tags_list = [None] * len(predicted_labels)
    for i, tags in enumerate(predicted_labels):
        tags_list[i] = ' '.join(map(str, tags))

    print ("tags_list:".format(tags_list))
    print ('x_test_filename'.format(x_test_filename))  
    print (':')
    print (tags_list)
    print (':')
    print (x_test_filename)

    final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]
    final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
    final_df.to_csv('./submission_file.csv', index=False)
    classifier.close()
    
    
if __name__ == '__main__':
    main2()
