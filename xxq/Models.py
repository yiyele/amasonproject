import numpy as np
import os
import pandas as pd
from itertools import chain
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

import tensorflow.contrib.keras.api.keras as k
from tensorflow.contrib.keras.api.keras.models import Sequential,Model
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout, Flatten,Activation,GlobalAveragePooling2D,concatenate
from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D, BatchNormalization,Input,AveragePooling2D
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from tensorflow.contrib.keras.api.keras.callbacks import Callback, EarlyStopping
from tensorflow.contrib.keras import backend
import copy
import xxq.model.vgg as vgg
import xxq.model.squeezenet as squeezenet
import xxq.model.resnet as resnet
import xxq.model.inception as inception
import xxq.model.densenet as densenet
import xxq.model.alexnet as alexnet

class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []


    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

class AmazonKerasClassifier:
    def __init__(self):
        self.losses = []
        self.classifier = Sequential()
        self.x_vail = []
        self.y_vail = []
        self.train_filepath = ''
        self.train_img_filepath = ''
        self.valid_filepath = ''
        self.valid_img_filepath = ''
        self.test_img_filepath = ''
        self.test_addition_img_filepath = ''
        self.test_img_name_list = ''
        self.y_map = {}
        
    def setTrainFilePath(self,value):
        self.train_filepath = value
    def getTrainFilePath(self):
        return self.train_filepath
    def setValidFilePath(self,value):
        self.valid_filepath = value
    def getValidFilePath(self):
        return self.valid_filepath
    def setTrainImgFilePath(self,value):
        self.train_img_filepath = value
    def getTrainImgFilePath(self):
        return self.train_img_filepath
    def setValidImgFilePath(self,value):
        self.valid_img_filepath = value
    def getValidImgFilePath(self):
        return self.valid_img_filepath
    def setTestImgFilePath(self,value):
        self.test_img_filepath = value
    def getTestImgFilePath(self):
        return self.test_img_filepath
    def setTestAdditionImgFilePath(self,value):
        self.test_addition_img_filepath = value
    def getTestAdditionImgFilePath(self):
        return self.test_addition_img_filepath
    def getTestImgNameList(self):
        return self.test_img_name_list
    def getYMap(self):
        return self.y_map
        
    def vgg(self,type=16,bn=False,img_size=(224,224),img_channels=3,output_size=1000):
        if type == 16 and bn == False:
            layer_list = vgg.vgg16(num_classes=output_size)
        elif type == 16 and bn == True:
            layer_list = vgg.vgg16_bn(num_classes=output_size)
        elif type == 11 and bn == False:
            layer_list = vgg.vgg11(num_classes=output_size)
        elif type == 11 and bn == True:
            layer_list = vgg.vgg11_bn(num_classes=output_size)
        elif type == 13 and bn == False:
            layer_list = vgg.vgg13(num_classes=output_size)
        elif type == 13 and bn == True:
            layer_list = vgg.vgg13_bn(num_classes=output_size)
        elif type == 19 and bn == False:
            layer_list = vgg.vgg19(num_classes=output_size)
        elif type == 19 and bn == True:
            layer_list = vgg.vgg19_bn(num_classes=output_size)
        else:
            print ("请输入11,13,16,19这四个数字中的一个！")
        self.classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))
        for i,value in enumerate(layer_list):
            self.classifier.add(eval(value))
    
    def squeezenet(self,type,img_size=(64,64),img_channels=3,output_size=1000):
        input_shape = Input(shape=(*img_size, img_channels))
        if type == 1:
            x = squeezenet.squeezenet1_0(input_shape,num_classes=output_size)
        elif type == 1.1:
            x = squeezenet.squeezenet1_1(input_shape,num_classes=output_size)
        else:
            print ("请输入1,1.0这两个数字中的一个！")
        model = Model(inputs=input_shape,outputs=x)
        self.classifier = model
        
    def resnet(self,type,img_size=(64,64),img_channels=3,output_size=1000):
        input_shape = Input(shape=(*img_size, img_channels))
        if type == 18:
            x = resnet.resnet18(input_shape,num_classes=output_size)
        elif type == 34:
            x = resnet.resnet34(input_shape,num_classes=output_size)
        elif type == 50:
            x = resnet.resnet50(input_shape,num_classes=output_size)
        elif type == 101:
            x = resnet.resnet101(input_shape,num_classes=output_size)
        elif type == 152:
            x = resnet.resnet152(input_shape,num_classes=output_size)
        else:
            print ("请输入18,34,50,101,152这五个数字中的一个！")
            return
        model = Model(inputs=input_shape,outputs=x)
        self.classifier = model
    
    def inception(self,img_size=(299,299),img_channels=3,output_size=1000):
        input_shape = Input(shape=(*img_size, img_channels))
        x = inception.inception_v3(input_shape,num_classes=output_size, aux_logits=True, transform_input=False)
        model = Model(inputs=input_shape,outputs=x)
        self.classifier = model
        
    def densenet(self,type,img_size=(299,299),img_channels=3,output_size=1000):
        input_shape = Input(shape=(*img_size, img_channels))
        if type == 161:
            x = densenet.densenet161(input_shape,num_classes=output_size)
        elif type == 121:
            x = densenet.densenet121(input_shape,num_classes=output_size)
        elif type == 169:
            x = densenet.densenet169(input_shape,num_classes=output_size)
        elif type == 201:
            x = densenet.densenet201(input_shape,num_classes=output_size)
        else:
            print ("请输入161,121,169,201这四个数字中的一个！")
            return 
        model = Model(inputs=input_shape,outputs=x)
        self.classifier = model
    
    def alexnet(self,img_size=(299,299),img_channels=3,output_size=1000):
        input_shape = Input(shape=(*img_size, img_channels))
        x = alexnet.alexnet(input_shape,num_classes=output_size)
        model = Model(inputs=input_shape,outputs=x)
        self.classifier = model
    
    def add_conv_layer(self, img_size=(32, 32), img_channels=3):
        self.classifier.add(BatchNormalization(input_shape=(*img_size, img_channels)))

        self.classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(64, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(128, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

        self.classifier.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        self.classifier.add(Conv2D(256, (3, 3), activation='relu'))
        self.classifier.add(MaxPooling2D(pool_size=2))
        self.classifier.add(Dropout(0.25))

    def add_flatten_layer(self):
        self.classifier.add(Flatten())

    def add_ann_layer(self, output_size):
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(BatchNormalization())
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    def _get_fbeta_score2(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        result_threshold_list_final,score_result = self.grid_search_best_threshold(y_valid,np.array(p_valid))
        return result_threshold_list_final,score_result
    
    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')
    
    def grid_search_best_threshold(self,y_valid,p_valid):
        threshold_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
        result_threshold_list_temp = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
        result_threshold_list_final = [0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]
        for i in range(17):
            score_result = 0
            for j in range(9):
                result_threshold_list_temp[i] = threshold_list[j]
                score_temp = fbeta_score(y_valid, p_valid > result_threshold_list_temp, beta=2, average='samples')
                if score_result < score_temp:
                    score_result = score_temp
                    result_threshold_list_final[i] = threshold_list[j]
            result_threshold_list_temp[i] = result_threshold_list_final[i]
        return result_threshold_list_final,score_result
    
    def train_model(self, x_train, y_train, learn_rate=0.001, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train,
                                                              test_size=validation_split_size)
        
        self.x_vail = X_valid
        self.y_vail = y_valid
        opt = Adam(lr=learn_rate)

        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.classifier.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=(X_valid, y_valid),
                            callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]
    
    def train_model_generator(self,generator_train,generator_valid,learn_rate=0.001, epoch=5,batchSize=128, steps=32383, validation_steps=8096, train_callbacks=()):
        history = LossHistory()
        #valid 8096  32383
        opt = Adam(lr=learn_rate)
        
        steps = steps / batchSize + 1 - 9
        validation_steps = validation_steps / batchSize + 1
        if steps % batchSize == 0:
            steps = steps / batchSize - 9
        if validation_steps % batchSize == 0:
            validation_steps = validation_steps / batchSize
        
        print(steps,validation_steps)
        self.classifier.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='auto')

        self.classifier.fit_generator(generator_train,
                            steps_per_epoch=steps,
                            epochs=epoch,
                            verbose=1,
                            validation_data=generator_valid,
                            validation_steps = validation_steps,
                            callbacks=[history, *train_callbacks, earlyStopping])
        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)
        return [history.train_losses, history.val_losses, fbeta_score]
 
    def generate_trainOrValid_img_from_file(self,train_set_folder,train_csv_file,img_resize=(32, 32),batchSize=128,process_count=cpu_count()):
        labels_df = pd.read_csv(train_csv_file)
        labels = sorted(set(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values])))
        labels_map = {l: i for i, l in enumerate(labels)}
        
        files_path = []
        tags_list = []
        for file_name, tags in labels_df.values:
            files_path.append('{}/{}.jpg'.format(train_set_folder, file_name))
            tags_list.append(tags)
        
        X = []
        Y = []
        
        iter_num = 1
        self.y_map = {v: k for k, v in labels_map.items()}
        with ThreadPoolExecutor(process_count) as pool:
            for img_array, targets in tqdm(pool.map(self._train_transform_to_matrices,
                                                    [(file_path, tag, labels_map, img_resize)
                                                     for file_path, tag in zip(files_path, tags_list)]),
                                           total=len(files_path)):
                if iter_num % batchSize == 0:
                    X = []
                    Y = []
                    iter_num = 0
                X.append(img_array)
                Y.append(targets)
                iter_num += 1
                if iter_num == batchSize:
                    print (iter_num)
                    yield (np.array(X),np.array(Y))

            
    def _train_transform_to_matrices(self,*args):
        file_path, tags, labels_map, img_resize = list(args[0])
        img = Image.open(file_path)
        img.thumbnail(img_resize)  
    
        img_array = np.asarray(img.convert("RGB"), dtype=np.float32) / 255
    
        targets = np.zeros(len(labels_map))
        for t in tags.split(' '):
            targets[labels_map[t]] = 1
        return img_array, targets
    
    def generate_test_img_from_file(self,test_set_folder,img_resize=(32, 32),batchSize=128,process_count=cpu_count()):
        x_test = []
        x_test_filename = []
        files_name = os.listdir(test_set_folder)
        
        X = []
        Y = []
        iter_num = 1
        with ThreadPoolExecutor(process_count) as pool:
            for img_array, file_name in tqdm(pool.map(_test_transform_to_matrices,
                                                      [(test_set_folder, file_name, img_resize)
                                                       for file_name in files_name]),
                                             total=len(files_name)):
                x_test.append(img_array)
                x_test_filename.append(file_name)
                self.test_img_name_list = x_test_filename
                
                if iter_num % batchSize == 0:
                    X = []
                    Y = []
                    iter_num = 0
                X.append(img_array)
                Y.append(targets)
                iter_num += 1
                if iter_num == batchSize:
                    print (iter_num)
                    yield (np.array(X),np.array(Y))
    
    def _test_transform_to_matrices(self,*args):
        test_set_folder, file_name, img_resize = list(args[0])
        img = Image.open('{}/{}'.format(test_set_folder, file_name))
        img.thumbnail(img_resize)
        # Convert to RGB and normalize
        img_array = np.array(img.convert("RGB"), dtype=np.float32) / 255
        return img_array, file_name
    
    def save_weights(self, weight_file_path):
        self.classifier.save_weights(weight_file_path)
    
    def load_weights(self, weight_file_path):
        self.classifier.load_weights(weight_file_path)
        
    def setBestThreshold(self):
        result_threshold_list_final,score_result = self._get_fbeta_score2(self.classifier, self.x_vail, self.y_vail)
        print ('最好得分:{}'.format(score_result))
        print ('最好的阈值:{}'.format(result_threshold_list_final))
        return result_threshold_list_final

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
        return predictions
    
    def predict_generator(self, generator):
        predictions = self.classifier.predcit_generator(generator)
        return predictions

    def map_predictions(self, predictions, labels_map, thresholds):
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        backend.clear_session()
