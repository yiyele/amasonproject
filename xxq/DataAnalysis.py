# -*- coding: utf-8 -*-
'''
Created on 2017年6月13日

@author: USER
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import pandas as pd
#import seaborn as sns

def showImages(imageInfo,labels_set,train_jpeg_dir):
    images_title = [imageInfo[imageInfo['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg' 
                for i, label in enumerate(labels_set)]

    plt.rc('axes', grid=False)
    _, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
    axs = axs.ravel()
    
    for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
        img = Image.open(train_jpeg_dir + '/' + image_name)
        #img = mpimg.imread(train_jpeg_dir + '/' + image_name)
        axs[i].imshow(img)
        axs[i].set_title('{} - {}'.format(image_name, label))
    plt.show()
    