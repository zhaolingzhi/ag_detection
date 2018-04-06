#encoding:utf-8
import numpy as np
import keras
import scipy as sc
from input import *
from net_model import *
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD
from keras import initializers

log_info="log_info.txt"
path="/home/zlz/PycharmProjects/ag_detection/model.h5"
batch_size=50
epochs=100


file=open('//home//zlz//PycharmProjects//ag_detection//Folds//my_image.txt','r')
list=get_list(file)
my_test=read_image(list)
model=net_model('Alexnet-simple-gender')
model.load_weights("model.h5")
result=model.predict(my_test)
print result
