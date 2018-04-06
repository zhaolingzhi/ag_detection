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
epochs=10


def train_model():
    history = model.fit(train_image, train_gender_mark, batch_size=batch_size, epochs=epochs,
                        validation_data=(test_image, test_gender_mark),
                        callbacks=[keras.callbacks.ModelCheckpoint(path, monitor='val_loss', verbose=0,
                                                                   save_best_only=True, mode='auto')]
                        )
    accy = history.history['acc']
    np_accy = np.array(accy)
    np.savetxt(log_info, np_accy)
    print "train finish"


train_image, train_gender_mark=gender_data("test_fold_is_0//gender_train.txt",num=2000)
test_image, test_gender_mark=gender_data("test_fold_is_0//gender_train.txt",num=500)


model=Sequential()
model=net_model('Alexnet-simple-gender',model)

train_model()

score = model.evaluate(test_image,test_gender_mark,batch_size=50)

print score
