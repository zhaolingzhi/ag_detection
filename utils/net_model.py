import keras
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,\
                         Dropout,Activation,BatchNormalization,\
                         Input,AveragePooling2D,ZeroPadding2D,add,InputLayer
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from util import *
import random

model_path="/home/zlz/PycharmProjects/ag_detection/models/"
pic_path="/home/zlz/PycharmProjects/ag_detection/pic/"


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def Conv_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def net_model(name,lr=0.1,decay=0.0001):
    model = Sequential()
    if name=='VGG-13-gender-1.0':
        model.add(Conv2D(64, (3, 3), strides=(1, 1), input_shape=(224, 224, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 2), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name=='Alexnet-simple-gender':
        model.add(Conv2D(96, (7, 7), activation='relu', strides=4,padding='same',
                         input_shape=(227, 227, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(256, (5, 5), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(384, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name=='Alexnet-gender':
        model.add(Conv2D(96, (11, 11), activation='relu', strides=4,
                         input_shape=(227, 227, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(256, (5, 5), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(384, (3, 3), activation='relu',padding='same'))

        model.add(Conv2D(384, (3, 3), activation='relu',padding='same'))

        model.add(Conv2D(256, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.5, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-2.0':
        model.add(Conv2D(96, (7, 7), activation='relu', strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-3.0':
        model.add(Conv2D(96, (7, 7), activation='relu', strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-4.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-5.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Dropout(0.5))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Dropout(0.5))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-6.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-7.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-8.0':
        model.add(Conv2D(96, (5, 5), strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(384, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-9.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(512,(7,7),padding='valid'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-10.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='same'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Conv2D(512,(7,7),padding='valid'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-11.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='same'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-12.0':
        model.add(InputLayer(input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='same'))

        model.add(BatchNormalization(momentum=0.9))
        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(BatchNormalization(momentum=0.9))
        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Flatten())
        model.add(BatchNormalization(momentum=0.9))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(BatchNormalization(momentum=0.9))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(BatchNormalization(momentum=0.9))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-13.0':
        model.add(InputLayer(input_shape=(227, 227, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='same'))

        model.add(BatchNormalization())
        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(BatchNormalization())
        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(512))

        model.add(BatchNormalization())
        model.add(Dense(512))

        model.add(BatchNormalization())
        model.add(Dense(2,activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-14.0':
        model.add(InputLayer(input_shape=(227, 227, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(BatchNormalization())
        model.add(Conv2D(256, (5, 5), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(BatchNormalization())
        model.add(Conv2D(384, (3, 3), padding='same',activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(512,activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(512,activation='relu'))

        model.add(BatchNormalization())
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-15.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='same'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-gender-16.0':
        model.add(Conv2D(96, (7, 7), activation='relu', strides=4, padding='valid',
                         input_shape=(227, 227, 3)))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'ResNet-gender-1.0':

        inpt = Input(shape=(224, 224, 3))
        x = ZeroPadding2D((3, 3))(inpt)
        x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(2, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        sgd = SGD(decay=0.0001, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    elif name == 'ResNet-gender-2.0':

        inpt = Input(shape=(224, 224, 3))
        x = ZeroPadding2D((3, 3))(inpt)
        x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=64, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=128, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))
        x = Conv_Block(x, nb_filter=256, kernel_size=(3, 3))

        x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
        x = Conv_Block(x, nb_filter=512, kernel_size=(3, 3))
        x = AveragePooling2D(pool_size=(7, 7))(x)
        x = Flatten()(x)
        x = Dense(2, activation='softmax')(x)

        model = Model(inputs=inpt, outputs=x)
        sgd = SGD(decay=0.0001, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name=='Alexnet-simple-age':
        model.add(Conv2D(96, (7, 7), activation='relu', strides=4,padding='same',
                         input_shape=(227, 227, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(256, (5, 5), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(384, (3, 3), activation='relu',padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation='softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-age-2.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='same',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(8))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-age-3.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2,padding='same'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(8))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('softmax'))

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    elif name == 'Alexnet-simple-age-4.0':
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid',
                         input_shape=(227, 227, 3)))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(256))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(8))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('softmax'))
        model.summary()
        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    elif name == 'Alexnet-simple-age-5.0':
        model.add(InputLayer(input_shape=(227, 227, 3)))
        model.add(BatchNormalization())
        model.add(Conv2D(96, (7, 7), strides=4, padding='valid'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(BatchNormalization())
        model.add(Conv2D(256, (5, 5), padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(BatchNormalization())
        model.add(Conv2D(384, (3, 3), padding='same'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=2, padding='same'))

        model.add(Flatten())
        model.add(BatchNormalization())
        model.add(Dense(512))

        model.add(BatchNormalization())
        model.add(Dense(512))

        model.add(BatchNormalization())
        model.add(Dense(8, activation='softmax'))
#        model.summary()
        plot_model(model, to_file=name + ".png", show_shapes=True, show_layer_names=True)
        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print name + " net has been made"

    model.summary()
    print name + " net has been made"
    plot_model(model, to_file=pic_path + name + ".png", show_shapes=True, show_layer_names=True)
    return model


def train_model(model,train_image,train_mark,val_image,val_mark,batch_size,epochs,name):
    history = model.fit(train_image, train_mark, batch_size=batch_size, epochs=epochs,
                        verbose=2,
                        validation_data=(val_image, val_mark),
                        callbacks=[keras.callbacks.ModelCheckpoint(path+name, monitor='val_loss', verbose=0,
                                                                   save_best_only=True, mode='auto'),
                                   keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
                                   ]
                        )
    print "train finish"
    return history.history['acc']


def train_generator_model(model,filepath,val_image,val_mark,batch_size,epochs,name,classes,num,isFlip):

    def generate_arrays_from_file(filepath,batch_size,classes,num,isFlip):
        picture_path = 'Folds/aligned/'

        while 1:
            file = open(filepath, 'r')
            filelist = get_list(file)
            count=0
            flip=False
            copy=filelist[:]
            while True:

                batch_file=random.sample(filelist,batch_size)
                for item in batch_file:
                    filelist.remove(item)
                image_path=[]
                image_mark=[]
                for l in batch_file:
                    image_path.append(l.split()[0])
                    image_mark.append(int(l.split()[1]))

                image_mark=keras.utils.to_categorical(image_mark,classes)
                for i in range(len(image_path)):
                    image_path[i] = picture_path + image_path[i]
                if flip is False:
                    images=read_image(image_path)
                else:
                    images=read_image_flip(image_path)

                yield (images,image_mark)
                count += 1
                if count == num:
                    if isFlip is False:
                        break
                    else:
                        if flip is False:
                            filelist = copy[:]
                            flip = True
                            count = 0
                        else:
                            break

    history = model.fit_generator(generate_arrays_from_file(filepath,batch_size,classes,num,isFlip),
                                  steps_per_epoch=num if isFlip is False else num*2,
                                  epochs=epochs,
                                  verbose=2,
                                  validation_data=(val_image, val_mark),
                                  callbacks=[keras.callbacks.ModelCheckpoint(model_path + name, monitor='val_loss', verbose=0,
                                                                             save_best_only=True, mode='auto'),
                                             keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0,
                                                                           mode='auto')
                                             ])
    print "train finish"
    return history.history['acc']