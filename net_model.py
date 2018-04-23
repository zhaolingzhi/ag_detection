import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

path="/home/zlz/PycharmProjects/ag_detection/"


def net_model(name,lr=0.1,decay=0.0001):
    model = Sequential()
    if name=='VGG-like':
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        sgd = SGD(lr=lr, decay=decay, momentum=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        plot_model(model,to_file="VGG-like.png",show_shapes=True,show_layer_names=True)
        print "VGG-like model has been made"

        return model
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
        #plot_model(model,to_file="Alexnet-simple-gender.png",show_shapes=True,show_layer_names=True)
        print "Alexnet-simple-gender model has been made"

        return model
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
        plot_model(model,to_file="Alexnet-simple-gender.png",show_shapes=True,show_layer_names=True)
        print "Alexnet-simple-gender model has been made"

        return model


def train_model(model,train_image,train_mark,val_image,val_mark,batch_size,epochs,name):
    history = model.fit(train_image, train_mark, batch_size=batch_size, epochs=epochs,
                        verbose=2,
                        validation_data=(val_image, val_mark),
                        callbacks=[keras.callbacks.ModelCheckpoint(path+name, monitor='val_loss', verbose=0,
                                                                   save_best_only=True, mode='auto'),
                                   keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='auto')
                                   ]
                        )
    print "train finish"
    return history.history['acc']
