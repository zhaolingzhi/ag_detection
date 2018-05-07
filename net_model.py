import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Flatten,Dropout,Activation,BatchNormalization
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model
from util import *

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
        plot_model(model,to_file="Alexnet-gender.png",show_shapes=True,show_layer_names=True)
        print "Alexnet-simple-gender model has been made"

        return model
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
        plot_model(model,to_file=name+".png",show_shapes=True,show_layer_names=True)
        print "Alexnet-simple-gender model has been made"

        return model
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
        plot_model(model,to_file=name+".png",show_shapes=True,show_layer_names=True)
        print "Alexnet-simple-gender model has been made"

        return model
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
        model.summary()

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print name+" net has been made"

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
        plot_model(model,to_file=name+".png",show_shapes=True,show_layer_names=True)
        print "Alexnet-simple-gender model has been made"

        return model

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
        model.summary()

        sgd = SGD(lr=lr, decay=decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        print name+" net has been made"

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


def train_generator_model(model,filepath,val_image,val_mark,batch_size,epochs,name,classes,num):

    def generate_arrays_from_file(filepath,batch_size,classes):
        picture_path = 'Folds/aligned/'
        mark_path = 'Folds//train_val_txt_files_per_fold//'

        while 1:
            file = open(mark_path + filepath, 'r')
            filelist = get_list(file)
            count=0
            while True:

                batch_file=filelist[:batch_size]
                filelist=filelist[batch_size:]

                image_path=[]
                image_mark=[]
                for l in batch_file:
                    image_path.append(l.split()[0])
                    image_mark.append(int(l.split()[1]))

                image_mark=keras.utils.to_categorical(image_mark,classes)
                for i in range(len(image_path)):
                    image_path[i] = picture_path + image_path[i]

                images=read_image(image_path)

                yield (images,image_mark)
                count+=1
                if count==num:
                    break

    history = model.fit_generator(generate_arrays_from_file(filepath,batch_size,classes),
                                  steps_per_epoch=num,
                                  epochs=epochs,
                                  verbose=2,
                                  validation_data=(val_image, val_mark),
                                  callbacks=[keras.callbacks.ModelCheckpoint(path + name, monitor='val_loss', verbose=0,
                                                                             save_best_only=True, mode='auto'),
                                             keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0,
                                                                           mode='auto')
                                             ])
    print "train finish"
    return history.history['acc']