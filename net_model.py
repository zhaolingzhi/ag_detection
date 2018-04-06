from keras.models import Sequential
import keras
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Dense,Flatten,Dropout
from keras.optimizers import SGD
from keras.utils.vis_utils import plot_model

def net_model(name,model):
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
        sgd = SGD(lr=1e-3, decay=9e-8, momentum=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        plot_model(model,to_file="VGG-like.png",show_shapes=True,show_layer_names=True)
        print "VGG-like model has been made"

        return model
    elif name=='Alexnet-simple-gender':
        model.add(Conv2D(96, (7, 7), activation='relu', strides=4,
                         kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                         input_shape=(227, 227, 3)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(256, (5, 5), activation='relu',
                         kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Conv2D(384, (3, 3), activation='relu',
                         kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        sgd = SGD(lr=1e-3, decay=9e-8, momentum=0.0, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        plot_model(model,to_file="Alexnet-simple-gender.png",show_shapes=True,show_layer_names=True)
        print "Alexnet-simple-gender model has been made"

        return model
