#encoding:utf-8
import keras
from util import *
picture_path='Folds/aligned/'


def gender_data(path,num=100000,label=0):
    'get the mark and picture as list return,return is image, gender_mark'
    gender_info=open(path,'r')
    gender_list=get_list(gender_info)

    gender_mark=[]
    image_path=[]

    for l in gender_list:
        gender_mark.append(int(l.split()[1]))
        image_path.append(l.split()[0])

    gender_mark = keras.utils.to_categorical(gender_mark, num_classes=2)

    for i in range(len(image_path)):
        image_path[i] = picture_path + image_path[i]

    if len(gender_list)-label*num < num/2 and label > 0:
        return None,None
    else:
        image_path = image_path[label * num:(label + 1) * num]
        gender_mark = gender_mark[label * num:(label + 1) * num]

    image = read_image(image_path)

    print "read "+path+" finish"

    return image, gender_mark


def age_data(path,num=100000,label=0):
    'get the mark and picture as list return,return is image, age_mark'
    age_info=open(path,'r')
    age_list=get_list(age_info)

    age_mark=[]
    image_path=[]

    for l in age_list:
        age_mark.append(int(l.split()[1]))
        image_path.append(l.split()[0])

    age_mark = keras.utils.to_categorical(age_mark, num_classes=8)

    for i in range(len(image_path)):
        image_path[i] = picture_path + image_path[i]

    if len(age_list)-label*num < num/2 and label > 0:
        return None,None
    else:
        image_path = image_path[label * num:(label + 1) * num]
        age_mark = age_mark[label * num:(label + 1) * num]

    image = read_image(image_path)

    print "read "+path+" finish"

    return image, age_mark


