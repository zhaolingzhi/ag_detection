from PIL import Image as img
import numpy as np

def get_list(file):
    list=file.readlines()
    for i in range(len(list)):
        list[i]=list[i].strip('\n')
    return list


def read_image(path):
    list=[]
    for p in path:
        image=img.open(p)
        list.append(np.asarray(image.resize((227,227)),dtype='float64')/256)
    return np.asarray(list)


def func(x,y):
    if y==0:
        return x
    else:
        return func(y,x%y)


