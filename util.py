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
        if p.strip():
            image=img.open(p)
            list.append(np.asarray(image.resize((227,227)),dtype='float64')/256)
    return np.asarray(list)


def saveinfo(file,history):
    np_accy = np.array(history)
    np.savetxt(file, np_accy)

def read_image_flip(path):
    list = []
    for p in path:
        if p.strip():
            image = img.open(p)
            image = image.transpose(img.FLIP_LEFT_RIGHT)
            list.append(np.asarray(image.resize((227, 227)), dtype='float64') / 256)
    return np.asarray(list)
