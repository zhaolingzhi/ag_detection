from PIL import Image as img
import numpy as np

def get_list(file):
    list=file.readlines()
    for i in range(len(list)):
        list[i]=list[i].strip('\n')
    return list


def read_image(path):
    list=[]
    region=(110,70,710,670)
    for p in path:
        if p.strip():
            image=img.open(p)
            # if image.width == 816 and image.height>670:
            #     image = image.crop(region)
            # else:
            #     region=(int(image.width*110/816),int(image.height*70/816)\
            #             ,int(image.width*710/816),int(image.height*670/816))
            #     image = image.crop(region)
            #     region = (110, 70, 710, 670)
            list.append(np.asarray(image.resize((227,227)),dtype='float64')/256)
    return np.asarray(list)


def saveinfo(file,history):
    np_accy = np.array(history)
    np.savetxt(file, np_accy)

def read_image_flip(path):
    list = []
    region = (110, 70, 710, 670)
    for p in path:
        if p.strip():
            image = img.open(p)

            if image.width == 816 and image.height>670:
                image = image.crop(region)
            else:
                region=(int(image.width*110/816),int(image.height*70/816)\
                        ,int(image.width*710/816),int(image.height*670/816))
                image = image.crop(region)
                region = (110, 70, 710, 670)

            image = image.transpose(img.FLIP_LEFT_RIGHT)
            list.append(np.asarray(image.resize((227, 227)), dtype='float64') / 256)
    return np.asarray(list)
