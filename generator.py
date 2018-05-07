from PIL import Image
import matplotlib.pyplot as plt
from util import *

mark_path='Folds//train_val_txt_files_per_fold//test_fold_is_0//gender_train.txt'
picture_path='Folds/aligned/'

gender_info=open(mark_path)
gender_list=get_list(gender_info)

gender_mark=[]
image_path=[]

for l in gender_list:
      gender_mark.append(l.split()[1])
      image_path.append(l.split()[0])

new_list=[]

for i in range(len(gender_list)):
    print i
    path=image_path[i].strip()
    image=Image.open(picture_path+path)
    image=image.transpose(Image.FLIP_LEFT_RIGHT)
    nameSplit=path.split('.')
    nameSplit[-2]=nameSplit[-2]+'horizontal_flip'
    image_path[i]='.'.join(nameSplit)
    image.save(picture_path+image_path[i])
    new_list.append(image_path[i]+' '+gender_mark[i])

gender_info=open(mark_path,'w')
list=new_list+gender_list
gender_info.writelines(line+'\n' for line in list)


