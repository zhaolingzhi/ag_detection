#encoding:utf-8
import time
from input import *
from net_model import *
from util import *

log_info="log_info.txt"
result="result.txt"

train_image=None
train_gender_mark=None
label=0
size=0
history=[]

num=500
batch_size=100
epochs=200
lr=0.1
decay=0.001

start_time = time.time()

model=net_model('Alexnet-simple-gender',lr=lr,decay=decay)

val_image, val_gender_mark = gender_data("test_fold_is_0//gender_val.txt")

while True:
    if train_image is not None:
        del train_image, train_gender_mark
    train_image, train_gender_mark = gender_data("test_fold_is_0//gender_train.txt", num=num, label=label)
    if train_image is False:
        break
    label = label + 1
    size=size+len(train_gender_mark)
    history = history + train_model(model, train_image, train_gender_mark, val_image, val_gender_mark,
                                    batch_size=batch_size,epochs=epochs)

saveinfo(log_info,history)

del train_image, train_gender_mark, val_image, val_gender_mark
test_image, test_gender_mark=gender_data("test_fold_is_0//gender_test.txt")
score = model.evaluate(test_image,test_gender_mark,batch_size=50)

end_time = time.time()

f=open(result,'a')
f.write("train:"+str(size)+" val:"+str(len(val_gender_mark))+" test:"+str(len(test_gender_mark))+" :acc="+score[1] +\
        " batch_size="+str(batch_size)+" epochs="+str(epochs) +\
        " lr="+str(lr)+" decay="+str(decay)+\
        " time:"+str(int(end_time-start_time))+'\n')
