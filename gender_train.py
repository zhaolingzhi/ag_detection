#encoding:utf-8
import time
from input import *
from net_model import *
from util import *
import datetime


log_info="gender_log_info.txt"
result="gender_result.txt"
target_label="gender"

train_image=None
train_gender_mark=None

size_train=24512
batch_size=64

num = int(size_train/batch_size)
epochs=100
lr=0.01
decay=0.0001
name="_gender_model.h5"
netname="Alexnet-simple-gender-4.0"

nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
start_time = time.time()

model=net_model(netname,lr=lr,decay=decay)

val_image, val_gender_mark = gender_data("test_fold_is_0//gender_val.txt")
size_val=len(val_gender_mark)

history = train_generator_model(model,"test_fold_is_0//gender_train.txt",val_image, val_gender_mark,batch_size=batch_size,
                              epochs=epochs,name=nowTime+name,classes=2,num=num)

saveinfo(log_info,history)

del val_image, val_gender_mark
test_image, test_gender_mark=gender_data("test_fold_is_0//gender_test.txt")
size_test=len(test_gender_mark)
score = model.evaluate(test_image,test_gender_mark,batch_size=batch_size)

end_time = time.time()

f=open(result,'a')
f.write(nowTime+" "+target_label+"  "+netname+"   train:"+str(size_train)+" val:"+str(size_val)+" test:"+str(size_test)+" :acc="+str(score[1]) +\
        " batch_size="+str(batch_size)+" epochs="+str(epochs) +\
        " lr="+str(lr)+" decay="+str(decay)+\
        " time:"+str(int(end_time-start_time))+'\n')

print score