#encoding:utf-8
import time
from input import *
from net_model import *
from util import *
import datetime

log_info="age_log_info.txt"
result="age_result.txt"
target_label="age"

train_image=None
train_age_mark=None

size_train=11823
batch_size=64
num = int(size_train/batch_size)

epochs=100
lr=0.1
decay=0.001
name="_age_model.h5"

nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
start_time = time.time()

model=net_model('Alexnet-simple-age-2.0',lr=lr,decay=decay)

val_image, val_age_mark = age_data("test_fold_is_0//age_val.txt")
size_val=len(val_age_mark)

history = train_generator_model(model,"test_fold_is_0//age_train.txt",val_image, val_age_mark,batch_size=batch_size,
                              epochs=epochs,name=nowTime+name,classes=8,num=num)


saveinfo(log_info,history)

del val_image, val_age_mark
test_image, test_age_mark = age_data("test_fold_is_0//age_test.txt")
size_test=len(test_age_mark)
score = model.evaluate(test_image,test_age_mark,batch_size=batch_size)

end_time = time.time()

f=open(result,'a')
f.write(nowTime+" "+target_label+"   train:"+str(size_train)+" val:"+str(size_val)+" test:"+str(size_test)+" :acc="+str(score[1]) +\
        " batch_size="+str(batch_size)+" epochs="+str(epochs) +\
        " lr="+str(lr)+" decay="+str(decay)+\
        " time:"+str(int(end_time-start_time))+'\n')

print score
