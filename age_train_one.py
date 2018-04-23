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
label=0
size_train=0
history=[]

num=4000
batch_size=50
epochs=200
lr=0.1
decay=0.0001
name="_age_model.h5"

nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
start_time = time.time()

model=net_model('Alexnet-simple-age',lr=lr,decay=decay)

val_image, val_age_mark = age_data("test_fold_is_0//age_val.txt")
size_val=len(val_age_mark)

train_image, train_age_mark = age_data("test_fold_is_0//age_train.txt", num=num, label=label)
size_train=size_train+len(train_age_mark)
history = history + train_model(model, train_image, train_age_mark, val_image, val_age_mark,
                                    batch_size=batch_size,epochs=epochs,name=nowTime+name)

saveinfo(log_info,history)

del train_image, train_age_mark, val_image, val_age_mark
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
