#encoding:utf-8
import time
from utils.input import *
from utils.net_model import *
from utils.util import *
import datetime

dir="/home/zlz/PycharmProjects/ag_detection/Folds/train_val_txt_files_per_fold/test_fold_is_3/"
train_file="age_train.txt"

log_info="age_log_info.txt"
result="age_result.txt"
target_label="age"

train_image=None
train_age_mark=None

size_train=len(open(dir+train_file,'r').readlines())
batch_size=64
num = int(size_train/batch_size)
isFlip=True
epochs=100
lr=0.001
decay=0.00001
name="_age_model.h5"
net_name='Alexnet-simple-age-3.0'

nowTime=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print "\n\n==============================="
print "==============================="
print nowTime
print "==============================="
start_time = time.time()

model=net_model(net_name,lr=lr,decay=decay)

val_image, val_age_mark = age_data(dir+"age_val.txt")
size_val=len(val_age_mark)

history = train_generator_model(model,dir+train_file,val_image, val_age_mark,batch_size=batch_size,
                              epochs=epochs,name="age/"+nowTime+name,classes=8,num=num,isFlip=isFlip)


saveinfo(log_info,history)

del val_image, val_age_mark
test_image, test_age_mark = age_data(dir+"age_test.txt")
size_test=len(test_age_mark)
score = model.evaluate(test_image,test_age_mark,batch_size=batch_size,verbose=0)

end_time = time.time()

f=open(result,'a')
f.write(dir+" "+nowTime+" "+target_label+"  "+net_name+"   train:"+str(size_train if isFlip is False else size_train*2)+" val:"+str(size_val)+" test:"+str(size_test)+" :acc="+str(score[1]) +\
        " batch_size="+str(batch_size)+" epochs="+str(epochs) +\
        " lr="+str(lr)+" decay="+str(decay)+\
        " time:"+str(int(end_time-start_time))+'\n')

print score
