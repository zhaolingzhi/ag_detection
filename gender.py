#encoding:utf-8

from input import *
from net_model import *

train_image, train_gender_mark=gender_data("test_fold_is_0//gender_train.txt",num=4000)
val_image,val_gender_mark=gender_data("test_fold_is_0//gender_val.txt",num=1000)
test_image, test_gender_mark=gender_data("test_fold_is_0//gender_test.txt",num=1000)


model=net_model('Alexnet-simple-gender')

train_model(model,train_image,train_gender_mark,val_image,val_gender_mark)

score = model.evaluate(test_image,test_gender_mark,batch_size=50)

print score
