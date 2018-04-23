#encoding:utf-8

from input import *
from net_model import *
batch_size=50
epochs=200
lr=0.1
decay=0.001


train_image, train_gender_mark=gender_data("test_fold_is_0//gender_train.txt",num=2000)
val_image,val_gender_mark=gender_data("test_fold_is_0//gender_val.txt",num=500)
test_image, test_gender_mark=gender_data("test_fold_is_0//gender_test.txt",num=500)


model=net_model('Alexnet-simple-gender',lr=lr,decay=decay)

train_model(model, train_image, train_gender_mark, val_image, val_gender_mark,
                                    batch_size=batch_size,epochs=epochs)

score = model.evaluate(test_image,test_gender_mark,batch_size=50)

print score
