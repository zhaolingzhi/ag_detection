import time
from input import *
from net_model import *
from util import *
import datetime

lr=0.1
decay=0.0001
model_name="/home/zlz/PycharmProjects/ag_detection/2018-04-22 19:32:55_age_model.h5"
num=4000
batch_size=50
epochs=200

model=net_model('Alexnet-simple-gender',lr=lr,decay=decay)
model.load_weights(model_name)
test_image, test_gender_mark = age_data("test_fold_is_0//gender_test.txt")
score = model.evaluate(test_image,test_gender_mark,batch_size=batch_size)
print score