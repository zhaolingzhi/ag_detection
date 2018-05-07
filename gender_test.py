import time
from input import *
from net_model import *
from util import *
import datetime

lr=0.1
decay=0.001
model_name="/home/zlz/PycharmProjects/ag_detection/2018-05-06 17:18:21_gender_model.h5"
batch_size=64
epochs=200

model=net_model('Alexnet-simple-gender-4.0',lr=lr,decay=decay)
model.load_weights(model_name)
test_image, test_gender_mark = gender_data("test_fold_is_0//gender_test.txt")
score = model.evaluate(test_image,test_gender_mark,batch_size=batch_size)
print score