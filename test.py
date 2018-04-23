#encoding:utf-8
from input import *
from net_model import *

log_info="log_info.txt"
path="/home/zlz/PycharmProjects/ag_detection/model.h5"
batch_size=50
epochs=100

file=open('//home//zlz//PycharmProjects//ag_detection//Folds//my_image.txt','r')
list=get_list(file)
my_test=read_image(list)
model=net_model('Alexnet-simple-gender')
model.load_weights("model_2.h5")
result=model.predict(my_test)
print result
