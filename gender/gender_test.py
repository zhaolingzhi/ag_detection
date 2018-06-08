from utils.input import *
from utils.net_model import *

lr=0.1
decay=0.001
model_name="/home/zlz/PycharmProjects/ag_detection/2018-05-19 19:18:34_gender_model.h5"
dir="/home/zlz/PycharmProjects/ag_detection/Folds/train_val_txt_files_per_fold/test_fold_is_0/"
batch_size=64
epochs=200

model=net_model('Alexnet-simple-gender-11.0',lr=lr,decay=decay)
model.load_weights(model_name)
test_image, test_gender_mark = gender_data(dir+"gender_test.txt")
result = model.evaluate(test_image,test_gender_mark,batch_size=batch_size)
print result

