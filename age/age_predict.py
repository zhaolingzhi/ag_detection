from utils.input import *
from utils.net_model import *

lr=0.1
decay=0.001
model_name="/home/zlz/PycharmProjects/ag_detection/age/age_model.h5"


model=net_model('Alexnet-simple-age-3.0',lr=lr,decay=decay)
model.load_weights(model_name)

file=open("/home/zlz/PycharmProjects/ag_detection/Folds/my_image.txt",'r')
image_path=get_list(file)

for i in range(len(image_path)):
    image_path[i] = picture_path="/home/zlz/PycharmProjects/ag_detection/" + image_path[i]

image = read_image(image_path)
result = model.predict(image)
print result

