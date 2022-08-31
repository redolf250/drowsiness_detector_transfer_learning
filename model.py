import os
import cv2
import pickle
import random
import numpy as np
import pandas as pd 
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


images_directory = 'D:\\Data\\work\\data'

categories = ['Closed_Eyes','Open_Eyes']

data=[]

for category in categories:
    path = os.path.join(images_directory,category)
    label = categories.index(category)
    
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        train_data = cv2.imread(imgpath,0)
        train_data = cv2.resize(train_data,(80,80))
        image = np.array(train_data).flatten()
        data.append([image,label])

pickle_in = open('custom_transformed.pickle','wb')
pickle.dump(data,pickle_in)
pickle_in.close()

pickle_in = open('D:\\Data\\work\\custom_transformed.pickle','rb')
data = pickle.load(pickle_in)
pickle_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

train_x, test_x, train_y,test_y = train_test_split(features,labels, test_size= 0.1)

model = SVC(C=1, kernel= 'poly', gamma= 'auto')
train_model = model.fit(train_x,train_y)

prediction = train_model.predict(test_x)
accuracy = train_model.score(test_x,test_y)

pickle_in = open('D:\\Data\\work\\custom_model.sav','wb')
pickle.dump(train_model,pickle_in)
pickle_in.close()

pickle_in = open('D:\\Data\\work\\custom_model.sav','rb')
loaded_model = pickle.load(pickle_in)
pickle_in.close()

print('Accuracy of the mode is : {}'.format(accuracy)) 

print('Predictions is: {}'.format(categories[prediction[70]]))


eye_state = test_x[70].reshape(80,80)
plt.imshow(eye_state, cmap= 'gray')
print(np.array(test_x[70]).size)
plt.show()

