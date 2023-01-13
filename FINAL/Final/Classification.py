import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import Flatten,Dense

features = ['mel','bkl']
location = "DataImage"
t_path="TRAIN"
T_path="TEST"

def img_reshape(img):
    return img/255


#train and test
X_t=[]
Y_t=[]
X_T=[]
Y_T=[]

for i in features:
    path = os.path.join(os.getcwd(),location,t_path,i)
    f = os.listdir(path)
    for imgs in f:
        show_img = cv2.imread(os.path.join(path,imgs))
        show_img=cv2.resize(show_img,(28,28))
        X_t.append(show_img)
        Y_t.append(i)


X_t_arr = np.array(X_t, ndmin=2)
Y_t_arr = np.array(Y_t)

#(2030, 28, 28, 3)
print(X_t_arr.shape)
print(Y_t_arr.shape)



for i in features:
    path=os.path.join(os.getcwd(),location,T_path,i)
    folder = os.listdir(path)
    for imgs in folder:
        show_img = cv2.imread(os.path.join(path,imgs))
        show_img = cv2.resize(show_img, (28,28), interpolation=cv2.INTER_AREA)
        X_T.append(show_img)
        Y_T.append(i)


X_T_arr = np.array(X_T)
Y_T_arr = np.array(Y_T)
#pandas.get_dummies() converts categorical data into dummy or indicator variables.
Y_T_arr = pd.get_dummies(Y_t_arr)
Y_t_arr = pd.get_dummies(Y_T_arr)


X_T_arr = np.array(list(map(img_reshape, X_t_arr)))
X_t_arr = np.array(list(map(img_reshape, X_T_arr)))

#adding layers to the model
mlp = Sequential()
mlp.add(Dense(8))
mlp.add(Dense(8))
mlp.add(Conv2D(50,(3,3), input_shape=(28,28,3))) #total of 50 filters
mlp.add(MaxPooling2D(2,2)) #maxpooling is to reduce spatial dimensions of the output volume
mlp.add(Conv2D(50,(3,3)))
mlp.add(MaxPooling2D(2,2))
mlp.add(Flatten())
mlp.add(Dense(512,activation='sigmoid'))
mlp.add(Dense(2,activation='softmax'))
mlp.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])



history = mlp.fit(X_t_arr, Y_t_arr, verbose=1,batch_size=128, epochs=250, shuffle='True', validation_data=(X_t_arr, Y_t_arr))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('ACCURACY')
plt.xlabel('epochs')
plt.legend(['training','validation'])
plt.show()



accuracy = mlp.evaluate(X_T_arr,Y_T_arr)
print("test accuracy: {}".format(accuracy[1]))