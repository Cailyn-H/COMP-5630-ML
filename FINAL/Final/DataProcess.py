import os
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

#only using columns with this names
features = ['mel', 'bkl']

#new directory to copy necessary data
newDataSet = "DataImage"
t_path = "TRAIN"
T_path = "TEST"

#must find the location of datafolder in your computer
path1 = "place location of images_part_1 from the local computer HERE"
path2 = "place location of images_part_2 from the local computer HERE"
img_1 = os.listdir(path1)
img_2 = os.listdir(path2)


#if running this code for the first time, try this under comment block
"""
try:
    os.mkdir(newDataSet)
except:
    pass

try:
    for i in features:
        os.makedirs(os.path.join(os.getcwd(), newDataSet, t_path, i))
except:
    pass

try:
    for i in features:
        os.makedirs(os.path.join(os.getcwd(), newDataSet, T_path, i))
except:
    pass
"""


data = pd.read_csv("metadata.csv")
X = data.drop(data[(data.dx != "mel") & (data.dx != "bkl")].index)
X_t, X_T = train_test_split(X, test_size=0.3, random_state=100, stratify=X.dx)

images = list(X.image_id)
img_TRAIN = list(X_t.image_id)
img_TEST = list(X_t.image_id)
X.set_index('image_id', inplace=True)


for i in images:
    image_ID = i + '.jpg'
    if i in img_TRAIN:
        deep = t_path
    else:
        deep = T_path
    if image_ID in img_1:
        img_dir = os.path.join(path1, image_ID)
    else:
        img_dir = os.path.join(path2, image_ID)
    label = X.loc[i, 'dx']
    destination = os.path.join(os.getcwd(), newDataSet, deep, label, image_ID)
    shutil.copyfile(img_dir, destination)

