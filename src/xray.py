import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import ImageDataGenerator
import glob
import csv
import re
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

filenames = []
file_path = '/Users/astyakghanavatian/Desktop/NIHData/images/'
file_path2 = '/Users/astyakghanavatian/Desktop/NIHData/images2/'

# --- get file names for images in order to extract features
for file in glob.glob(file_path + '*.png'):
    img_path = file
    file = file.split("/")
    filenames.append(file[len(file)-1])

for file in glob.glob(file_path2 + '*.png'):
    img_path = file
    file = file.split("/")
    filenames.append(file[len(file)-1])


# --- get data features for all images
with open('Data_Entry_2017.csv') as file:
    data = file.read().splitlines()


cleaned_data = []
for d in data[:]:
    temp = d.split(",")
    if(filenames.__contains__(temp[0])):
        cleaned_data.append(temp)


# --- disease label names mapped to indices in a list
disease_labels = ['Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax', 'Edema',
        'Emphysema', 'Fibrosis', 'Effusion','Pneumonia', 'Pleural_Thickening', 'Cardiomegaly',
        'Mass', 'Nodule', 'Hernia', 'No Finding']


#print(cleaned_data[1][1])
print(pd.DataFrame(cleaned_data))


# --- fill binary matrix with images and their labels
# --- keep track of indices prior to train/test split
labels_matrix = np.zeros((12120,15), dtype=int)
img_index = {}
for i in cleaned_data[:]:
    img_index[cleaned_data.index(i)] = i[0]
    temp = i[1].split("|")
    #print("AFTER SPLIT: ", temp)
    for t in temp[:]:
        #print(disease_labels.index(t))
        labels_matrix[cleaned_data.index(i)][disease_labels.index(t)] = 1


print(pd.DataFrame(labels_matrix))

print(img_index[0])


X_train, X_test = train_test_split(cleaned_data, test_size=0.20)


train_data = pd.DataFrame(X_train)
test_data = pd.DataFrame(X_test)


# --- KERAS SEQUENCE FOR BATCH PROCESSING???????????
"""
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
"""


#print("TRAIN SPLIT: \n")
#print(pd.DataFrame(X_train))

#print("TEST SPLIT: \n")
#print(pd.DataFrame(X_test))


"""
kf = KFold(n_splits=5)

indices = kf.split(cleaned_data)
print(type(indices))


# --- retrieve train and test splits to run in Neural Network
for train_index, test_index in indices:
    print("TRAIN:", train_index, "TEST:", test_index)
    #X_train, X_test = cleaned_data[train_index], cleaned_data[test_index]

train_data = pd.DataFrame()
"""