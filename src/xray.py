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
#file_path2 = '/Users/astyakghanavatian/Desktop/NIHData/images2/'

# --- get file names for images in order to extract features
for file in glob.glob(file_path + '*.png'):
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
cd_df = pd.DataFrame(cleaned_data)


# --- fill binary matrix with images and their labels
# --- keep track of indices prior to train/test split
labels_matrix = np.zeros((12120,15), dtype=int)
img_index = {}
labels_dict = {}
for i in cleaned_data[:]:
    img_index[cleaned_data.index(i)] = i[0]
    temp = i[1].split("|")
    #print("AFTER SPLIT: ", temp)
    #labels_dict[i[0]] = temp
    for t in temp[:]:
        #print(disease_labels.index(t))
        labels_matrix[cleaned_data.index(i)][disease_labels.index(t)] = 1


labels_df = pd.DataFrame(labels_matrix)

new_df = labels_df[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].rename(
    {0 : 'Atelectasis', 1 : 'Consolidation', 2 : 'Infiltration', 3 : 'Pneumothorax',
     4 : 'Edema', 5 : 'Emphysema', 6 : 'Fibrosis', 7 : 'Effusion',
     8 : 'Pneumonia', 9 : 'Pleural_Thickening', 10 : 'Cardiomegaly',
     11 : 'Mass', 12 : 'Nodule', 13 : 'Hernia', 14 : 'No Finding'}, axis=1)

new_df.insert(0, 'FileNames', pd.Series(cd_df[0]), True)

print(new_df)



X_train, X_test = train_test_split(new_df, test_size=0.20)


#train_data = pd.DataFrame(X_train)
#test_data = pd.DataFrame(X_test)


# --- KERAS SEQUENCE FOR BATCH PROCESSING???????????
datagen = ImageDataGenerator(
    rescale=1./255)

train_generator = datagen.flow_from_dataframe(
        dataframe=X_train,
        directory=file_path,
        x_col='FileNames',
        y_col=disease_labels,
        #target_size=(256, 256),
        shuffle=True,
        batch_size=32,
        class_mode="multi_output")
        #classes = disease_labels

""""
validation_generator = datagen.flow_from_dataframe(
        dataframe=X_test,
        directory=file_path,
        x_col='FileNames',
        #y_col=None,
        #target_size=(150, 150),
        batch_size=32,
        class_mode=None)
"""

test_generator=datagen.flow_from_dataframe(
        dataframe=X_test,
        directory=file_path,
        x_col="FileNames",
        batch_size=1,
        shuffle=False,
        class_mode=None)

"""
model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)
"""


"""


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



"""
for file in glob.glob(file_path2 + '*.png'):
    img_path = file
    file = file.split("/")
    filenames.append(file[len(file)-1])
"""