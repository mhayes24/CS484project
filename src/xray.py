import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import glob
import csv
import re

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
labels_matrix = np.zeros((12120,15), dtype=int)
for i in cleaned_data[:]:
    temp = i[1].split("|")
    #print("AFTER SPLIT: ", temp)
    for t in temp[:]:
        #print(disease_labels.index(t))
        labels_matrix[cleaned_data.index(i)][disease_labels.index(t)] = 1


print(pd.DataFrame(labels_matrix))
