import numpy as np
import pandas as pd
import math
import tensorflow as tf
from keras import Sequential, Model, Input
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras import regularizers, optimizers
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, Activation
from keras.preprocessing.image import ImageDataGenerator
import glob
import csv
import re
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from timeit import default_timer as timer
import sys
np.set_printoptions(threshold=sys.maxsize)

start = timer()

filenames = []
file_path = '/Users/astyakghanavatian/Desktop/NIHData/images/'
#file_path = '/Users/astyakghanavatian/Desktop/NIHData/images2/'



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

#labels_matrix = np.zeros((12120,15), dtype=int)
labels_matrix = np.zeros((len(filenames),15), dtype=int)

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

#print(new_df)



X_train, X_test = train_test_split(new_df, test_size=0.20)


#train_data = pd.DataFrame(X_train)
#test_data = pd.DataFrame(X_test)


# --- KERAS SEQUENCE FOR BATCH PROCESSING???????????
datagen = ImageDataGenerator(
    rescale=1./255)

batches = 4

train_generator = datagen.flow_from_dataframe(
        dataframe=X_train,
        directory=file_path,
        x_col='FileNames',
        y_col=disease_labels,
        target_size=(250, 250),
        shuffle=True,
        batch_size=batches,
        class_mode="multi_output")
        #classes = disease_labels


validation_generator = datagen.flow_from_dataframe(
        dataframe=X_test,
        directory=file_path,
        x_col='FileNames',
        #y_col=None,
        target_size=(250, 250),
        batch_size=batches,
        class_mode=None)


test_generator=datagen.flow_from_dataframe(
        dataframe=X_test,
        directory=file_path,
        x_col="FileNames",
        target_size=(250,250),
        batch_size=1,
        shuffle=False,
        class_mode=None)


# --- CNN implementation
input = Input(shape = (250,250,1))
x = Conv2D(32, (3, 3), padding = 'same')(input)
x = Activation('relu')(x)
x = Conv2D(32, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(64, (3, 3), padding = 'same')(x)
x = Activation('relu')(x)
x = Conv2D(64, (3, 3))(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size = (2, 2))(x)
x = Dropout(0.25)(x)
x = Flatten()(x)
x = Dense(512)(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)
out0 = Dense(1, activation = 'sigmoid')(x)
out1 = Dense(1, activation = 'sigmoid')(x)
out2 = Dense(1, activation = 'sigmoid')(x)
out3 = Dense(1, activation = 'sigmoid')(x)
out4 = Dense(1, activation = 'sigmoid')(x)
out5 = Dense(1, activation = 'sigmoid')(x)
out6 = Dense(1, activation = 'sigmoid')(x)
out7 = Dense(1, activation = 'sigmoid')(x)
out8 = Dense(1, activation = 'sigmoid')(x)
out9 = Dense(1, activation = 'sigmoid')(x)
out10 = Dense(1, activation = 'sigmoid')(x)
out11 = Dense(1, activation = 'sigmoid')(x)
out12 = Dense(1, activation = 'sigmoid')(x)
out13 = Dense(1, activation = 'sigmoid')(x)
out14 = Dense(1, activation = 'sigmoid')(x)

out_list = [out0, out1, out2, out3, out4, out5,
            out6, out7, out8, out9, out10, out11,
            out12, out13, out14]
model = Model(input,out_list)


model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="binary_crossentropy",metrics=["accuracy"])

print('Fitting model ...\n')
model.fit_generator(generator=train_generator, steps_per_epoch=math.ceil(len(filenames)/batches), epochs=1,
                    use_multiprocessing=True)

print('Making predictions ...\n')
predict = model.predict_generator(test_generator)

print('PREDICTION: \n')
print("Length of predict element 0: ", len(predict[0]))
print(pd.DataFrame(predict[0]))
print("Length of predict list: ", len(predict))

test_labels = []
for i in predict[:]:
    test_labels.append((i > 0.4).astype(np.int))



X_test = X_test.drop(columns="FileNames")
X_test = X_test.transpose()
test_predictions = np.array(test_labels)
ground_truth = X_test.to_numpy()


print(test_predictions.shape)
print(ground_truth.shape)

ftp = 0
ftn = 0
ffp = 0
ffn = 0


for i in range(len(test_labels)):
    tn, fp, fn, tp = confusion_matrix(test_labels[i], ground_truth[i], labels=[0,1]).ravel()
    ftp += tp
    ftn += tn
    ffp += fp
    ffn += fn


# --- calculate F1 Score
precision = ftp / (ftp + ffp)
recall = ftp / (ftp + ffn)
f1_score = 2 * ((precision*recall) / (precision+recall))

print("\n")
print("tp: ", ftp)
print("tn: ", ftn)
print("fp: ", ffp)
print("fn: ", ffn)
print("F1 Score: ", f1_score)


end = timer()
print("Total run-time: %.2f seconds" % (end - start))










"""

.reshape(X_test.shape)
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


"""
model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800)



model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(None, None, 3)),
    Flatten(),
    Dense(len(disease_labels), activation='softmax')
]
)



loss_ary = ["categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"
            , "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy",
            "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy", 
            "categorical_crossentropy", "categorical_crossentropy", "categorical_crossentropy"]
"""



"""
# Create CNN model
# Will use a combination of convolutional, max pooling, and dropout layers for this purpose
model = Model(outputs=disease_labels)

model.add(Conv2D(filters = 8, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (250, 250, 3)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 16, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))

model.add(Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = 3))
model.add(Dropout(0.2))

# add in fully connected dense layers to model, then output classifiction probabilities using a softmax activation function
model.add(Flatten())
model.add(Dense(500, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(len(disease_labels), activation = 'sigmoid'))
"""