# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import cv2
import csv
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import GridSearchCV

nb_img = 400
img_rows = 121
img_cols = 121


def get_data():
    subjects = pd.read_csv("data/train/subjects.tsv", sep='\t')
    X = []
    y = []

    for img_path in glob.glob("data/train/*.png"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_rows, img_cols))
        
        img_path = os.path.basename(img_path) # pour avoir les bons /
        img_name = img_path.split('/')[-1][:-4]

        label = subjects.loc[subjects['Subject'] == img_name, 'Label'].values[0]

        X.append(img)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    X = X.reshape(X.shape[0], img_rows, img_cols, 3)
    X = X.astype('float32') / 255.0 # normalization

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    
    # plt.figure(figsize=(10,10))
    # for i in range(25):
    #     plt.subplot(5,5,i+1)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.grid(False)
    #     plt.imshow(X_train[i])
    #     plt.xlabel(y_train[i])
    # plt.show()

    return X_train, X_test, y_train, y_test


def get_model_cnn():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2))) 
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))  # classification binaire
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

def get_model_vgg16():
    vgg = VGG16(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

    model = models.Sequential()
    model.add(vgg)
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

def get_model_resnet50():
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(img_rows, img_cols, 3))

    model = models.Sequential()
    model.add(resnet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return model

def train(batch_size, num_epoch, model):
    X_train, X_test, y_train, y_test = get_data()

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)

    model.fit(X_train, y_train, batch_size=batch_size, epochs=num_epoch, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # plot_learning_curves(history)

    return model

def plot_learning_curves(history):
    # accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

def validate(model, model_name):
    """write the output file"""

    X_val = []
    subject_names = []


    for img_path in glob.glob("data/validation/*.png"):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (img_rows, img_cols))

        img_name = os.path.basename(img_path)[:-4] 
        subject_names.append(img_name)

        X_val.append(img)

    X_val = np.array(X_val)
    X_val = X_val.reshape(X_val.shape[0], img_rows, img_cols, 3)
    X_val = X_val.astype('float32') / 255.0  # normalization


    predictions = model.predict(X_val)
    #print(np.unique(predictions))

    binary_predictions = (predictions > 0.5).astype(int)

    output_df = pd.DataFrame({'Subject': subject_names, 'Label': binary_predictions.flatten()})

    output_df.to_csv("src/output_"+ model_name + ".tsv", sep='\t', index=False)
    
    # X_val = []
    # Y_val = []
    # val = []
    

def main() -> None:

    num_epoch = 5
    batch_size = 32

    model_cnn = train(batch_size, num_epoch, get_model_cnn())
    validate(model_cnn, "cnn")

    model_vgg16 = train(batch_size, num_epoch, get_model_vgg16())
    validate(model_vgg16, "vgg16")

    model_resnet50 = train(batch_size, num_epoch, get_model_resnet50())
    validate(model_resnet50, "resnet50")


if __name__ == "__main__":
    main()