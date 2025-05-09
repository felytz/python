# -*- coding: utf-8 -*-
"""Clasificación Flores_Iris.ipynb"""

import numpy as np
import pandas as pd
import scipy as sc
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

"""##Binary Cross Entropy Loss"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.utils import to_categorical

import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

"""##Dataset"""

iris = datasets.load_iris()

iris

X = iris.data
y = iris.target

names = iris['target_names']
feature_names = iris['feature_names']

feature_names

# One hot encoding
enc = OneHotEncoder()
Y = enc.fit_transform(y[:, np.newaxis]).toarray()

X.shape, Y.shape

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

X_train.shape, X_test.shape

# Visualize the data sets
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
for target, target_name in enumerate(names):
    X_plot = X[y == target]
    plt.plot(X_plot[:, 0], X_plot[:, 1], linestyle='none', marker='o', label=target_name)
plt.xlabel(feature_names[0])
plt.ylabel(feature_names[1])
plt.axis('equal')
plt.legend();

plt.subplot(1, 2, 2)
for target, target_name in enumerate(names):
    X_plot = X[y == target]
    plt.plot(X_plot[:, 2], X_plot[:, 3], linestyle='none', marker='o', label=target_name)
plt.xlabel(feature_names[2])
plt.ylabel(feature_names[3])
plt.axis('equal')
plt.legend();

"""##Modelo"""

classifier = Sequential()
classifier.add(Input(shape=(X_train.shape[1],)))
classifier.add(Dense(8, activation = 'relu'))

#output layer with 1 output neuron which will predict 1 or 0
classifier.add( Dense( 3, activation = 'softmax' ) )

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

"""###Entrenamiento"""

# Commented out IPython magic to ensure Python compatibility.
historial = classifier.fit(X_train, y_train, validation_split=0.2, epochs = 300, verbose = 0)

score = classifier.evaluate(X_test, y_test, verbose=0)
score

plt.plot(historial.epoch,historial.history['loss'], 'b',label="loss")
plt.plot(historial.epoch,historial.history['val_loss'], '*r',label="val_loss")
plt.title(u'MSE')
plt.xlabel(u'época')
plt.ylabel(r'$Loss(\omega, b)$')
plt.legend(loc='upper right')
plt.ylim([np.min(historial.history['loss'])-0.1,np.max(historial.history['loss'])+0.1])
plt.grid(True)
plt.savefig('iris.png')
plt.show()

prediccion = classifier.predict(X_test)

y_pred = np.argmax(prediccion, axis=1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

con = confusion_matrix( np.argmax(y_test, axis=1) , y_pred )

disp = ConfusionMatrixDisplay( confusion_matrix = con,  display_labels = iris.target_names  ).plot()
plt.show()

print('.py finished running')