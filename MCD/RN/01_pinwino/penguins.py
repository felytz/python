# -*- coding: utf-8 -*-
"""# EJERCICIO [Pinguinos](https://github.com/mwaskom/seaborn-data/blob/master/penguins.csv)"""

import seaborn as sns

df = sns.load_dataset("penguins")

df.head()

print(df['species'].unique())

sns.pairplot(df, hue="species")

#1 Limpieza del dataset

df.describe()

df.isna().sum()

df[df.isna().any(axis=1)]

"""## Modificacion"""

df_c=df.copy()

nans=df_c[df_c.isna().any(axis=1)].index
nans

df_c.iloc[[nans[0],nans[-1]]]

df_c=df_c.drop([nans[0], nans[-1]])
nans = nans[1:-1]

df_c['sex'].describe()

"""168 Male, 165 Female"""

df_c[df_c.isna().any(axis=1)]

for i in nans[::2]: # Every other row starting at 0
  df_c.loc[i,'sex']='Male'
for i in nans[1::2]: # Every other row starting at 1
  df_c.loc[i,'sex']='Female'

df_c[df_c.isna().any(axis=1)]

df_c

df_c=df_c.reset_index()

df_c = df_c.drop('index', axis=1)
df_c.head()

"""#2 Preprocesamiento"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

for i in ['species','sex','island']:
  codificador = OneHotEncoder()
  codificacion = codificador.fit_transform(df_c[[i]])
  new_col_names = [cat.lower() for cat in codificador.categories_[0]]
  new_col = pd.DataFrame(codificacion.toarray(), columns=new_col_names)
  df_c = pd.concat([df_c, new_col], axis="columns")
  df_c=df_c.drop([i],axis=1)

df_c.head()

species=["adelie", "chinstrap", "gentoo"]
X = df_c.drop(species,axis =1)  # Inputs
Y = df_c[species]               # Output

X.head()

Y.head()

"""#3 Modelo"""

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense,Input

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape

classifier = Sequential()
classifier.add(Input(shape=(X_train_scaled.shape[1],)))  # Input layer with 9 features
classifier.add(Dense(8, activation = 'relu'))      # Hidden layer with 8 neurons
classifier.add( Dense(3, activation = 'softmax' ) ) # Output layer with 3 neurons (for 3 classes) and softmax activation

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy',metrics=['accuracy'])

classifier.summary()

# Commented out IPython magic to ensure Python compatibility.
historial = classifier.fit(X_train_scaled, y_train, epochs = 300, verbose = 0, validation_data=(X_test_scaled, y_test))

score = classifier.evaluate(X_test_scaled, y_test, verbose=0)
score

import numpy as np
import matplotlib.pyplot as plt

plt.plot(historial.epoch,historial.history['loss'], 'b',label="loss")
plt.plot(historial.epoch,historial.history['val_loss'], '*r',label="val_loss")
plt.title(u'MSE')
plt.xlabel(u'época')
plt.ylabel(r'$Loss(\omega, b)$')
plt.legend(loc='upper right')
plt.ylim([np.min(historial.history['loss'])-0.1,np.max(historial.history['loss'])+0.1])
plt.grid(True)
plt.show()

"""#4 Matriz de confusión"""

prediccion = classifier.predict(X_test_scaled)

y_pred = np.argmax(prediccion, axis=1)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

con = confusion_matrix( np.argmax(y_test, axis=1) , y_pred )
disp = ConfusionMatrixDisplay( confusion_matrix = con, display_labels = species).plot()
plt.savefig('penguins')
plt.show()