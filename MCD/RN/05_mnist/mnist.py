# -*- coding: utf-8 -*-
"""MNIST ropa v1.ipynb"""

import numpy as np
import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from kerastuner.tuners import RandomSearch

"""# [MNIST](https://keras.io/api/datasets/fashion_mnist/)

Se trata de un dataset de imágenes de prendas de ropa. Las clases son

```
Etiqueta	Descripción
0	        T-shirt/top
1	        Trouser
2	        Pullover
3	        Dress
4	        Coat
5	        Sandal
6	        Shirt
7	        Sneaker
8	        Bag
9	        Ankle boot
```
"""

(img_train, label_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

type(label_train) , type(img_train)

#ids_img = np.random.randint(0,img_train.shape[0],16)
#for i in range(len(ids_img)):
#    img = img_train[ids_img[i],:,:]
    #plt.subplot(4,4,i+1)
    #plt.imshow(img,cmap='gray')
    #xlabel = "True: {0}".format(label_train[ids_img[i]])
    #plt.xlabel(xlabel)
    #plt.axis('off')
#plt.suptitle('16 imágenes del dataset')
#plt.show()

img = 1
df = pd.DataFrame(img_train[img,:,:])
print(label_train[img])
df.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greys')

print("TRAIN")
unique_elements, count_elements = np.unique(label_train,return_counts=True)
print("Clases=",unique_elements)
print("Num de muestras por clase = ",count_elements)

print("TEST")
unique_elements, count_elements = np.unique(y_test,return_counts=True)
print("Clases=",unique_elements)
print("Num de muestras por clase = ",count_elements)

X_train = img_train[:-10000]
X_val   = img_train[-10000:]
y_train = label_train[:-10000]
y_val   = label_train[-10000:]

X_train.shape, X_val.shape, X_test.shape


X_train.shape, X_val.shape, X_test.shape

num_classes = 10
y_train  = keras.utils.to_categorical(y_train, num_classes)
y_val    = keras.utils.to_categorical(y_val, num_classes)
y_test_c = keras.utils.to_categorical(y_test, num_classes)

y_train.shape, y_val.shape

X_train.shape

"""#Modelo"""

def build_model(hp):
    model = keras.Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(Dense(units=hp_units, activation='relu'))
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
    model.add(Dropout(hp_dropout))
    model.add(Dense(10, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
              loss='categorical_crossentropy',  # Change this line
              metrics=['accuracy'])
    return model
    
# randomSearch
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,  # Número de modelos diferentes para probar
    executions_per_trial=3,  # Número de veces que se entrena cada modelo
    directory='my_tuner_dir',
    project_name='mnist_tuning'
)
    
# mejor modelo
tuner.search(X_train, y_train, epochs=5, validation_split=0.2)

# mejor modelo
best_model = tuner.get_best_models(num_models=1)[0]
    
import datetime
import time
import zoneinfo
zona_hermosillo = zoneinfo.ZoneInfo("America/Hermosillo")

hora_actual = datetime.datetime.now( tz=zona_hermosillo )
print("Empieza la ejecución : ",hora_actual)


# Commented out IPython magic to ensure Python compatibility.
nepocas = 100
historial = best_model.fit( X_train, y_train, validation_data=(X_val,y_val),  epochs = nepocas  )

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="lower left")
    plt.ylim((0,1.1))
    plt.grid()
    plt.savefig('hist.png')


plot_hist(historial)

def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="lower left")
    #plt.ylim((0,1))
    plt.grid()
    plt.savefig('hist_loss.png')


plot_hist_loss(historial)

best_model.evaluate(X_test, y_test_c)

"""## Matriz de confusión"""

prediccion_test = best_model.predict(X_test)

pred_test = np.zeros(X_test.shape[0])

for id in range(X_test.shape[0]):
    pred_test[id] = np.argmax( prediccion_test[id] )

#df_y_pred  = pd.DataFrame(data=pred_test, columns=['species'])
#ohe_y_pred = pd.DataFrame( data = ohe.fit_transform(df_y_pred) )
#y_pred_    = ohe.inverse_transform(ohe_y_pred)

con = confusion_matrix( y_test , pred_test )
disp = ConfusionMatrixDisplay( confusion_matrix = con ).plot()
plt.savefig('confusion_matrix.png')