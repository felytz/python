# -*- coding: utf-8 -*-
"""RN_py_Desbalance Clases Thyroid_sick.ipynb"""

from imblearn.datasets       import fetch_datasets
from imblearn.under_sampling import TomekLinks
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.over_sampling  import ADASYN

import time
from collections import Counter
import numpy             as np
import seaborn           as sn
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import tensorflow        as tf

import keras
from keras.models            import Sequential
from keras.layers            import Dense, Flatten, Input
from keras.callbacks         import  EarlyStopping, ReduceLROnPlateau

#from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import RobustScaler
from sklearn.metrics         import ConfusionMatrixDisplay, confusion_matrix

import keras_tuner as kt

t = time.time( ) # <---- tiempo inicial

df = pd.DataFrame(fetch_datasets()["thyroid_sick"].data)
df['target'] = fetch_datasets()["thyroid_sick"].target

df

columns_names = df.columns
for col in columns_names:
    if np.std(df[col])==0:
        print(f"La columna {col} tiene std=0... se eliminó")
        df = df.drop([col], axis=1) #eliminar la columna

df.isnull().sum()

df.describe()

df.info()

for col in df.columns:
  mediana = df[col].median()
  std = df[col].std()

  for j in range( df.shape[0] ):
    if (df[col][j]<mediana-4*std or df[col][j]>mediana+4*std  ):
      print(f"En la columns {col} hay outliers")
      break

#df[df['target']==-1] = 0
df.target.replace(-1,0,inplace=True)

"""#X, y"""

X = df[ df.columns[df.columns != 'target'] ]
y = df.target

X

X.shape, y.shape

unique_elements, count_elements = np.unique(y,return_counts=True)
print(unique_elements)
print(count_elements)
print(f"Hay un {100*count_elements[0]/count_elements.sum()}% muestras de la clase 0")
print(f"Hay un {100*count_elements[1]/count_elements.sum()}% muestras de la clase 1")

fig1,ax1=plt.subplots()
ax1.pie( count_elements, labels=unique_elements, autopct='%1.1f%%',shadow=True,startangle=90 )
plt.show()

"""#X_train, X_test"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y )

unique_elements, count_elements = np.unique(y_train,return_counts=True)
print(unique_elements)
print(count_elements)

unique_elements, count_elements = np.unique(y_test,return_counts=True)
print(unique_elements)
print(count_elements)

"""##Aplicando robust scaler"""

scaler = RobustScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print( X_train.shape, y_train.shape )
print( X_test.shape,  y_test.shape  )

unique_elements, count_elements = np.unique(y_train,return_counts=True)
print(unique_elements)
print(count_elements)

unique_elements, count_elements = np.unique(y_test,return_counts=True)
print(unique_elements)
print(count_elements)

"""#Entrenando"""

def build_model(hp):
    model = Sequential()
    model.add( Input(  shape=(X_train.shape[1],)) )
    model.add(
        Dense(
            hp.Int('units_1', min_value=1, max_value=40, step=2),              
            hp.Choice('activation', ['leaky_relu','relu','selu']),             
            name = "dense1"
        )
    )

    model.add( Dense( 1, activation = 'sigmoid', name = "predictions" ) ) # <---- capa de salida

    lr = hp.Choice( 'lr', values=[1e-2, 1e-3, 1e-4] )
    optimizers_dict = {
        "Adam":    keras.optimizers.Adam(learning_rate=lr),
        "SGD":     keras.optimizers.SGD(learning_rate=lr),
        "Adagrad": keras.optimizers.Adagrad(learning_rate=lr)
        }

    hp_optimizers = hp.Choice(
        'optimizer',
        values=[ "SGD", "Adam", "Adagrad"]
        )

    model.compile( optimizer    = optimizers_dict[hp_optimizers],
                    loss      = "binary_crossentropy",
                    metrics   = ['accuracy']
                    )

    return model

build_model(kt.HyperParameters())

tuner = kt.Hyperband( # https://keras.io/api/keras_tuner/tuners/hyperband/
    build_model,
    objective            = kt.Objective("val_accuracy", "max"),
    executions_per_trial = 1,
    max_epochs           = 50,
    factor               = 3,
    directory            = 'salida',
    project_name         = 'intro_to_HP',
    overwrite            = True
)

hist = tuner.search(X_train, y_train, validation_split=0.2 )

"""###Construyendo el mejor modelo"""

# Get the optimal hyperparameters
#if tuner.results_summary():
best_hps = tuner.get_best_hyperparameters()[0]
print(f"La búsqueda de hiperparámetros está completa.")
print(f"capa densa1 con {best_hps['units_1']} unidades y activacion {best_hps['activation']}")
print(f" tasa de aprendizaje = {best_hps['lr']}")
#else:
 #   print("Tuner search has not been completed or did not find any results.")

mi_mejor_modelo = tuner.hypermodel.build(best_hps)

historial = mi_mejor_modelo.fit(X_train, y_train,validation_split=0.2,  epochs=20,  verbose=0 )

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.ylim((0,1))
    plt.grid()
    plt.show()


plot_hist(historial)

def plot_hist_loss(hist):
    plt.plot(hist.history["loss"],'.r')
    plt.plot(hist.history["val_loss"],'*b')
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.ylim((0,1))
    plt.grid()
    plt.show()


plot_hist_loss(historial)

mi_mejor_modelo.evaluate(X_test, y_test)

prediccion_test = mi_mejor_modelo.predict(X_test)

unique_elements, count_elements = np.unique(np.round(prediccion_test),return_counts=True)
print(unique_elements)
print(count_elements)

y_test

prediccion_test=prediccion_test.ravel()

pred_test = np.zeros(X_test.shape[0])

for id in range(X_test.shape[0]):
    pred_test[id] = np.round( prediccion_test[id] )

con = confusion_matrix( y_test , pred_test )
disp = ConfusionMatrixDisplay( confusion_matrix = con,  display_labels = ['No-Thyroid','Thyroid'] ).plot()
plt.show()

"""
#UnderSampling

##RandomUnderSampler

Submuestrear la(s) clase(s) mayoritaria(s) seleccionando muestras al azar con o sin reemplazo.
"""

rus = RandomUnderSampler( random_state = 42 )
X_resample, y_resample = rus.fit_resample(X_train, y_train)

_resampleunique_elements, count_elements = np.unique(y_resample,return_counts=True)
print(unique_elements)
print(count_elements)

fig1,ax1=plt.subplots()
ax1.pie( count_elements, labels=unique_elements, autopct='%1.1f%%',shadow=True,startangle=90 )
plt.show()

X_resample.shape, y_resample.shape

import keras_tuner as kt

def build_model(hp):
    model = Sequential()
    model.add( Input(  shape=(X_resample.shape[1],)) )
    model.add(
        Dense(
            hp.Int('units_1', min_value=1, max_value=40, step=2),               # Tune number of units.
            hp.Choice('activation', ['leaky_relu','relu','selu']),                     # Tune the activation function to use.
            name = "dense1"
        )
    )

    model.add( Dense( 1, activation = 'sigmoid', name = "predictions" ) ) # <---- capa de salida


    lr = hp.Choice( 'lr', values=[1e-2, 1e-3, 1e-4] )
    optimizers_dict = {
        "Adam":    keras.optimizers.Adam(learning_rate=lr),
        "SGD":     keras.optimizers.SGD(learning_rate=lr),
        "Adagrad": keras.optimizers.Adagrad(learning_rate=lr)
        }

    hp_optimizers = hp.Choice(
        'optimizer',
        values=[ "SGD", "Adam", "Adagrad"]
        )

    model.compile( optimizer    = optimizers_dict[hp_optimizers],
                    loss      = "binary_crossentropy",
                    metrics   = ['accuracy']
                    )

    return model

build_model(kt.HyperParameters())

tuner = kt.Hyperband( # https://keras.io/api/keras_tuner/tuners/hyperband/
    build_model,
    objective            = kt.Objective("val_accuracy", "max"),
    executions_per_trial = 1,
    max_epochs           = 50,
    factor               = 3,
    directory            = 'salida',
    project_name         = 'intro_to_HP',
    overwrite            = True
)

"""###Búsqueda de los mejores hiperparámetros"""

hist = tuner.search(X_resample, y_resample, validation_split=0.2 )

"""###Construyendo el mejor modelo"""

# Get the optimal hyperparameters
if tuner.results_summary():
    best_hps = tuner.get_best_hyperparameters()[0]
else:
    print("Tuner search has not been completed or did not find any results.")

print(f"La búsqueda de hiperparámetros está completa.")
print(f"capa densa1 con {best_hps['units_1']} unidades y activacion {best_hps['activation']}")
print(f" tasa de aprendizaje = {best_hps['lr']}")

mi_mejor_modelo = tuner.hypermodel.build(best_hps)

mi_mejor_modelo.summary()

historial = mi_mejor_modelo.fit(X_resample, y_resample,validation_split=0.2,  epochs=100,  verbose=0 )

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.ylim((0,1.1))
    plt.grid()
    plt.show()


plot_hist(historial)

def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.ylim((0,1))
    plt.grid()
    plt.show()


plot_hist_loss(historial)

mi_mejor_modelo.evaluate(X_test, y_test)

"""### Matriz de confusión"""

prediccion_test = mi_mejor_modelo.predict(X_test)

y_test

pred_test = np.zeros(X_test.shape[0])

for id in range(X_test.shape[0]):
    pred_test[id] = np.round( prediccion_test[id] )

con = confusion_matrix( y_test , pred_test )
disp = ConfusionMatrixDisplay( confusion_matrix = con,  display_labels = ['No-Thyroid','Thyroid'] ).plot()
plt.savefig('confusio_01.png')

"""##Tomek

Submuestreo eliminando los enlaces de Tomek
"""

from imblearn.under_sampling import TomekLinks

print( Counter(y_train) )
undersample = TomekLinks( sampling_strategy = "majority", n_jobs=2 ) # not minority, majority,  not majority, auto
X_resample, y_resample = undersample.fit_resample( X_train, y_train )
print(f"{Counter(y_resample)}, se eliminaron {y_train.sum()-y_resample.sum()} de la clase 1" )

unique_elements, count_elements = np.unique(y_resample,return_counts=True)
print(unique_elements)
print(count_elements)

fig1,ax1=plt.subplots()
ax1.pie( count_elements, labels=unique_elements, autopct='%1.1f%%',shadow=True,startangle=90 )
plt.show()

def build_model(hp):
    model = Sequential()
    model.add( Input(  shape=(X_resample.shape[1],)) )
    model.add(
        Dense(
            hp.Int('units_1', min_value=1, max_value=40, step=2),               # Tune number of units.
            hp.Choice('activation_1', ['leaky_relu','relu','silu' ]),                     # Tune the activation function to use.
            name = "dense1"
        )
    )
    model.add(
        Dense(
            hp.Int('units_2', min_value=1, max_value=40, step=2),               # Tune number of units.
            hp.Choice('activation_2', ['leaky_relu','relu','silu' ]),                     # Tune the activation function to use.
            name = "dense2"
        )
    )
    model.add( Dense( 1, activation = 'sigmoid', name = "predictions" ) ) # <---- capa de salida


    lr = hp.Choice( 'lr', values=[1e-2, 1e-3, 1e-4] )
    optimizers_dict = {
        "Adam":    keras.optimizers.Adam(learning_rate=lr),
        "SGD":     keras.optimizers.SGD(learning_rate=lr),
        "Adagrad": keras.optimizers.Adagrad(learning_rate=lr)
        }

    hp_optimizers = hp.Choice(
        'optimizer',
        values=[ "SGD", "Adam", "Adagrad"]
        )

    model.compile( optimizer    = optimizers_dict[hp_optimizers],
                    loss      = "binary_crossentropy",
                    metrics   = ['accuracy']
                    )

    return model

build_model(kt.HyperParameters())

tuner = kt.Hyperband( # https://keras.io/api/keras_tuner/tuners/hyperband/
    build_model,
    objective            = kt.Objective("val_accuracy", "max"),
    executions_per_trial = 1,
    max_epochs           = 50,
    factor               = 3,
    directory            = 'salida',
    project_name         = 'intro_to_HP',
    overwrite            = True
)

"""###Búsqueda de los mejores hiperparámetros"""

hist = tuner.search(X_resample, y_resample, validation_split=0.2 )

"""###Construyendo el mejor modelo"""

# Get the optimal hyperparameters
if tuner.results_summary():
    best_hps = tuner.get_best_hyperparameters()[0]
else:
    print("Tuner search has not been completed or did not find any results.")

mi_mejor_modelo = tuner.hypermodel.build(best_hps)

mi_mejor_modelo.summary()

historial = mi_mejor_modelo.fit(X_resample, y_resample,  epochs=100,  verbose=0 )

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
 #   plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.ylim((0,1.1))
    plt.grid()
    plt.show()


plot_hist(historial)

def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
 #   plt.plot(hist.history["val_accuracy"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.ylim((0,1))
    plt.grid()
    plt.show()


plot_hist_loss(historial)

mi_mejor_modelo.evaluate(X_test, y_test)

prediccion_test = mi_mejor_modelo.predict(X_test).ravel()

pred_test = np.zeros(X_test.shape[0])

for id in range(X_test.shape[0]):
    pred_test[id] = np.round( prediccion_test[id] )

"""###Matriz de confusión"""

con = confusion_matrix( y_test , pred_test )
disp = ConfusionMatrixDisplay( confusion_matrix = con,  display_labels = ['No-Thyroid','Thyroid'] ).plot()
plt.savefig('confusio_02.png')

"""---

##ClusterCentroids
Submuestreo basado en la regla de limpieza del vecindario.
"""

from imblearn.under_sampling import ClusterCentroids

ncr = ClusterCentroids()

X_resample, y_resample = ncr.fit_resample(X_train, y_train)

unique_elements, count_elements = np.unique(y_resample,return_counts=True)
print(unique_elements)
print(count_elements)

fig1,ax1=plt.subplots()
ax1.pie( count_elements, labels=unique_elements, autopct='%1.1f%%',shadow=True,startangle=90 )
plt.show()

def build_model(hp):
    model = Sequential()
    model.add( Input(  shape=(X_resample.shape[1],)) )
    model.add(
        Dense(
            hp.Int('units_1', min_value=1, max_value=40, step=2),               # Tune number of units.
            hp.Choice('activation', ['leaky_relu','relu' ]),                     # Tune the activation function to use.
            name = "dense1"
        )
    )
    model.add(
        Dense(
            hp.Int('units_2', min_value=1, max_value=40, step=2),               # Tune number of units.
            hp.Choice('activation_2', ['leaky_relu','relu','silu' ]),                     # Tune the activation function to use.
            name = "dense2"
        )
    )
    model.add( Dense( 1, activation = 'sigmoid', name = "predictions" ) ) # <---- capa de salida

    lr = hp.Choice( 'lr', values=[1e-2, 1e-3, 1e-4] )
    optimizers_dict = {
        "Adam":    keras.optimizers.Adam(learning_rate=lr),
        "SGD":     keras.optimizers.SGD(learning_rate=lr),
        "Adagrad": keras.optimizers.Adagrad(learning_rate=lr)
        }

    hp_optimizers = hp.Choice(
        'optimizer',
        values=[ "SGD", "Adam", "Adagrad"]
        )

    model.compile( optimizer    = optimizers_dict[hp_optimizers],
                    loss      = "binary_crossentropy",
                    metrics   = ['accuracy']
                    )

    return model

build_model(kt.HyperParameters())

tuner = kt.Hyperband( # https://keras.io/api/keras_tuner/tuners/hyperband/
    build_model,
    objective            = kt.Objective("val_accuracy", "max"),
    executions_per_trial = 1,
    max_epochs           = 50,
    factor               = 3,
    directory            = 'salida',
    project_name         = 'intro_to_HP',
    overwrite            = True
)

"""###Búsqueda de los mejores hiperparámetros"""

hist = tuner.search(X_resample, y_resample, validation_split=0.2 )

"""###Construyendo el mejor modelo"""

# Get the optimal hyperparameters
if tuner.results_summary():
    best_hps = tuner.get_best_hyperparameters()[0]
else:
    print("Tuner search has not been completed or did not find any results.")

print(f"La búsqueda de hiperparámetros está completa.")
print(f"capa densa1 con {best_hps['units_1']} unidades y activacion {best_hps['activation']}")
print(f"capa densa1 con {best_hps['units_2']} unidades y activacion {best_hps['activation_2']}")
print(f"tasa de aprendizaje = {best_hps['lr']}")

mi_mejor_modelo = tuner.hypermodel.build(best_hps)

mi_mejor_modelo.summary()

historial = mi_mejor_modelo.fit(X_resample, y_resample, validation_split=0.2,  epochs=100,  verbose=0 )

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.ylim((0,1))
    plt.grid()
    plt.show()


plot_hist(historial)

def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.ylim((-0.1,1))
    plt.grid()
    plt.show()


plot_hist_loss(historial)

mi_mejor_modelo.evaluate(X_test, y_test)

prediccion_test = mi_mejor_modelo.predict(X_test)

y_test

pred_test = np.zeros(X_test.shape[0])

for id in range(X_test.shape[0]):
    pred_test[id] = np.round( prediccion_test[id] )

"""###Matriz de confusión"""

con = confusion_matrix( y_test , pred_test )
disp = ConfusionMatrixDisplay( confusion_matrix = con,  display_labels = ['No-Thyroid','Thyroid'] ).plot()
plt.savefig('confusio_03.png')


"""#Oversampling

##SMOTE (agrega datos sintéticos)
"""

from imblearn.over_sampling import SMOTE  #variante de SMOTE

ada = SMOTE (random_state=42)

X_resample, y_resample  = ada.fit_resample( X_train, y_train )

print(Counter(y_resample))

unique_elements, count_elements = np.unique(y_resample,return_counts=True)
print(unique_elements)
print(count_elements)

fig1,ax1=plt.subplots()
ax1.pie( count_elements, labels=unique_elements, autopct='%1.1f%%',shadow=True,startangle=90 )
plt.show()

def build_model(hp):
    model = Sequential()
    model.add( Input(  shape=(X_resample.shape[1],)) )
    model.add(
        Dense(
            hp.Int('units_1', min_value=1, max_value=40, step=2),               # Tune number of units.
            hp.Choice('activation', ['leaky_relu','relu' ]),                     # Tune the activation function to use.
            name = "dense1"
        )
    )

    model.add( Dense( 1, activation = 'sigmoid', name = "predictions" ) ) # <---- capa de salida

    lr = hp.Choice( 'lr', values=[1e-2, 1e-3, 1e-4] )
    optimizers_dict = {
        "Adam":    keras.optimizers.Adam(learning_rate=lr),
        "SGD":     keras.optimizers.SGD(learning_rate=lr),
        "Adagrad": keras.optimizers.Adagrad(learning_rate=lr)
        }

    hp_optimizers = hp.Choice(
        'optimizer',
        values=[ "SGD", "Adam", "Adagrad"]
        )

    model.compile( optimizer    = optimizers_dict[hp_optimizers],
                    loss      = "binary_crossentropy",
                    metrics   = ['accuracy']
                    )

    return model

build_model(kt.HyperParameters())

tuner = kt.Hyperband( # https://keras.io/api/keras_tuner/tuners/hyperband/
    build_model,
    objective            = kt.Objective("val_accuracy", "max"),
    executions_per_trial = 1,
    max_epochs           = 50,
    factor               = 3,
    directory            = 'salida',
    project_name         = 'intro_to_HP',
    overwrite            = True
)

"""###Búsqueda de los mejores hiperparámetros"""

hist = tuner.search(X_resample, y_resample, validation_split=0.2 )

"""###Construyendo el mejor modelo"""

# Get the optimal hyperparameters
if tuner.results_summary():
    best_hps = tuner.get_best_hyperparameters()[0]
else:
    print("Tuner search has not been completed or did not find any results.")

print(f"La búsqueda de hiperparámetros está completa.")
print(f"capa densa1 con {best_hps['units_1']} unidades y activacion {best_hps['activation']}")
print(f" tasa de aprendizaje = {best_hps['lr']}")

mi_mejor_modelo = tuner.hypermodel.build(best_hps)

mi_mejor_modelo.summary()

historial = mi_mejor_modelo.fit(X_resample, y_resample,  epochs=100,  verbose=0 )

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    #plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.ylim((0,1))
    plt.grid()
    plt.show()


plot_hist(historial)

def plot_hist_loss(hist):
    plt.plot(hist.history["loss"])
    #plt.plot(hist.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    #plt.ylim((0,1))
    plt.grid()
    plt.show()


plot_hist_loss(historial)

mi_mejor_modelo.evaluate(X_test, y_test)

prediccion_test = mi_mejor_modelo.predict(X_test).ravel()

y_test

pred_test = np.zeros(X_test.shape[0])

for id in range(X_test.shape[0]):
    pred_test[id] = np.round( prediccion_test[id])

"""###Matriz de confusión"""

con = confusion_matrix( y_test , pred_test )
disp = ConfusionMatrixDisplay( confusion_matrix = con,  display_labels = ['No-Thyroid','Thyroid'] ).plot()
plt.savefig('confusio_04.png')

tiempo = time.time() - t
if(tiempo<60):
  print(f'Tiempo de entrenamiento total {tiempo:.5f} segs')
else:
  if(tiempo/60<60):
    print(f'Tiempo de entrenamiento total {tiempo/60:.5f} mins')
  else:
    if( (tiempo/60)/60 >=1 ):
      print(f'Tiempo de entrenamiento total {(tiempo/60)/60:.5f} hrs')