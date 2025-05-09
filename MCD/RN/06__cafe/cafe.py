import time
import datetime
import zoneinfo
import os
import keras
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet_v2, resnet_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint

# %pip install kaggle

os.environ['KAGGLE_USERNAME'] = 'user'
#os.environ['KAGGLE_KEY'] = 'kaggle datasets download -d gpiosenka/coffee-bean-dataset-resized-224-x-224'

#!kaggle datasets download -d gpiosenka/coffee-bean-dataset-resized-224-x-224
#%pip install unzip
#!unzip /content/coffee-bean-dataset-resized-224-x-224.zip -d data
data_train_dir = '/content/data/train'
class_names = sorted(os.listdir(data_train_dir))
n_classes = len(class_names)

class_dis = [len(glob('/content/data/train/' + name + "/*.png")) for name in class_names]

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_train_dir,
  validation_split = 0.2,
  subset           = "training",
  seed             = 123,
  image_size       = (img_height, img_width),
  batch_size       = batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
  data_train_dir,
  validation_split = 0.2,
  subset           = "validation",
  seed             = 123,
  image_size       = (img_height, img_width),
  batch_size       = batch_size)

# modelo 1
batch_size = 32
img_height = 150
img_width = 150

# import imageio
# import os

# import matplotlib.image  as mpimg
# import matplotlib.pyplot as plt
# import tensorflow        as tf
# import numpy             as np

# from os.path          import join
# from tensorflow       import keras
# from keras.models     import Sequential
# from keras.layers     import Dense, Flatten, Dropout, BatchNormalization, Activation, Input
# from keras.optimizers import Adam, Adagrad, SGD, Nadam


# from keras.utils      import load_img
# from keras.utils      import img_to_array

# from scipy            import misc, ndimage
# from numpy            import expand_dims

# from sklearn.metrics  import confusion_matrix
# from sklearn.metrics  import  ConfusionMatrixDisplay

train_ds, val_ds = image_dataset_from_directory(
                    directory        =  data_train_dir,
                    labels           = 'inferred',
                    class_names      =  ['Dark','Green','Light','Medium'],
                    image_size       = (img_height,img_width),
                    label_mode       = 'int',
                    seed             = 123,
                    shuffle          = True,
                    validation_split = 0.2,
                    subset           = "both",
                    batch_size       = batch_size,
)

#train_ds.class_names

train_label = np.concatenate([y for x, y in train_ds], axis=0)
#len(train_label)

#unique_elements, count_elements = np.unique(train_label,return_counts=True)
#print(f"Las clases existentes son: {unique_elements} ")
#print(f"Total de muestras: {count_elements}")
#for i in unique_elements:
#  print(f"Hay un {100*count_elements[i]/count_elements.sum():.2f}% muestras de la clase {i}")

#fig1,ax1=plt.subplots()
#ax1.pie( count_elements, labels=unique_elements, autopct='%1.1f%%',shadow=True,startangle=90 )
#plt.show()

#val_label = np.concatenate([y for x, y in val_ds], axis=0)
#len(val_label)

#val_ds.class_names

#unique_elements, count_elements = np.unique(val_label,return_counts=True)
#print(f"Las clases existentes son: {unique_elements} ")
#print(f"Total de muestras: {count_elements}")
#for i in unique_elements:
#  print(f"Hay un {100*count_elements[i]/count_elements.sum():.2f}% muestras de la clase {i}")

#fig1,ax1=plt.subplots()
#ax1.pie( count_elements, labels=unique_elements, autopct='%1.1f%%',shadow=True,startangle=90 )
#plt.show()

##Gráficas de loss y accuracy

##Matriz de confusión

#Aumento de datos

##Modelo2

##Gráficas de loss y accuracy

##Matriz de confusión