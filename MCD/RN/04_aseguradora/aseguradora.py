# -*- coding: utf-8 -*-
import time
import seaborn as sns
from sklearn import datasets
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras_tuner as kt
from keras.models import Sequential
from sklearn import preprocessing
from keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN
from keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv('aug_train.csv')

# Data preprocessing
df = df.dropna()
df_clean = df.drop('id', axis=1)

# Rename columns
df_clean = df_clean.rename(columns={
    "Age": "age", "Driving_License": "license", "Region_Code": "region",
    "Previously_Insured": "prev_ins", "Annual_Premium": "premium",
    "Vintage": "days", "Policy_Sales_Channel": "channel", "Response": "response",
    "Gender": 'gender', 'Vehicle_Age': 'v_age', 'Vehicle_Damage': 'v_dmg'
})

# Simplify vehicle age categories
df_clean['v_age'] = df_clean['v_age'].apply(lambda x: "< 1 Year" if x == "< 1 Year" else "> 1 Year")

# One-Hot Encoding
df_clean = pd.get_dummies(df_clean, columns=['v_age', 'gender', 'v_dmg'], drop_first=True)

# Remove outliers in 'premium'
premium_data = df_clean['premium'].dropna()
Q1, Q3 = np.percentile(premium_data, [25, 75])
IQR = Q3 - Q1
lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
df_wo_outliers = df_clean[(df_clean['premium'] >= lower_bound) & (df_clean['premium'] <= upper_bound)]

# Correlation analysis
corr = df_wo_outliers.corr(method='pearson')
corr_with_price = corr['response']
columns_to_keep = corr.columns[np.abs(corr_with_price) > 0.1]
df_filter = df_wo_outliers[columns_to_keep]

# Simplify 'channel' column
channel_mapping = {152: 'channel 152', 26: 'channel 26', 124: 'channel 124'}
df_filter['channel'] = df_filter['channel'].map(channel_mapping).fillna('others')
df_filter = pd.get_dummies(df_filter, columns=['channel'])

# Final correlation analysis
corr = df_filter.corr(method='pearson')
corr_with_price = corr['response']
columns_to_keep = corr.columns[np.abs(corr_with_price) > 0.1]
df_filter = df_filter[columns_to_keep]

# Prepare features and target
X = df_filter.drop('response', axis=1)
y = df_filter['response']

# Resample data to handle imbalance
smote_enn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smote_enn.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled)

# Scale data
scaler = preprocessing.RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define model building function
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))

    # Tune the number of layers and units
    for i in range(hp.Int("num_layers", 1, 2)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=10, max_value=30, step=5),
            activation=hp.Choice(f"activation_{i}", ["relu", "selu"])
        ))

    model.add(Dense(1, activation='sigmoid', name="predictions"))

    # Tune learning rate and optimizer
    lr = hp.Choice('lr', values=[1e-3, 1e-4])
    optimizer = hp.Choice('optimizer', values=["Adam", "Adagrad"])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

    return model

# Hyperparameter tuning
tuner = kt.RandomSearch(
    build_model,
    objective=kt.Objective("val_accuracy", "max"),
    max_trials=5,  # Reduced number of trials for faster execution
    executions_per_trial=1,
    directory='salida',
    project_name='intro_to_HP',
    overwrite=True
)

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Perform hyperparameter search
tuner.search(X_train_scaled, y_train, validation_split=0.2, callbacks=[early_stopping])

# Get the best model
best_hps = tuner.get_best_hyperparameters()[0]
best_model = tuner.hypermodel.build(best_hps)

# Train the best model
history = best_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=20, callbacks=[early_stopping], verbose=0)

# Evaluate the model
test_loss, test_accuracy = best_model.evaluate(X_test_scaled, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.savefig('training_hist.jpg')

plot_history(history)

# Confusion Matrix
y_pred = best_model.predict(X_test_scaled)
y_pred = np.round(y_pred).astype(int)

conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['No Response', 'Response'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('confusion_m.jpg')