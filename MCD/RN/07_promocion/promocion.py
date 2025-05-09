import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from imblearn.combine import SMOTEENN
import keras_tuner as kt
from keras_tuner import BayesianOptimization

# Random control
np.random.seed(42)
tf.random.set_seed(42)

# deshabilitar GPU para tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# datos
df = pd.read_csv('employee_promotion.csv')
df_clean = df.copy()
df_clean=df_clean.drop(['employee_id'], axis=1)
df_clean = df_clean.dropna()

#########################################
### 1. Análisis Exploratorio Inicial ###
#########################################

# Función para graficar distribución
def plot_distribution(y, title, filename):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title(title)
    plt.xlabel('Promovido')
    plt.ylabel('Cantidad')
    plt.xticks([0, 1], ['No', 'Sí'])
    plt.savefig(filename)
    plt.close()

# Distribución de la variable objetivo original
plot_distribution(df_clean['is_promoted'], 
                 'Distribución Original de Promociones', 
                 'distribucion_promociones_original.png')

#########################################
## 2.Limpieza de variables categoricas ##
#########################################

threshold = 10 # Frecuencia(en porcentaje) de una categoria
value_counts = df_clean['department'].value_counts(normalize=True).mul(100)
df_clean['department'] = np.where(df_clean['department'].isin(value_counts[value_counts >= threshold].index), 
                                 df_clean['department'], 
                                 'Other')
                                 
threshold = 2 # Frecuencia(en porcentaje) de una categoria
value_counts = df_clean['region'].value_counts(normalize=True).mul(100)
df_clean['region'] = np.where(df_clean['region'].isin(value_counts[value_counts >= threshold].index), 
                                 df_clean['region'], 
                                 'region_lt_2')
                                 
threshold = 6 # Frecuencia(en porcentaje) de una categoria
value_counts = df_clean['region'].value_counts(normalize=True).mul(100)
df_clean['region'] = np.where(df_clean['region'].isin(value_counts[value_counts >= threshold].index), 
                                 df_clean['region'], 
                                 'region_lt_6')

# Se elimina las categorias que no cumplan con el criterio
df_clean = df_clean.drop(df_clean[df_clean['education'] == 'Below Secondary'].index)

# Aplicar One-Hot Encoding
target_col=['is_promoted']
num_cols=['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'awards_won', 'avg_training_score']
binary_cols=['education', 'recruitment_channel', 'gender'] # binary categorical columns
df_clean = pd.get_dummies(df_clean, columns=binary_cols, drop_first=True) 
cat_cols=['department', 'region'] # non-binary categorical columns
df_clean = pd.get_dummies(df_clean, columns=cat_cols)

# Verificación de columnas constantes
constant_columns = [col for col in df_clean.columns if df_clean[col].nunique() == 1]
if constant_columns:
    print(f"Columnas constantes encontradas y eliminadas: {constant_columns}")
    df_clean = df_clean.drop(columns=constant_columns)

# Conservar columnas relevantes
corr = df_clean.corr(method='pearson')
corr_with_price = corr['is_promoted']
columns_to_keep = corr.columns[np.abs(corr_with_price) > 0.1]
df_clean = df_clean[columns_to_keep]

# Separar características (X) y etiquetas (y) antes de escalar
X = df_clean.drop('is_promoted', axis=1)
y = df_clean['is_promoted']

# Dividir en conjuntos de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  
)

#########################################
### 3. Detección de Outliers ###
#########################################

# Seleccionar solo columnas numéricas
numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

# Función para detectar outliers usando el método IQR
def detect_outliers(df, columns):
    outlier_info = {}
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers)/len(df)*100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    return outlier_info

# Detectar outliers
outlier_info = detect_outliers(X_train, numeric_cols)

# Mostrar información de outliers
print("\nDetección de Outliers:")
for col, info in outlier_info.items():
    print(f"Columna: {col}")
    print(f"  - Número de outliers: {info['count']}")
    print(f"  - Porcentaje de outliers: {info['percentage']:.2f}%")
    print(f"  - Límite inferior: {info['lower_bound']:.2f}")
    print(f"  - Límite superior: {info['upper_bound']:.2f}")
    print("")

# Crear gráficas de caja para visualizar outliers
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(y=X_train[col])
    plt.title(f'Boxplot de {col}')
    plt.ylabel('Valor')

plt.tight_layout()
plt.savefig('outliers.png')
plt.close()

#########################################
### 4. Escalado de Datos ###
#########################################

# Aplicar RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#########################################
### 5. Balanceo de Datos ###
#########################################

# Técnica 0: Datos originales (sin balanceo)
# Calculamos pesos de clases para el conjunto original
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Técnica 1: Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train_scaled, y_train)
plot_distribution(y_train_under, 
                 'Distribución después de Undersampling', 
                 'distribucion_promociones_undersampling.png')

# Técnica 2: Oversampling con SMOTE
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train_scaled, y_train)
plot_distribution(y_train_over, 
                 'Distribución después de SMOTE', 
                 'distribucion_promociones_smote.png')

# Técnica 3: Combinación SMOTE + ENN
smote_enn = SMOTEENN(random_state=42)
X_train_smoteenn, y_train_smoteenn = smote_enn.fit_resample(X_train_scaled, y_train)
plot_distribution(y_train_smoteenn, 
                 'Distribución después de SMOTEENN', 
                 'distribucion_promociones_smoteenn.png')

#########################################
### 6. Modelado y Evaluación ###
#########################################

# Define model building function
def build_model(hp):
    model = Sequential()
    model.add(Input(shape=(X_train_scaled.shape[1],)))

    # Tune the number of layers and units
    for i in range(hp.Int("num_layers", 1, 2)):
        model.add(Dense(
            units=hp.Int(f'units_{i}', min_value=32, max_value=256, step=32),
            activation=hp.Choice(f"activation_{i}", ["relu", "selu"])
        ))
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(1, activation='sigmoid', name="predictions"))

    # Tune learning rate and optimizer
    lr = hp.Choice('lr', values=[1e-3, 1e-4])
    optimizer_name = hp.Choice('optimizer', values=["adam", "adagrad"])
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    else:
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy', tf.keras.metrics.AUC()])
    
    return model

# Función para entrenar y evaluar modelos
def train_and_evaluate(X_train_balanced, y_train_balanced, X_test, y_test, technique_name, class_weight=None):
    # Hyperparameter tuning
    tuner = BayesianOptimization(
        build_model,
        objective='val_auc',
        max_trials=30,
        executions_per_trial=10,
        directory='salida',
        project_name=f'bayesian_opt_{technique_name}',
        overwrite=True,
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        mode='min'
    )
    
    tuner.search(
        X_train_balanced, y_train_balanced,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
        class_weight=class_weight
    )
    
    # Get best model
    best_hps = tuner.get_best_hyperparameters()[0]
    best_model = tuner.hypermodel.build(best_hps)
    
    history = best_model.fit(
        X_train_balanced, y_train_balanced,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0,
        class_weight=class_weight
    )
    
    # Evaluation
    y_pred = (best_model.predict(X_test) > 0.5).astype(int)
    print(f"\nResultados para {technique_name}:")
    print(classification_report(y_test, y_pred))
    
    # Gráficas
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfica de precisión
    ax1.plot(history.history['accuracy'], label='Precisión entrenamiento')
    ax1.plot(history.history['val_accuracy'], label='Precisión validación')
    ax1.set_title(f'Precisión - {technique_name}')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisión')
    ax1.legend()
    
    # Gráfica de pérdida
    ax2.plot(history.history['loss'], label='Pérdida entrenamiento')
    ax2.plot(history.history['val_loss'], label='Pérdida validación')
    ax2.set_title(f'Pérdida - {technique_name}')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Pérdida')
    ax2.legend()
    
    plt.savefig(f'{technique_name}_graficas.png')
    plt.close()
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                             display_labels=["No promovido", "Promovido"])
    disp.plot()
    plt.title(f'Matriz de Confusión - {technique_name}')
    plt.savefig(f'{technique_name}_confusion_matrix.png')
    plt.close()
    
    return best_model, history

# Técnicas de balanceo
techniques = {
    "Original (Class Weight)": (X_train_scaled, y_train, class_weights),
    "Undersampling": (X_train_under, y_train_under, None),
    "Oversampling SMOTE": (X_train_over, y_train_over, None),
    "SMOTEENN": (X_train_smoteenn, y_train_smoteenn, None)
}

# Diccionario para almacenar resultados
results = {}

# Entrenar y evaluar cada técnica
for name, (X_bal, y_bal, weights) in techniques.items():
    print(f"\nEntrenando modelo con técnica: {name}")
    model, history = train_and_evaluate(X_bal, y_bal, X_test_scaled, y_test, name, weights)
    results[name] = {
        'model': model,
        'history': history,
        'test_loss': model.evaluate(X_test_scaled, y_test, verbose=0)[0],
        'test_accuracy': model.evaluate(X_test_scaled, y_test, verbose=0)[1],
        'test_auc': model.evaluate(X_test_scaled, y_test, verbose=0)[2]
    }

# Comparación final de métricas
print("\nComparación Final de Técnicas:")
comparison_df = pd.DataFrame({
    'Técnica': results.keys(),
    'Pérdida Test': [res['test_loss'] for res in results.values()],
    'Precisión Test': [res['test_accuracy'] for res in results.values()],
    'AUC Test': [res['test_auc'] for res in results.values()]
}).sort_values('AUC Test', ascending=False)

print(comparison_df.to_string(index=False))

# Identificar el mejor modelo basado en AUC
best_technique = max(results.items(), key=lambda x: x[1]['test_auc'])[0]
print(f"\nLa mejor técnica fue: {best_technique}")