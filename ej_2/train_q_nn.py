import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_data
from tensorflow.keras.callbacks import ModelCheckpoint

# Configuración de rutas y parámetros
MODEL_SAVE_PATH = 'models/flappy_q_nn_model.keras'  # Cambiado a formato .keras
NUM_ACTIONS = 2
EPOCHS = 500
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table.pkl'
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos ---
X = np.array(list(q_table.keys()))
y = np.array(list(q_table.values()))

# Normalizar Q-values
y = y / np.max(np.abs(y))

# Dividir datos
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=VALIDATION_SPLIT, 
    random_state=RANDOM_STATE
)

# --- Definir modelo ---
from tensorflow.keras import regularizers
from tensorflow.keras import layers, Sequential

# Modelo mejorado con BatchNormalization, Dropout y esquema de bloques
model = Sequential([
    layers.Input(shape=(X.shape[1],)),

    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dense(256),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dropout(0.2),

    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dense(128),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dropout(0.2),

    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dense(64),
    layers.BatchNormalization(),
    layers.Activation('elu'),
    layers.Dropout(0.2),

    layers.Dense(32),
    layers.Dense(16),
    layers.Dense(NUM_ACTIONS, activation='linear')
])

model.summary()


# Compilar con configuración explícita para evitar problemas de serialización
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.MeanSquaredError(),  # Usando la clase en lugar del string
    metrics=[keras.metrics.MeanAbsoluteError()],
)

# Callback para guardar el mejor modelo
checkpoint = ModelCheckpoint(
    'models/best_model.keras',  # Formato .keras
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min',
)

# --- Entrenamiento ---
history = model.fit(
    X_train, y_train,
    batch_size=16,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=[checkpoint],
    verbose=1,
    shuffle=True
)

output_dir = 'models/output'
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(6, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Progression')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training.png'))
plt.close()

plt.figure(figsize=(6, 5))
plt.plot(history.history['mean_absolute_error'], label='Training MAE')
plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
plt.title('Model MAE Progression')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'mae.png'))
plt.close()

# --- Guardar y evaluar ---
# Guardar modelo final
model.save(MODEL_SAVE_PATH)
print(f'Modelo guardado en {MODEL_SAVE_PATH}')

# Cargar y evaluar el mejor modelo
try:
    best_model = keras.models.load_model('models/best_model.keras')
    train_loss, train_mae = best_model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = best_model.evaluate(X_val, y_val, verbose=0)
    
    print("\nResultados del mejor modelo:")
    print(f"Training Loss: {train_loss:.4f} - Training MAE: {train_mae:.4f}")
    print(f"Validation Loss: {val_loss:.4f} - Validation MAE: {val_mae:.4f}")
except Exception as e:
    print(f"\nError al cargar el mejor modelo: {e}")
    print("Evaluando con el modelo final en su lugar...")
    train_loss, train_mae = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"Training Loss: {train_loss:.4f} - Training MAE: {train_mae:.4f}")
    print(f"Validation Loss: {val_loss:.4f} - Validation MAE: {val_mae:.4f}")