import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

MODEL_SAVE_PATH = 'models/flappy_q_rnn_model.keras'
QTABLE_PATH = 'flappy_birds_q_table.pkl'
NUM_ACTIONS = 2
EPOCHS = 100
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

X = np.array(list(q_table.keys()))
y = np.array(list(q_table.values()))

encoder = OneHotEncoder()
pos_column = X[:, 0].reshape(-1, 1)
encoded_pos = encoder.fit_transform(pos_column).toarray()

X = np.concatenate([X[:, :-1], encoded_pos], axis=1)

y = y / np.max(np.abs(y))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE)

X_train_rnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_rnn   = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

model = keras.Sequential([
    layers.Input(shape=(1, X_train.shape[1])),
    layers.SimpleRNN(64, activation='tanh', return_sequences=False),
    layers.Dense(NUM_ACTIONS, activation='linear')
])

model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

checkpoint = keras.callbacks.ModelCheckpoint(
    'models/best_model.keras',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

history = model.fit(
    X_train_rnn, y_train,
    validation_data=(X_val_rnn, y_val),
    batch_size=16,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=1,
    shuffle=True
)

os.makedirs('models/output', exist_ok=True)

plt.figure()
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.savefig('models/output/loss_rnn.png')
plt.close()

model.save(MODEL_SAVE_PATH)
print(f'Modelo guardado en {MODEL_SAVE_PATH}')

try:
    best_model = keras.models.load_model('models/flappy_q_nn_model.keras')
    train_loss, train_mae = best_model.evaluate(X_train_rnn, y_train, verbose=0)
    val_loss, val_mae = best_model.evaluate(X_val_rnn, y_val, verbose=0)
    print(f"Mejor modelo - Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
    print(f"Mejor modelo - Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
except:
    print("Error al cargar mejor modelo. Evaluando modelo final.")
    train_loss, train_mae = model.evaluate(X_train_rnn, y_train, verbose=0)
    val_loss, val_mae = model.evaluate(X_val_rnn, y_val, verbose=0)
    print(f"Final model - Train Loss: {train_loss:.4f}, MAE: {train_mae:.4f}")
    print(f"Final model - Val Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")
