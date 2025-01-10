#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score, classification_report
)
from scikeras.wrappers import KerasClassifier
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt

# 1) DATA LOADING AND PREPARATION ------------------------------------------
# Load the Fashion MNIST dataset
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Rescale to [0,1]
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

# Reshape to (28,28,1) for the CNN
X_train = np.expand_dims(X_train, axis=-1)  # shape -> (None, 28, 28, 1)
X_val   = np.expand_dims(X_val,   axis=-1)
X_test  = np.expand_dims(X_test,  axis=-1)

# 2) CNN MODEL DEFINITION --------------------------------------------------
def create_cnn(
    conv1_filters=32,
    conv2_filters=64,
    conv3_filters=64,
    kernel_size=3,
    dense_units=128,
    dropout_rate=0.3,
    learning_rate=0.001
):
    """
    A CNN model for Fashion MNIST with 3 conv blocks:
    - 3 Convolutional blocks (Conv2D -> LeakyReLU -> BN -> MaxPool -> Dropout)
    - Flatten + Dense layer with LeakyReLU + BN + Dropout
    - Final Dense(10) with softmax
    """
    model = keras.Sequential()

    # First conv block
    model.add(keras.Input(shape=(28, 28, 1)))
    model.add(Conv2D(filters=conv1_filters, kernel_size=kernel_size, padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Second conv block
    model.add(Conv2D(filters=conv2_filters, kernel_size=kernel_size, padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Third conv block
    model.add(Conv2D(filters=conv3_filters, kernel_size=kernel_size, padding='same'))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate))
    
    # Flatten and Dense
    model.add(Flatten())
    model.add(layers.Dense(dense_units))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))

    # Compile
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
    return model

# Wrap the model using scikeras
model_wrapper = KerasClassifier(
    model=create_cnn,
    verbose=0  # to avoid too much console output
)

# 3) RANDOM SEARCH SETUP ---------------------------------------------------
# Let's define a parameter distribution
param_dist = {
    'model__conv1_filters': randint(16, 64),
    'model__conv2_filters': randint(32, 128),
    'model__conv3_filters': randint(32, 128),
    'model__kernel_size':   randint(2, 5),
    'model__dense_units':   randint(64, 256),
    'model__dropout_rate':  uniform(0.0, 0.5),
    'model__learning_rate': uniform(1e-4, 1e-2),
    'batch_size':           randint(32, 128),
    # We increase the epochs range to 20-40
    'epochs':               randint(20, 40)
}

random_search = RandomizedSearchCV(
    estimator=model_wrapper,
    param_distributions=param_dist,
    n_iter=20,        # increased to explore more combinations
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# 4) FIT RANDOM SEARCH -----------------------------------------------------
random_search.fit(X_train, y_train, validation_data=(X_val, y_val))

print("\nBest parameters found by Random Search:")
print(random_search.best_params_)

# Evaluate on the test set (best estimator)
test_score = random_search.best_estimator_.score(X_test, y_test)
print(f"Test set accuracy (Random Search model): {test_score:.4f}")

# 5) BUILD FINAL MODEL + DATA AUGMENTATION ---------------------------------

# Extract best params
best_params = random_search.best_params_

conv1_filters = best_params['model__conv1_filters']
conv2_filters = best_params['model__conv2_filters']
conv3_filters = best_params['model__conv3_filters']
kernel_size   = best_params['model__kernel_size']
dense_units   = best_params['model__dense_units']
dropout_rate  = best_params['model__dropout_rate']
learning_rate = best_params['model__learning_rate']
epochs        = best_params['epochs']
batch_size    = best_params['batch_size']

# Build the "final" CNN with the found hyperparameters
model_best = create_cnn(
    conv1_filters=conv1_filters,
    conv2_filters=conv2_filters,
    conv3_filters=conv3_filters,
    kernel_size=kernel_size,
    dense_units=dense_units,
    dropout_rate=dropout_rate,
    learning_rate=learning_rate
)

# We set up data augmentation for the final training
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True  # set to True/False as desired
)
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

# Our custom callback to track Accuracy, Precision, Recall, F1 each epoch
class MetricsLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        
    def on_epoch_end(self, epoch, logs=None):
        y_val_pred = np.argmax(self.model.predict(X_val), axis=1)
        
        accuracy_val = np.mean(y_val_pred == y_val)
        precision_val = precision_score(y_val, y_val_pred, average='weighted')
        recall_val = recall_score(y_val, y_val_pred, average='weighted')
        f1_val = f1_score(y_val, y_val_pred, average='weighted')
        
        self.val_accuracy.append(accuracy_val)
        self.val_precision.append(precision_val)
        self.val_recall.append(recall_val)
        self.val_f1.append(f1_val)
        
        print(f"End of epoch {epoch+1}: "
              f"val_accuracy={accuracy_val:.4f}, "
              f"val_precision={precision_val:.4f}, "
              f"val_recall={recall_val:.4f}, "
              f"val_f1={f1_val:.4f}")

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "best_cnn_model.keras",
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

metrics_logger = MetricsLogger()

# Final training with data augmentation
history = model_best.fit(
    train_generator,
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=[metrics_logger, checkpoint_callback, early_stopping],
    verbose=1
)

print("\nLoading the best CNN model from disk...")
model_best = keras.models.load_model("best_cnn_model.keras")

# 6) FINAL EVALUATION AND PLOTS --------------------------------------------

# Predictions on the test set
y_pred = np.argmax(model_best.predict(X_test), axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Test Set):")
print(cm)

# Additional metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall    = recall_score(y_test, y_pred, average='weighted')
f1        = f1_score(y_test, y_pred, average='weighted')
accuracy_test = np.mean(y_pred == y_test)

print(f"\nFinal Metrics on the Test Set (best CNN model):")
print(f"Accuracy:  {accuracy_test:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# Classification report
report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

# Plot confusion matrix
plt.figure(figsize=(8, 8))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix (Test Set)")
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, range(10))
plt.yticks(tick_marks, range(10))
max_val = cm.max()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 ha='center', va='center',
                 color='red' if cm[i, j] > (max_val / 2) else 'black')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.show()

# Plot of metrics on validation set
epochs_range = range(1, len(metrics_logger.val_accuracy) + 1)

plt.figure(figsize=(10, 6))
plt.plot(epochs_range, metrics_logger.val_accuracy, label='Accuracy')
plt.plot(epochs_range, metrics_logger.val_precision, label='Precision')
plt.plot(epochs_range, metrics_logger.val_recall, label='Recall')
plt.plot(epochs_range, metrics_logger.val_f1, label='F1')
plt.title('Metrics Progress on Validation Set')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
