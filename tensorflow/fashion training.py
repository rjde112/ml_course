#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU, BatchNormalization, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scikeras.wrappers import KerasClassifier

from scipy.stats import uniform, randint
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Per salvare scaler e PCA
import joblib

# 1) DATA LOADING AND PREPARATION ------------------------------------------
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize to [0,1]
X_train_full = X_train_full.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1, random_state=42
)

# Reshape 28x28 to a single vector of length 784
X_train = X_train.reshape((-1, 28 * 28))
X_val = X_val.reshape((-1, 28 * 28))
X_test = X_test.reshape((-1, 28 * 28))

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# PCA for dimensionality reduction to 50 components
pca = PCA(n_components=50, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 1.5) VISUALIZATION OF EXAMPLES FROM THE DATASET -------------------------
def plot_sample_images(images, labels, class_names, num_samples=25):
    """
    Display a grid of sample images with their labels.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[labels[i]])
    plt.tight_layout()
    plt.show()

# Class names for Fashion MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Visualize examples from the training set
print("Displaying sample images from the dataset...")
plot_sample_images(X_train_full.reshape((-1, 28, 28)), y_train_full, class_names)

# 2) MODEL DEFINITION AND RANDOMIZED SEARCH --------------------------------
def create_model(first_layer_neurons=128, 
                 second_layer_neurons=64, 
                 third_layer_neurons=64,
                 dropout_rate=0.0,
                 learning_rate=0.001):
    """
    Improved neural network for Fashion MNIST:
    - Three Dense layers with LeakyReLU
    - Dropout to mitigate overfitting
    - BatchNormalization to stabilize training
    - 10-class output with Softmax
    """
    model = keras.Sequential()
    
    # Define the input shape explicitly
    model.add(keras.Input(shape=(50,)))
    
    # First Dense layer + BN + Dropout
    model.add(layers.Dense(first_layer_neurons))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Second Dense layer + BN + Dropout
    model.add(layers.Dense(second_layer_neurons))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Third Dense layer + BN + Dropout
    model.add(layers.Dense(third_layer_neurons))
    model.add(LeakyReLU(negative_slope=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(layers.Dense(10, activation='softmax'))
    
    # Compile
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# SciKeras wrapper
model_wrapper = KerasClassifier(
    model=create_model,
    verbose=0  # Avoid excessive output during Random Search
)

# Hyperparameter space
param_dist = {
    'model__first_layer_neurons':    randint(64, 256),
    'model__second_layer_neurons':   randint(32, 128),
    'model__third_layer_neurons':    randint(32, 128),
    'model__dropout_rate':           uniform(0.0, 0.4),
    'model__learning_rate':          uniform(1e-4, 1e-2),
    'epochs':                        randint(5, 20),
    'batch_size':                    randint(32, 128)
}

random_search = RandomizedSearchCV(
    estimator=model_wrapper,
    param_distributions=param_dist,
    n_iter=5,        # puoi aumentare per una ricerca più approfondita
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train_pca, y_train, validation_data=(X_val_pca, y_val))

print("\nBest parameters found by Random Search:")
print(random_search.best_params_)

# Evaluate on the test set
score_test = random_search.best_estimator_.score(X_test_pca, y_test)
print(f"Test set accuracy (Random Search model): {score_test:.4f}")

# 3) FINAL TRAINING + CHECKPOINT & CUSTOM CALLBACK -------------------------
best_params = random_search.best_params_

first_layer_neurons   = best_params['model__first_layer_neurons']
second_layer_neurons  = best_params['model__second_layer_neurons']
third_layer_neurons   = best_params['model__third_layer_neurons']
dropout_rate          = best_params['model__dropout_rate']
learning_rate         = best_params['model__learning_rate']
epochs                = best_params['epochs']
batch_size            = best_params['batch_size']

# Build the "final" model with the found hyperparameters
model_best = create_model(
    first_layer_neurons   = first_layer_neurons,
    second_layer_neurons  = second_layer_neurons,
    third_layer_neurons   = third_layer_neurons,
    dropout_rate          = dropout_rate,
    learning_rate         = learning_rate
)

# Custom callback to log Accuracy, Precision, Recall, F1 after each epoch
class MetricsLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_accuracy = []
        self.val_precision = []
        self.val_recall = []
        self.val_f1 = []
        
    def on_epoch_end(self, epoch, logs=None):
        y_val_pred = np.argmax(self.model.predict(X_val_pca), axis=1)
        
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

# Callback to save the best model
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    "best_model.keras",      # saves the entire model
    monitor='val_accuracy',  # metric to monitor
    mode='max',
    save_best_only=True,
    verbose=1
)

# EarlyStopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    mode='max',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

metrics_logger = MetricsLogger()

history = model_best.fit(
    X_train_pca, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val_pca, y_val),
    callbacks=[metrics_logger, checkpoint_callback, early_stopping],
    verbose=1
)

# Load the best model saved on file 
print("\nLoading the best model from disk...")
model_best = keras.models.load_model("best_model.keras")

# 4) FINAL EVALUATION AND PLOTS --------------------------------------------

# Predictions on the test set
y_pred = np.argmax(model_best.predict(X_test_pca), axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix (Test Set):")
print(cm)

# Additional metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy_test = np.mean(y_pred == y_test)

print(f"\nFinal Metrics on the Test Set (best model):")
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

# Plot of the metrics on the validation set
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

# SALVATAGGIO SCALER E PCA -------------------------------------------------
print("\nSalvataggio di best_model.keras, scaler.pkl e pca.pkl ...")

# il modello è già stato salvato in checkpoint_callback, ma salviamo
# nuovamente, così siamo sicuri di salvare la versione finale
model_best.save("best_model.keras")  

# Salviamo gli oggetti di preprocessing
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

# Class names for Fashion MNIST
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

#SAVE IMAGES
def save_individual_images(images, labels, class_names, output_dir, num_samples=25):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_samples):
        image = images[i]

        label_name = class_names[labels[i]].replace('/', '_')
        output_path = os.path.join(output_dir, f"image_{i+1}_{label_name}.png")
        

        plt.imsave(output_path, image, cmap='binary')
        print(f"Saved: {output_path}")

print("\nSaving individual sample images from the dataset (Train) to 'images' folder...")
save_individual_images(
    X_train_full,  # qui stiamo prendendo i primi 25 della parte "train" originale
    y_train_full, 
    class_names,
    output_dir="images"
)