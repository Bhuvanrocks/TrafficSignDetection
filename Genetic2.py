import numpy as np
import random
import tensorflow.keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
import gc  # Garbage collector

# Dataset paths
path = "/content/drive/MyDrive/finalDataset/Dataset"
labelFile = "/content/drive/MyDrive/finalDataset/labels.csv"

# Constants
imageDimensions = (32, 32, 1)  # Grayscale reduces input channels
testRatio = 0.2
validationRatio = 0.2
batch_size = 32
epochs = 15  # Full training for best model
evaluation_epochs = 15  # Short runs during GA
noOfClasses = len(os.listdir(path))

# Load dataset
print("Loading Dataset...")
images, classNo = [], []
for class_id in range(noOfClasses):
    class_path = os.path.join(path, str(class_id))
    print(f"Processing class {class_id}...")
    for img_name in os.listdir(class_path):
        img = cv2.imread(os.path.join(class_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale early
        img = cv2.equalizeHist(img) / 255.0  # Normalize
        images.append(img)
        classNo.append(class_id)
    print(f"Class {class_id} loaded with {len(os.listdir(class_path))} images.")

# Convert to NumPy arrays
print("Converting dataset to NumPy arrays...")
images = np.array(images).reshape(-1, imageDimensions[0], imageDimensions[1], 1)
classNo = np.array(classNo)

# Split dataset
print("Splitting dataset into train, validation, and test sets...")
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validationRatio)
print("Dataset split completed.")

# Convert labels to categorical
print("Converting labels to categorical format...")
y_train = to_categorical(y_train, noOfClasses)
y_val = to_categorical(y_val, noOfClasses)
y_test = to_categorical(y_test, noOfClasses)

# Data Augmentation
print("Setting up data augmentation...")
dataGen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    rotation_range=10
)
dataGen.fit(X_train)
print("Data augmentation setup completed.")

print("Dataset Loaded Successfully!")

# Genetic Algorithm Parameters
population_size = 10
generations = 10

# Define ranges for parameters
param_ranges = {
    "conv1_filters": [32, 64, 128],
    "conv2_filters": [32, 64, 128],
    "kernel_size": [3, 5],
    "dense_units": [256, 512, 1024],
    "dropout_rate": [0.2, 0.3, 0.5],
    "learning_rate": [0.001, 0.0005, 0.0001]
}

print("Initializing Genetic Algorithm...")

def random_params():
    return {key: random.choice(values) for key, values in param_ranges.items()}

def create_model(params):
    print(f"Creating model with params: {params}")
    model = Sequential([
        Conv2D(params["conv1_filters"], (params["kernel_size"], params["kernel_size"]), activation='relu',
               input_shape=(imageDimensions[0], imageDimensions[1], 1)),
        Conv2D(params["conv2_filters"], (params["kernel_size"], params["kernel_size"]), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(params["dropout_rate"]),

        Flatten(),
        Dense(params["dense_units"], activation='relu'),
        Dropout(params["dropout_rate"]),
        Dense(noOfClasses, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def fitness(params):
    print(f"Evaluating Model: {params}")
    model = create_model(params)
    history = model.fit(
        dataGen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=evaluation_epochs,  
        verbose=1
    )
    max_val_acc = max(history.history['val_accuracy'])
    K.clear_session()
    del model
    gc.collect()
    print(f"Model achieved validation accuracy: {max_val_acc}")
    return max_val_acc

def crossover(parent1, parent2):
    child = {}
    for key in param_ranges.keys():
        child[key] = parent1[key] if random.random() < 0.5 else parent2[key]
    return child

def mutate(params):
    param_to_mutate = random.choice(list(param_ranges.keys()))
    params[param_to_mutate] = random.choice(param_ranges[param_to_mutate])
    return params

population = [random_params() for _ in range(population_size)]
for gen in range(generations):
    print(f"\n================== Generation {gen + 1}/{generations} ==================")
    fitness_scores = []
    for i, params in enumerate(population):
        print(f"\nEvaluating Individual {i + 1}/{population_size}")
        score = fitness(params)
        fitness_scores.append((params, score))
    fitness_scores.sort(key=lambda x: x[1], reverse=True)
    top_performers = fitness_scores[:population_size // 2]
    next_population = [params for params, _ in top_performers]
    while len(next_population) < population_size:
        parent1, parent2 = random.sample(top_performers, 2)
        child = crossover(parent1[0], parent2[0])
        if random.random() < 0.2:
            child = mutate(child)
        next_population.append(child)
    population = next_population
    gc.collect()

best_params = fitness_scores[0][0]
print("\n================== Best Model Training ==================")
print("Best Parameters:", best_params)
final_model = create_model(best_params)
history = final_model.fit(
    dataGen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),
    epochs=epochs,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
               ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)],
    verbose=1
)
score = final_model.evaluate(X_test, y_test, verbose=1)
print(f'\nFinal Test Score: {score[0]}')
print(f'Final Test Accuracy: {score[1]}')
final_model.save("/content/modelNew.h5")
print("\nOptimized Model saved successfully!")
