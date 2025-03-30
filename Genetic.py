import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import os

# Dataset Paths
path = r"D:\TrafficDataSetMain\Dataset"
labelFile = r"D:\TrafficDataSetMain\labels.csv"
imageDimensions = (32, 32, 3)
testRatio = 0.2
validationRatio = 0.2


def load_dataset():
    images, classNo = [], []
    myList = os.listdir(path)
    print("Total Classes Detected:", len(myList))
    for count, folder in enumerate(myList):
        myPicList = os.listdir(os.path.join(path, folder))
        for y in myPicList:
            curImg = cv2.imread(os.path.join(path, folder, y))
            images.append(curImg)
            classNo.append(count)
    images, classNo = np.array(images), np.array(classNo)
    return images, classNo, len(myList)


def preprocess_images(X):
    X = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in X])
    X = np.array([cv2.equalizeHist(img) for img in X])
    X = X / 255.0
    return X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)


def create_model(params, num_classes):
    model = Sequential()
    model.add(
        Conv2D(params['filters1'], (params['kernel1'], params['kernel1']), activation='relu', input_shape=(32, 32, 1)))
    model.add(Conv2D(params['filters2'], (params['kernel2'], params['kernel2']), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(params['dropout1']))
    model.add(Flatten())
    model.add(Dense(params['dense_units'], activation='relu'))
    model.add(Dropout(params['dropout2']))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=params['lr']), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def generate_random_params():
    return {
        'filters1': random.choice([32, 64, 128]),
        'kernel1': random.choice([3, 5]),
        'filters2': random.choice([32, 64, 128]),
        'kernel2': random.choice([3, 5]),
        'dropout1': random.uniform(0.2, 0.5),
        'dense_units': random.choice([128, 256, 512]),
        'dropout2': random.uniform(0.2, 0.5),
        'lr': random.choice([0.001, 0.0005, 0.0001]),
        'epochs': random.choice([5, 10, 15])
    }


def mutate_params(params):
    param_to_mutate = random.choice(list(params.keys()))
    old_value = params[param_to_mutate]
    params[param_to_mutate] = generate_random_params()[param_to_mutate]
    print(f"Mutation: {param_to_mutate} changed from {old_value} to {params[param_to_mutate]}")
    return params


def train_and_evaluate(params, X_train, y_train, X_val, y_val, num_classes):
    print(f"Training model with params: {params}")
    model = create_model(params, num_classes)
    model.fit(X_train, y_train, epochs=params['epochs'], validation_data=(X_val, y_val), batch_size=32, verbose=0)
    score = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Accuracy: {score[1]}")
    return score[1], model


def genetic_algorithm(X_train, y_train, X_val, y_val, num_classes, generations=5, population_size=10):
    population = [generate_random_params() for _ in range(population_size)]
    best_model, best_acc = None, 0

    for gen in range(generations):
        print(f"\nGeneration {gen + 1}/{generations}")
        scores = []
        models = []

        for i, params in enumerate(population):
            print(f"\nTraining Candidate {i + 1}/{population_size}")
            acc, model = train_and_evaluate(params, X_train, y_train, X_val, y_val, num_classes)
            scores.append((acc, params, model))
            models.append(model)

        scores.sort(reverse=True, key=lambda x: x[0])
        if scores[0][0] > best_acc:
            best_acc, best_model = scores[0][0], scores[0][2]

        print(f"Best model of Generation {gen + 1} with accuracy: {best_acc}")
        population = [scores[i][1] for i in range(population_size // 2)]
        for i in range(population_size // 2):
            new_params = mutate_params(population[i].copy())
            population.append(new_params)

    print(f"\nBest model found with accuracy: {best_acc}")
    best_model.save("best_model.h5")
    return best_model


# Load dataset
images, classNo, num_classes = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(images, classNo, test_size=testRatio)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validationRatio)

# Preprocess data
X_train, X_val, X_test = preprocess_images(X_train), preprocess_images(X_val), preprocess_images(X_test)
y_train, y_val, y_test = to_categorical(y_train, num_classes), to_categorical(y_val, num_classes), to_categorical(
    y_test, num_classes)

# Run Genetic Algorithm
best_model = genetic_algorithm(X_train, y_train, X_val, y_val, num_classes)
