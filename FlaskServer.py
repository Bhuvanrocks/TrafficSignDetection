from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load Model
model_path = r"D:\TrafficSignModel\TrafficModel.h5"
model = load_model(model_path)

# Ensure Uploads Directory Exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def equalize(img):
    return cv2.equalizeHist(img)


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0  # Normalize
    return img


def getClassName(classNo):
    class_labels = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution',
        'Dangerous curve to the left', 'Dangerous curve to the right', 'Double curve',
        'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals',
        'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
        'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
        'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_labels[classNo] if classNo < len(class_labels) else "Unknown"


def model_predict(img_path, model):
    print(f"Processing image: {img_path}")

    img = image.load_img(img_path, target_size=(32, 32))
    img = np.asarray(img)
    img = preprocessing(img)

    img = img.reshape(1, 32, 32, 1)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions)  # Fix: Replaced deprecated `predict_classes`

    result = getClassName(classIndex)
    return result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    f = request.files['file']
    if f.filename == '':
        return "No selected file", 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename))
    f.save(file_path)

    preds = model_predict(file_path, model)
    return preds


if __name__ == '__main__':
    app.run(port=5001, debug=True)
