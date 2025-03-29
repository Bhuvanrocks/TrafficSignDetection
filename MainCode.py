import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

#############################################

threshold = 0.75  # PROBABLITY THRESHOLD
font = cv2.FONT_HERSHEY_SIMPLEX
##############################################

# IMPORT THE TRAINED MODEL
model_path = r"path/model"
if not model_path:
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model = load_model(model_path)
print("Model loaded successfully!")


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def equalize(img):
    return cv2.equalizeHist(img)

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255.0
    return img

def getClassName(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h', 'Speed Limit 60 km/h',
        'Speed Limit 70 km/h', 'Speed Limit 80 km/h', 'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h',
        'Speed Limit 120 km/h', 'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop', 'No vehicles',
        'Vehicles over 3.5 metric tons prohibited', 'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
        'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing',
        'Beware of ice/snow', 'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left',
        'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[classNo] if 0 <= classNo < len(class_names) else "Unknown"

# IMAGE PATH
image_path = r"path/img"  # Change this to your image path

# READ IMAGE
imgOriginal = cv2.imread(image_path)
if imgOriginal is None:
    raise FileNotFoundError(f"Image '{image_path}' not found!")

# RESIZE IMAGE TO SLIGHTLY SMALLER SIZE
imgOriginal = cv2.resize(imgOriginal, (700, 700))

# PROCESS IMAGE
img = cv2.resize(imgOriginal, (32, 32))
img = preprocessing(img)
cv2.imshow("Processed Image", img)
img = img.reshape(1, 32, 32, 1)

# PREDICT IMAGE
predictions = model.predict(img)
classIndex = np.argmax(predictions)
probabilityValue = np.amax(predictions)

# DISPLAY RESULTS IN TOP LEFT
cv2.putText(imgOriginal, f"CLASS: {classIndex} {getClassName(classIndex)}", (20, 50), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
cv2.putText(imgOriginal, f"PROBABILITY: {round(probabilityValue * 100, 2)}%", (20, 100), font, 1.5, (0, 0, 255), 3, cv2.LINE_AA)
cv2.imshow("Result", imgOriginal)

cv2.waitKey(0)
cv2.destroyAllWindows()
