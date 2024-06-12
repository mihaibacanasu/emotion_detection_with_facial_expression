import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constants (make sure these match your model's requirements)
IMG_SIZE = 48
GENDER_IMG_SIZE_W = 80
GENDER_IMG_SIZE_H = 110
AGE_IMG_SIZE = 200
NUM_CLASSES_EMOTION = 7
NUM_CLASSES_AGE = 7

# Load the emotion detection model
emotion_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES_EMOTION, activation='softmax')
])

try:
    emotion_model.load_weights('model.h5')
except Exception as e:
    print(f"Error loading emotion model weights: {e}")
    exit()

# Load the gender classification model with the exact architecture
def create_gender_model():
    model = Sequential()
    model.add(layers.Conv2D(16, (8, 8), strides=(4, 4), padding='valid', input_shape=(GENDER_IMG_SIZE_W, GENDER_IMG_SIZE_H, 1)))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(32, (5, 5), padding="same"))
    model.add(layers.Activation('relu'))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(512))
    model.add(layers.Dropout(0.1))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(2))
    model.add(layers.Activation('softmax'))
    return model

gender_model = create_gender_model()

try:
    gender_model.load_weights('gender_model.h5')
except Exception as e:
    print(f"Error loading gender model weights: {e}")
    exit()

# Load the age estimation model with the exact architecture
def create_age_model():
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(AGE_IMG_SIZE, AGE_IMG_SIZE, 1)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(132, activation='relu'))
    model.add(layers.Dense(NUM_CLASSES_AGE, activation='softmax'))
    return model

age_model = create_age_model()

try:
    age_model.load_weights('age_model.h5')
except Exception as e:
    print(f"Error loading age model weights: {e}")
    exit()

# Prevent OpenCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# Dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Dictionary for gender classification
gender_dict = {0: "Female", 1: "Male"}

# Dictionary for age classification
age_dict = {0: "1-2", 1: "3-9", 2: "10-20", 3: "21-27", 4: "28-45", 5: "46-65", 6: "66-116"}

# Initialize Haar Cascade for face detection
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Start the webcam feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for emotion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            
            # Extract region of interest (ROI) for emotion detection
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE)), -1), 0)
            
            # Extract region of interest (ROI) for gender classification
            roi_color = gray[y:y + h, x:x + w]
            cropped_img_gender = np.expand_dims(np.expand_dims(cv2.resize(roi_color, (GENDER_IMG_SIZE_W, GENDER_IMG_SIZE_H)), -1), 0)
            
            # Extract region of interest (ROI) for age estimation
            roi_age = gray[y:y + h, x:x + w]
            cropped_img_age = np.expand_dims(np.expand_dims(cv2.resize(roi_age, (AGE_IMG_SIZE, AGE_IMG_SIZE)), -1), 0)
            
            # Predict emotion
            emotion_prediction = emotion_model.predict(cropped_img)
            emotion_maxindex = int(np.argmax(emotion_prediction))
            emotion_label = emotion_dict[emotion_maxindex]

            # Predict gender
            gender_prediction = gender_model.predict(cropped_img_gender)
            gender_maxindex = np.argmax(gender_prediction[0])
            gender_label = gender_dict[gender_maxindex]

            # Predict age
            age_prediction = age_model.predict(cropped_img_age)
            age_maxindex = np.argmax(age_prediction[0])
            age_label = age_dict[age_maxindex]

            # Display the emotion above the rectangle
            cv2.putText(frame, f"{emotion_label}", (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Display the gender to the left of the rectangle
            cv2.putText(frame, f"{gender_label}", (x - 70, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Display the age to the right of the rectangle
            cv2.putText(frame, f"Age: {age_label}", (x + w + 20, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
