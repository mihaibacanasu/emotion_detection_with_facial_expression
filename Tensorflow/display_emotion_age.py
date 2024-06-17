import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import os

# Suprima mesajele de logare TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Constante legate de numarul de pixeli ai imaginilor (inaltime si latime)
IMG_SIZE = 48
AGE_IMG_SIZE = 200
NUM_CLASSES_EMOTION = 7
NUM_CLASSES_AGE = 7

# Incarca modelul de detectare a emotiilor
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
    emotion_model.load_weights('models/model.h5')
except Exception as e:
    print(f"Error loading emotion model weights: {e}")
    exit()

# Incarca modelul de estimare a varstei cu aceeasi arhitectura
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
    age_model.load_weights('models/age_model.h5')
except Exception as e:
    print(f"Error loading age model weights: {e}")
    exit()

# Previne utilizarea OpenCL si mesajele de logare inutile
cv2.ocl.setUseOpenCL(False)

# Dictionar care asociaza fiecare eticheta cu o emotie (ordine alfabetica)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Dictionar pentru clasificarea varstei
age_dict = {0: "1-2", 1: "3-9", 2: "10-20", 3: "21-27", 4: "28-45", 5: "46-65", 6: "66-116"}

# Initializeaza Haar Cascade pentru detectarea fetelor
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Porneste fluxul webcam-ului
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Nu s-a putut deschide camera web.")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converteste cadrul in tonuri de gri pentru detectarea emotiilor
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Deseneaza un dreptunghi in jurul fetei
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            
            # Extrage regiunea de interes (ROI) pentru detectarea emotiilor
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (IMG_SIZE, IMG_SIZE)), -1), 0)
            
            # Extrage regiunea de interes (ROI) pentru estimarea varstei
            roi_age = gray[y:y + h, x:x + w]
            cropped_img_age = np.expand_dims(np.expand_dims(cv2.resize(roi_age, (AGE_IMG_SIZE, AGE_IMG_SIZE)), -1), 0)
            
            # Prezice emotia
            emotion_prediction = emotion_model.predict(cropped_img)
            emotion_maxindex = int(np.argmax(emotion_prediction))
            emotion_label = emotion_dict[emotion_maxindex]

            # Prezice varsta
            age_prediction = age_model.predict(cropped_img_age)
            age_maxindex = np.argmax(age_prediction[0])
            age_label = age_dict[age_maxindex]

            # Afiseaza emotia deasupra dreptunghiului
            cv2.putText(frame, f"{emotion_label}", (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # Afiseaza varsta in dreapta dreptunghiului
            cv2.putText(frame, f"Age: {age_label}", (x + w + 20, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Afiseaza cadrul rezultat
        cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
        
        # Intrerupe bucla daca se apasa 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"A aparut o eroare: {e}")
finally:
    # Elibereaza captura si inchide ferestrele
    cap.release()
    cv2.destroyAllWindows()
