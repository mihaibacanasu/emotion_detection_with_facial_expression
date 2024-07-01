import tensorflow as tf
import numpy as np
import argparse
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.regularizers import l2

# Functia pentru plotarea istoricului antrenamentului
def plot_model_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig('model_data_loss.png')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.savefig('model_data_accuracy.png')
    plt.show()

# Definirea cailor pentru seturile de date
train_dir = 'data/train'
val_dir = 'data/test'

# Parametrii
num_train = 18323
num_val = 4583
batch_size = 64
num_epoch = 50
IMG_SIZE = 48
NUM_CLASSES = 7

# Crearea generatorilor de date
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(IMG_SIZE, IMG_SIZE),
                                                    batch_size=batch_size,
                                                    color_mode='grayscale',
                                                    class_mode='categorical')
validation_generator = val_datagen.flow_from_directory(val_dir,
                                                target_size=(IMG_SIZE, IMG_SIZE),
                                                batch_size=batch_size,
                                                color_mode='grayscale',
                                                class_mode='categorical')

# Definirea modelului
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.01), input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer=l2(0.01)),
    BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compilarea modelului
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='auto',
    restore_best_weights=True
)

# Antrenarea modelului
model_info = model.fit(train_generator,
          steps_per_epoch=train_generator.n // batch_size,
          epochs=num_epoch,
          validation_data=validation_generator,
          validation_steps=validation_generator.n // batch_size)

# Salvarea greutatilor modelului cu extensia corecta
model.save_weights('model1_data.weights.h5')
plot_model_history(model_info)

# Generarea predictiilor si etichetelor reale pentru setul de date de testare
validation_generator.reset()
predictions = model.predict(validation_generator, steps=validation_generator.n // batch_size + 1)
y_pred = np.argmax(predictions, axis=1)
y_true = validation_generator.classes

# Calcularea matricii de confuzie
cm = confusion_matrix(y_true, y_pred)

# Vizualizarea matricii de confuzie
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(validation_generator.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.show()
