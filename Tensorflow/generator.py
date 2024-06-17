from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model
import os

# Ensure the PATH environment variable is correctly set
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz\\bin'


# define the CNN model
def define_cnn_model(input_shape=(48, 48, 1), num_classes=7):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    
    # image input
    in_image = Input(shape=input_shape)
    
    # Convolutional layer 1
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer=init)(in_image)
    
    # Convolutional layer 2
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer=init)(conv1)
    
    # MaxPooling layer 1
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Dropout layer 1
    drop1 = Dropout(0.25)(pool1)
    
    # Convolutional layer 3
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=init)(drop1)
    
    # MaxPooling layer 2
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Convolutional layer 4
    conv4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer=init)(pool2)
    
    # MaxPooling layer 3
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    # Dropout layer 2
    drop2 = Dropout(0.25)(pool3)
    
    # Flatten layer
    flat = Flatten()(drop2)
    
    # Dense layer 1
    dense1 = Dense(1024, activation='relu', kernel_initializer=init)(flat)
    
    # Dropout layer 3
    drop3 = Dropout(0.5)(dense1)
    
    # Output layer
    out_layer = Dense(num_classes, activation='softmax', kernel_initializer=init)(drop3)
    
    # define model
    model = Model(in_image, out_layer)
    return model


# define input shape and number of classes
input_shape = (48, 48, 1)
num_classes = 7

# create the model
model = define_cnn_model(input_shape, num_classes)

# summarize the model
model.summary()

# plot the model
plot_model(model, to_file='cnn_model_plot_generator.png', show_shapes=True, show_layer_names=True)
