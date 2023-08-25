# -*- coding: utf-8 -*-

# Import the necessary packages

import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import PIL
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import *
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.utils import plot_model
from IPython.display import display
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, LearningRateScheduler
import cv2


os.listdir(r'C:\Users\hp\PycharmProjects\pythonProject\train')
os.listdir(os.path.join('train', 'Mild'))
# Check the number of images in the dataset
train = []
label = []

# os.listdir returns the list of files in the folder, in this case image class names
for i in os.listdir('./train'):
  train_class = os.listdir(os.path.join('train', i))
  for j in train_class:
    img = os.path.join('train', i, j)
    train.append(img)
    label.append(i)

print('Number of train images : {} \n'.format(len(train)))

retina_df = pd.DataFrame({'Image': train,'Labels': label})
# Shuffle the data and split it into training and testing
retina_df = shuffle(retina_df)
train, test = train_test_split(retina_df, test_size = 0.2)






from PIL import Image, ImageEnhance
# Define a function to apply a green filter to an image
def apply_green_filter(image):
    # Convert the image to a numpy array
    img_array = np.array(image)

    # Apply the green filter by setting the red and blue channels to zero
    img_array[:, :, 0] = 0  # Red channel
    img_array[:, :, 2] = 0  # Blue channel

    # Convert the modified numpy array back to an image
    green_filtered_img = Image.fromarray(img_array)

    return green_filtered_img

def apply_clahe(image):
    img_array = np.array(image)

    # Split the input image into color channels
    r_channel = img_array[:, :, 0]
    g_channel = img_array[:, :, 1]
    b_channel = img_array[:, :, 2]

    # Apply CLAHE to the green channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_green_channel = clahe.apply(g_channel)
    # Reconstruct the image with enhanced green channel and original red and blue channels
    enhanced_img = np.stack((r_channel, enhanced_green_channel, b_channel), axis=-1)

    return Image.fromarray(enhanced_img)


def apply_illumination_correction(image, factor=1.5):
    # Open the image using PIL
    img = Image.open(image)

    # Enhance the brightness of the image
    enhancer = ImageEnhance.Brightness(img)
    corrected_img = enhancer.enhance(factor)

    return corrected_img

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize to desired input size
    img = apply_green_filter(img)
    img = apply_clahe(img)

    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    return img_array


# Step 3: Preprocess Images
train_images = [preprocess_image(image_path) for image_path in train['Image']]
test_images = [preprocess_image(image_path) for image_path in test['Image']]

from keras.utils import to_categorical
# Step 4: Convert Labels
from sklearn.preprocessing import LabelEncoder


# Create a label encoder
label_encoder = LabelEncoder()


# Fit the encoder on the train labels and transform both train and test labels
train_labels_encoded = label_encoder.fit_transform(train['Labels'])
test_labels_encoded = label_encoder.transform(test['Labels'])

# Convert encoded labels to one-hot encoded format
train_labels = to_categorical(train_labels_encoded)
test_labels = to_categorical(test_labels_encoded)

from keras.preprocessing.image import ImageDataGenerator

# Define augmentation settings
augmentation_settings = {
    "rescale": 1./255,
    "shear_range": 0.2,
    "zoom_range": 0.2,
    "horizontal_flip": True,
    "rotation_range": 20
}

# Create an instance of ImageDataGenerator
datagen = ImageDataGenerator(**augmentation_settings, validation_split=0.15)
# Load and preprocess images
train_images_array = np.array([np.array(img) for img in train_images])
train_labels_array = np.array(train_labels)

test_images_array = np.array([np.array(img) for img in test_images])
test_labels_array = np.array(test_labels)
# Generate augmented batches of data
augmented_generator_train = datagen.flow(
    train_images_array,
    train_labels_array,
    batch_size=32,
    shuffle=True
)
augmented_generator_test = datagen.flow(
    test_images_array,
    test_labels_array,
    batch_size=32,
    shuffle=True
)








def res_block(X, filter, stage):

  # Convolutional_block
  X_copy = X

  f1 , f2, f3 = filter

  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_conv_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = MaxPool2D((2,2))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_a')(X)
  X = Activation('relu')(X)

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_conv_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_b')(X)
  X = Activation('relu')(X)

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_c')(X)


  # Short path
  X_copy = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_conv_copy', kernel_initializer= glorot_uniform(seed = 0))(X_copy)
  X_copy = MaxPool2D((2,2))(X_copy)
  X_copy = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_conv_copy')(X_copy)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 1
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_1_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_a')(X)
  X = Activation('relu')(X)

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_1_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_b')(X)
  X = Activation('relu')(X)

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_1_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_1_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  # Identity Block 2
  X_copy = X


  # Main Path
  X = Conv2D(f1, (1,1),strides = (1,1), name ='res_'+str(stage)+'_identity_2_a', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_a')(X)
  X = Activation('relu')(X)

  X = Conv2D(f2, kernel_size = (3,3), strides =(1,1), padding = 'same', name ='res_'+str(stage)+'_identity_2_b', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_b')(X)
  X = Activation('relu')(X)

  X = Conv2D(f3, kernel_size = (1,1), strides =(1,1),name ='res_'+str(stage)+'_identity_2_c', kernel_initializer= glorot_uniform(seed = 0))(X)
  X = BatchNormalization(axis =3, name = 'bn_'+str(stage)+'_identity_2_c')(X)

  # ADD
  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X

input_shape = (256,256,3)

#Input tensor shape
X_input = Input(input_shape)

#Zero-padding

X = ZeroPadding2D((3,3))(X_input)

# 1 - stage

X = Conv2D(64, (7,7), strides= (2,2), name = 'conv1', kernel_initializer= glorot_uniform(seed = 0))(X)
X = BatchNormalization(axis =3, name = 'bn_conv1')(X)
X = Activation('relu')(X)
X = MaxPooling2D((3,3), strides= (2,2))(X)

# 2- stage

X = res_block(X, filter= [64,64,256], stage= 2)

# 3- stage

X = res_block(X, filter= [128,128,512], stage= 3)

# 4- stage

X = res_block(X, filter= [256,256,1024], stage= 4)

# # 5- stage

# X = res_block(X, filter= [512,512,2048], stage= 5)

#Average Pooling

X = AveragePooling2D((2,2), name = 'Averagea_Pooling')(X)

#Final layer

X = Flatten()(X)
X = Dense(5, activation = 'softmax', name = 'Dense_final', kernel_initializer= glorot_uniform(seed=0))(X)


model = Model( inputs= X_input, outputs = X, name = 'Resnet18')

model.summary()

nadam_optimizer = Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

"""# TASK #7: COMPILE AND TRAIN DEEP LEARNING MODEL"""

model.compile(optimizer = nadam_optimizer , loss = 'categorical_crossentropy', metrics= ['accuracy'])

#using early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

#save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)



# Set the tracking URI (optional if using a local server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

from keras.callbacks import EarlyStopping, ModelCheckpoint
with mlflow.start_run():
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("learning_rate", 0.001)
    # Split data into train and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.15,
                                                                          random_state=42)

    history = model.fit(augmented_generator_train, steps_per_epoch=augmented_generator_train.n // 32, validation_data=augmented_generator_train,
                        validation_steps=augmented_generator_train.n // 32, epochs=1, callbacks=[checkpointer, earlystopping])

    # Log training accuracy, validation accuracy, training loss, and validation loss
    mlflow.log_metric("training_accuracy", history.history['accuracy'][-1])
    mlflow.log_metric("validation_accuracy", history.history['val_accuracy'][-1])
    mlflow.log_metric("training_loss", history.history['loss'][-1])
    mlflow.log_metric("validation_loss", history.history['val_loss'][-1])

    # Save your model
    model.save("model.h5")
    # Log the saved model as an artifact
    mlflow.log_artifact("model.h5")
# TASK #8: ASSESS THE PERFORMANCE OF THE TRAINED MODEL
    model.load_weights(r'C:\Users\hp\PycharmProjects\pythonProject\weights.hdf5')
# Evaluate the performance of the model
    # Evaluate the model
    evaluate = model.evaluate(augmented_generator_test, steps = augmented_generator_test.n // 32, verbose =1)
    mlflow.log_metric("test_accuracy", evaluate[1])
    print("Test Loss:", evaluate[0])
    print("Test Accuracy:", evaluate[1])



# Launch the MLflow UI (in a separate terminal or command prompt)
# Run: mlflow ui

# End the MLflow run
mlflow.end_run()


# Assigning label names to the corresponding indexes
labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}

# Loading images and their predictions

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
# import cv2

prediction = []
original = []
image = []
count = 0

for item in range(len(test)):
  # code to open the image
  img= PIL.Image.open(test['Image'].tolist()[item])
  # resizing the image to (256,256)
  img = img.resize((256,256))
  # Applying the green filter
  img = apply_green_filter(img)
  img = apply_clahe(img)

  # appending image to the image list
  image.append(img)
  # converting image to array
  img = np.asarray(img, dtype= np.float32)
  # normalizing the image
  img = img / 255
  # reshaping the image in to a 4D array
  img = img.reshape(-1,256,256,3)
  # making prediction of the model
  predict = model.predict(img)
  # getting the index corresponding to the highest value in the prediction
  predict = np.argmax(predict)
  # appending the predicted class to the list
  prediction.append(labels[predict])
  # appending original class to the list
  original.append(test['Labels'].tolist()[item])



# Getting the test accuracy
score = accuracy_score(original, prediction)
print("Test Accuracy : {}".format(score))

# Print out the classification report
print(classification_report(np.asarray(original), np.asarray(prediction)))
















