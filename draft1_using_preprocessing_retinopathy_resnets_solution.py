# -*- coding: utf-8 -*-

# Import the necessary packages

import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
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







new_train_directory = r'C:\Users\hp\PycharmProjects\pythonProject\new_train'
new_test_directory = r'C:\Users\hp\PycharmProjects\pythonProject\new_test'




"""
# Get the list of subdirectories within the new train directory
new_train_subdirectories = os.listdir(new_train_directory)

# Get the list of subdirectories within the new test directory
new_test_subdirectories = os.listdir(new_test_directory)

# Print the number of instances in each class within the new train directory
for class_name in new_train_subdirectories:
    class_dir = os.path.join(new_train_directory, class_name)
    num_instances = len(os.listdir(class_dir))
    print(f"Train: Class '{class_name}' has {num_instances} instances.")
# Print the number of instances in each class within the new test directory
for class_name in new_test_subdirectories:
    class_dir = os.path.join(new_test_directory, class_name)
    num_instances = len(os.listdir(class_dir))
    print(f"Test: Class '{class_name}' has {num_instances} instances.")

"""
# TASK #3: PERFORM DATA EXPLORATION AND DATA VISUALIZATION"""







"""# TASK #4: PERFORM DATA AUGMENTATION AND CREATE DATA GENERATOR"""




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

# Custom image generator with green filter
class GreenFilteredImageDataGenerator(ImageDataGenerator):
    def __init__(self, green_filter=True,clahe=True,illumination_correction=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.green_filter = green_filter
        self.clahe = clahe
        self.illumination_correction = illumination_correction

    def load_img(self, path, color_mode='rgb', **kwargs):
        img = super().load_img(path, color_mode=color_mode, **kwargs)
        if self.green_filter:
            img = apply_green_filter(img)
        if self.clahe:
            img = apply_clahe(img)
        if self.clahe:
            img = apply_illumination_correction(img)
        return img



# For training datagenerator, we add normalization, shear angle, zooming range and horizontal flip
# Define augmentation settings*
# Update augmentation settings to include the green filter
augmentation_settings = {
    "rescale": 1./255,
    "shear_range": 0.2,
    "validation_split": 0.15
}

# Create run-time augmentation on training and test dataset with green filter
train_datagen = GreenFilteredImageDataGenerator(**augmentation_settings,green_filter=True,clahe=True,horizontal_flip=True)
test_datagen = GreenFilteredImageDataGenerator(green_filter=True, clahe=True , rescale=1.0/255 )

train_generator = train_datagen.flow_from_directory(
    new_train_directory,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

test_generator = test_datagen.flow_from_directory(
    new_test_directory,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical'
)





"""# TASK #6: BUILD RES-BLOCK BASED DEEP LEARNING MODEL"""

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

    history = model.fit(train_generator, steps_per_epoch = train_generator.n // 32, validation_data= train_generator, validation_steps= train_generator.n // 32, epochs = 1 , callbacks=[checkpointer , earlystopping])
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
    evaluate = model.evaluate(test_generator, steps = test_generator.n // 32, verbose =1)
    mlflow.log_metric("test_accuracy", evaluate[1])

    print('Accuracy Test : {}'.format(evaluate[1]))


# Launch the MLflow UI (in a separate terminal or command prompt)
# Run: mlflow ui

# End the MLflow run
mlflow.end_run()


# Assigning label names to the corresponding indexes
labels = {0: 'Mild', 1: 'Moderate', 2: 'No_DR', 3:'Proliferate_DR', 4: 'Severe'}

os.listdir('./new_test')
os.listdir(os.path.join('new_test', 'Mild'))

# Check the number of images in the dataset
test = []
label = []

# os.listdir returns the list of files in the folder, in this case image class names
for i in os.listdir('./new_test'):
  train_class = os.listdir(os.path.join('new_test', i))
  for j in train_class:
    img = os.path.join('new_test', i, j)
    test.append(img)
    label.append(i)


print('Number of test images:', len(test))

# Create a list of dictionaries
data = [{'Image': img, 'Labels': lbl} for img, lbl in zip(test, label)]

# Convert the list of dictionaries into a Pandas DataFrame
new_test = pd.DataFrame(data)

print('Number of test images:', len(new_test))

# Loading images and their predictions

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from PIL import Image
# import cv2

prediction = []
original = []
image = []
count = 0

for item in range(len(new_test)):
  # code to open the image
  img= PIL.Image.open(new_test['Image'].tolist()[item])
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
  original.append(new_test['Labels'].tolist()[item])



# Getting the test accuracy
score = accuracy_score(original, prediction)
print("Test Accuracy : {}".format(score))

# Print out the classification report
print(classification_report(np.asarray(original), np.asarray(prediction)))

















# Print out the classification report
print(classification_report(np.asarray(original), np.asarray(prediction)))

# plot the confusion matrix
plt.figure(figsize = (20,20))
cm = confusion_matrix(np.asarray(original), np.asarray(prediction))
ax = plt.subplot()
sns.heatmap(cm, annot = True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Original')
ax.set_title('Confusion_matrix')



#model.save('your_model.h5')



