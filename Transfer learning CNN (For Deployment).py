"""
This is good for deployment, not for research.
Another than this is just for research
If you wanna goog model, first do transfer learning, after that you do a regular research model, to warm up GPU model
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#---------------------------------------------------------------------------------------------------------

tf.keras.utils.get_file('dataset.zip', 'dataset_link')

#---------------------------------------------------------------------------------------------------------

import zipfile
zip_ref = zipfile.ZipFile('dataset.zip', 'r')
zip_ref.extractall('files')
zip_ref.close()

#---------------------------------------------------------------------------------------------------------

base_model=VGG16(weights='imagenet',include_top=False)

#---------------------------------------------------------------------------------------------------------
########## You can use this
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen =ImageDataGenerator(
    preprocessing_function=preprocess_input, # Use this preprocess input for transfer learning!
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest', 
    validation_split=0.1
)

#///////////////////////////////////////////////////////////////////////////////////////////////////////
#///////////////////////////////////////////////////////////////////////////////////////////////////////

########## You can use this too!
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
    validation_split=0.2
)

#---------------------------------------------------------------------------------------------------------

IMG_SIZE = 300

train_generator = train_datagen.flow_from_directory(
    '/content/temporary',             # Source directory
    target_size=(IMG_SIZE,IMG_SIZE),  # Resizes images
    batch_size=32,
    class_mode='categorical',         # Remember to change based the class
    subset = 'training'
)
    
validation_generator = train_datagen.flow_from_directory(
    '/content/temporary',
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
  
#---------------------------------------------------------------------------------------------------------

from tensorflow.keras.models import Model

CLASSES = 4 # Output class
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.Dropout(0.4)(x)
predictions = tf.keras.layers.Dense(CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#---------------------------------------------------------------------------------------------------------

dot_img_file = 'example.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

#---------------------------------------------------------------------------------------------------------

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') > 0.95):
      print("\nAccuracy better than target training!")
      self.model.stop_training = True

callbacks = myCallback()

#---------------------------------------------------------------------------------------------------------
%%time
history = model.fit(
    train_generator,
    epochs = 30,
    validation_data = validation_generator,
    callbacks=[callbacks]
)

#---------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#---------------------------------------------------------------------------------------------------------

akurasi_test_generator = train_datagen.flow_from_directory(
  '/content/TMP',
  target_size=(IMG_SIZE,IMG_SIZE),
  batch_size=32,
  class_mode='categorical',
)

results = model.evaluate(akurasi_test_generator)
print("test loss, test acc:", results)

#---------------------------------------------------------------------------------------------------------
# For saving model!
#---------------------------------------------------------------------------------------------------------

model.save("Model_Name.h5")

#---------------------------------------------------------------------------------------------------------
# If there is no selu activation

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
    
///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////

# If use the selu activation
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, 
  tf.lite.OpsSet.SELECT_TF_OPS 
]

tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

#---------------------------------------------------------------------------------------------------------
# To create confusion matrix!

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

target_names = []
for key in akurasi_test_generator.class_indices:
    target_names.append(key)

testing_generator = train_datagen.flow_from_directory(
  '/content/Temporary',
  target_size=(IMG_SIZE,IMG_SIZE),
  batch_size=32,
  class_mode='categorical',
  subset='validation',
  shuffle=False             # Use this shuffle for second training to get more good accuracy. This shuffle just able only for validation set!
)

#Confution Matrix and Classification Report

testing_generator.reset()
Y_predi = model.predict(testing_generator, total_image_validation //batches+1)
y_predi = np.argmax(Y_predi, axis=1)
print('Confusion Matrix')
print(confusion_matrix(testing_generator.classes, y_predi))
print('Classification Report')
print(classification_report(testing_generator.classes, y_predi, target_names=target_names))

#---------------------------------------------------------------------------------------------------------

from keras.preprocessing import image
import numpy as np

new_image = image.load_img('/content/a (2).jpeg', \
                           target_size = (IMG_SIZE,IMG_SIZE))
new_image

new_image = image.img_to_array(new_image)
new_image = np.expand_dims(new_image, axis = 0)

result = model.predict(new_image)
result_final = np.argmax(result)
result_final

#---------------------------------------------------------------------------------------------------------

# Reset if there is stack of RAM equipment
from numba import cuda
cuda.select_device(0)
cuda.close()
