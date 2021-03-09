import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    horizontal_flip = True,
    shear_range = 0.2,
    fill_mode = 'nearest',
    validation_split=0.1 # 0.25 is standard between 0.3 and 0.2
)

IMG_SIZE = 300

train_generator = train_datagen.flow_from_directory(
  '/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals',  # Source directory
  target_size=(IMG_SIZE,IMG_SIZE),  # Resizes images
  batch_size=32,
  class_mode='categorical',subset = 'training')
    
validation_generator = train_datagen.flow_from_directory(
  '/kaggle/input/animal-image-datasetdog-cat-and-panda/animals/animals',
  target_size=(IMG_SIZE,IMG_SIZE),
  batch_size=32,
  class_mode='categorical',
  subset='validation')

# Remember! For best result, need training untill get callbacks more than 0.9700 !
# This isn't VGG16 or etc

--------------------------------------------------------------------------------

# Or like this can be good too!
# In this trick, you can't use random fool model. You should use special model, or you will face overfitting

datagen = ImageDataGenerator(
    rescale=1./255.,
    horizontal_flip=True,
    brightness_range=[0.4,1.5],
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.1
)

IMG_SIZE = 300

train_generator = datagen.flow_from_directory(
    '/content/files',
    batch_size=32,
    class_mode='categorical',
    target_size=(IMG_SIZE,IMG_SIZE),
    color_mode='rgb',
    subset='training'
)
validation_generator = datagen.flow_from_directory(
    '/content/files',
    batch_size=32,
    class_mode='categorical',
    target_size=(IMG_SIZE,IMG_SIZE),
    subset='validation'
)
