# Dataset : https://www.kaggle.com/uciml/news-aggregator-dataset
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.models import Sequential
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import os

-------------------------------------------------------------------------------------------------------

# Just use 2 column, the object, and the classes
# This is the valid, structure is that left column is the object, and the right is the classes
df = pd.read_csv('/content/TMP_3/uci-news-aggregator.csv', usecols=['TITLE', 'CATEGORY'])

# This's how to replace the column! U should use this if the dataframe isn't same with the structure before!
df=df[['CATEGORY','TITLE']]

-------------------------------------------------------------------------------------------------------

# Check the NaN value
df['TITLE'].isnull().values.any()

-------------------------------------------------------------------------------------------------------

# Drop the NaN value
df.dropna()

-------------------------------------------------------------------------------------------------------

# Left side to total of classes, right side to all data that included into the classes before
df.CATEGORY.value_counts()

-------------------------------------------------------------------------------------------------------

df.head(10)

-------------------------------------------------------------------------------------------------------

category = pd.get_dummies(df.CATEGORY) # Create the new system!
category

-------------------------------------------------------------------------------------------------------

df_baru = pd.concat([df, category], axis=1)
df_baru

-------------------------------------------------------------------------------------------------------

df_baru = df_baru.drop(columns='CATEGORY')
df_baru

-------------------------------------------------------------------------------------------------------

text = df_baru['TITLE'].values # Convert to value
text

-------------------------------------------------------------------------------------------------------

label = df_baru[['e', 'b', 't', 'm']].values # Look at the classes
label

-------------------------------------------------------------------------------------------------------

# Split them
from sklearn.model_selection import train_test_split
text_latih, text_test, label_latih, label_test = train_test_split(text, label, test_size=0.2)

-------------------------------------------------------------------------------------------------------

# Tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
tokenizer = Tokenizer(num_words=5000, oov_token='x')

-------------------------------------------------------------------------------------------------------

tokenizer.fit_on_texts(text_latih)
tokenizer.fit_on_texts(text_test)

sekuens_latih = tokenizer.texts_to_sequences(text_latih)
sekuens_test = tokenizer.texts_to_sequences(text_test)

-------------------------------------------------------------------------------------------------------

padded_latih = pad_sequences(sekuens_latih,maxlen = 400) # Make 400, to avoid crazy user. If you wanna change it? You can change it
padded_latih

-------------------------------------------------------------------------------------------------------

padded_test = pad_sequences(sekuens_test,maxlen=400)
padded_test

-------------------------------------------------------------------------------------------------------

import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000,output_dim=18),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

-------------------------------------------------------------------------------------------------------

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy') > 1.0):
      print("\nValidasi akurasi di atas 75%, hentikan training!")
      self.model.stop_training = True

callbacks = myCallback()

-------------------------------------------------------------------------------------------------------

%%time
num_epochs = 1
history = model.fit(padded_latih, label_latih, epochs=num_epochs, 
                    validation_data=(padded_test, label_test),callbacks=[callbacks])
                    
-------------------------------------------------------------------------------------------------------

# Before use the prediction model, first retokenize!
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
tokenizer = Tokenizer(num_words=5000, oov_token='x')



txt = ["Regular fast food eating linked to fertility issues in women"]
seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=400)
pred = model.predict(padded)
labels = ['entertainment', 'bussiness', 'science/tech', 'health']
print(pred, labels[np.argmax(pred)])
