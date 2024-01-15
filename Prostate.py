
import os 
from pathlib import Path
import pandas as pd
import json


data_path = 'train'


files = [f for f in os.listdir(data_path) if f.startswith('case_')]


json_files = [js for file in files for js in os.listdir(f"train/{file}") if js.endswith('.json')]


data = []
for json_data in range(0, len(json_files), 2):
    with open(f'train/{json_files[json_data].split(".")[0]}/{json_files[json_data]}', 'r') as file:
        data.append(json.load(file))


risk_data = []
for json_data in range(0, len(json_files), 2):
    with open(f'train/{json_files[json_data].split(".")[0]}/{json_files[json_data+1]}', 'r') as file:
        risk_data.append(json.load(file))


data


clinical_df = pd.DataFrame(data)
score_df = pd.DataFrame(risk_data)


clinical_df, score_df


final_clinical_df = pd.merge(clinical_df, score_df, left_index=True, right_index=True)


final_clinical_df.rename(columns={0: 'risk'}, inplace=True)


final_clinical_df

 [markdown]
# Merging  Image and Clinical_data 

 [markdown]
# Training Model with Text data


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



final_clinical_df['risk'] = final_clinical_df['risk'].replace('Low', 0).replace('High', 1)


X = final_clinical_df.iloc[:, 1:-1]
y = final_clinical_df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=0)


y, X


# convert to float32 for normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


print('Build model...')
model = RandomForestRegressor()


model.fit(X_train, y_train)


preds_test=model.predict(X_test)
submission=pd.DataFrame({'Id':X_test.index,'risk':preds_test})


submission['risk'] = round(submission['risk'])


submission


model.score(X_train, y_train)

 [markdown]
# Trial 10

from skimage.transform import resize
from tensorflow.keras import layers, models
import tensorflow as tf
import nibabel as nilabel
import numpy as np


image_files = [f"train/{file}/{js}" for file in files for js in os.listdir(f"train/{file}") if js.endswith('.gz')]


labels = np.random.randint(0, 2, len(image_files))


train_files, test_files, train_labels, test_labels = train_test_split(image_files, labels, test_size=0.2, random_state=42)


def load_and_preprocess_nifti(file_path):
    nifti_image = nilabel.load(file_path)
    image = nifti_image.get_fdata()
    normalized_image_data = (image - np.min(image)) / (np.max(image) - np.min(image))
    target_shape = (32, 256, 24)
    resized_image = resize(normalized_image_data, target_shape, anti_aliasing=True)
    # normalized_image_data = normalized_image_data.reshape((32, 256, 24))
    return resized_image


X_train = np.array([load_and_preprocess_nifti(file) for file in train_files])
X_test = np.array([load_and_preprocess_nifti(file) for file in test_files])


# X_train = np.expand_dims(X_train, axis=-1)
# X_test = np.expand_dims(X_test, axis=-1)


y_train = tf.keras.utils.to_categorical(train_labels, num_classes=2)
y_test = tf.keras.utils.to_categorical(test_labels, num_classes=2)


model = models.Sequential([
    layers.Conv2D(16, 2, activation='tanh', input_shape=(32, 256, 24)),
    layers.Flatten(),
    layers.Dense(2, activation='tanh')
])


model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=100)



test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')
# 0.40677
# 0.47457
# 0.50847


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


preds = model.predict(X_test)


y_test


print(f1_score(np.argmax(y_test, axis=1), np.argmax(preds, axis=1), average="macro"))
print(precision_score(np.argmax(y_test, axis=1), np.argmax(preds, axis=1), average="macro"))
# print(precision_score(y_test, X_test, average="macro"))
print(recall_score(np.argmax(y_test, axis=1), np.argmax(preds, axis=1), average="macro"))


preds





