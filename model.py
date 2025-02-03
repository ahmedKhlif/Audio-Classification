import os
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, LSTM
from tensorflow.keras.layers import Dropout, Dense, TimeDistributed
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
from python_speech_features import mfcc
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint
from cfg import Config

def load_cached_data():
    if os.path.isfile(config.pickle_path):
        print(f"Loading existing data for {config.mode} model")
        with open(config.pickle_path, 'rb') as handle:
            cached_data = pickle.load(handle)
            return cached_data
    return None

def create_random_features():
    cached_data = load_cached_data()
    if cached_data:
        return cached_data.data[0], cached_data.data[1]
    
    features = []
    labels = []
    min_value, max_value = float('inf'), -float('inf')

    for _ in tqdm(range(num_samples)):
        random_class = np.random.choice(class_distribution.index, p=class_probability_distribution)
        file_name = np.random.choice(dataset[dataset.label == random_class].index)
        sample_rate, audio_signal = wavfile.read(f"clean/{file_name}")
        label = dataset.at[file_name, 'label']
        random_index = np.random.randint(0, audio_signal.shape[0] - config.step)
        audio_sample = audio_signal[random_index:random_index + config.step]
        mfcc_features = mfcc(audio_sample, sample_rate, numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft)
        min_value = min(np.amin(mfcc_features), min_value)
        max_value = max(np.amax(mfcc_features), max_value)
        features.append(mfcc_features)
        labels.append(classes.index(label))

    config.min_value = min_value
    config.max_value = max_value

    features, labels = np.array(features), np.array(labels)
    features = (features - min_value) / (max_value - min_value)

    if config.mode == 'conv':
        features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    elif config.mode == 'time':
        features = features.reshape(features.shape[0], features.shape[1], features.shape[2])

    labels = to_categorical(labels, num_classes=len(classes))

    config.data = (features, labels)
    
    with open(config.pickle_path, 'wb') as handle:
        pickle.dump(config, handle, protocol=2)

    return features, labels

def build_conv_model():
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=input_shape),
        Conv2D(32, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        Conv2D(128, (3, 3), activation='relu', strides=(1, 1), padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(len(classes), activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

dataset = pd.read_csv('instruments.csv')
dataset.set_index('fname', inplace=True)

# 

for file_name in dataset.index:
    sample_rate, signal = wavfile.read(f"clean/{file_name}")
    dataset.at[file_name, 'length'] = signal.shape[0] / sample_rate

classes = list(np.unique(dataset.label))
class_distribution = dataset.groupby(['label'])['length'].mean()

num_samples = 2 * int(dataset['length'].sum() / 0.1)
class_probability_distribution = class_distribution / class_distribution.sum()

fig, ax = plt.subplots()
ax.set_title('Class Distribution', y=1.08)
ax.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()

config = Config(mode="conv")

if config.mode == 'conv':
    features, labels = create_random_features()
    flat_labels = np.argmax(labels, axis=1)
    input_shape = (features.shape[1], features.shape[2], 1)
    model = build_conv_model()

class_weights = compute_class_weight('balanced', classes=np.unique(flat_labels), y=flat_labels)

checkpoint = ModelCheckpoint(config.model_path, monitor='val_accuracy', verbose=1, mode='max', save_best_only=True, save_weights_only=False, period=1)

model.fit(
    features, labels,
    epochs=10,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    callbacks=[checkpoint]
)

model.save(config.model_path)
