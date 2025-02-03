import pickle
import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score

def build_predictions(audio_directory):
    true_labels = []
    predicted_labels = []
    file_probabilities = {}

    print('Extracting features from audio files...')
    for audio_file in tqdm(os.listdir(audio_directory)):
        sample_rate, audio_waveform = wavfile.read(os.path.join(audio_directory, audio_file))
        actual_label = file_to_label_mapping[audio_file]
        actual_label_index = class_labels.index(actual_label)
        file_predictions = []

        for start_idx in range(0, audio_waveform.shape[0] - config.step, config.step):
            audio_sample = audio_waveform[start_idx:start_idx + config.step]
            mfcc_features = mfcc(
                audio_sample, 
                sample_rate, 
                numcep=config.nfeat, 
                nfilt=config.nfilt, 
                nfft=config.nfft
            )

            if config.min is None or config.max is None:
                config.min = np.min(mfcc_features)
                config.max = np.max(mfcc_features)
              
            normalized_features = (mfcc_features - config.min) / (config.max - config.min)

            if config.mode == 'conv':
                normalized_features = normalized_features.reshape(
                    1, normalized_features.shape[0], normalized_features.shape[1], 1
                )

            prediction = model.predict(normalized_features)
            file_predictions.append(prediction)
            predicted_labels.append(np.argmax(prediction))
            true_labels.append(actual_label_index)

        file_probabilities[audio_file] = np.mean(file_predictions, axis=0).flatten()

    return true_labels, predicted_labels, file_probabilities

# Load dataset and metadata
data_frame = pd.read_csv('instruments.csv')
class_labels = list(np.unique(data_frame.label))
file_to_label_mapping = dict(zip(data_frame.fname, data_frame.label))
pickle_path = os.path.join('pickles', 'conv.p')

# Load configuration
with open(pickle_path, 'rb') as handle:
    config = pickle.load(handle)

# Load pre-trained model
model = load_model(config.model_path)

# Build predictions
true_labels, predicted_labels, file_probabilities = build_predictions('clean')
accuracy = accuracy_score(y_true=true_labels, y_pred=predicted_labels)

print("Accuracy score: {:.2f}".format(accuracy))

# Append predictions to the dataframe
probability_list = []
for _, row in data_frame.iterrows():
    probabilities = file_probabilities[row.fname]
    probability_list.append(probabilities)
    for label, probability in zip(class_labels, probabilities):
        data_frame.at[_, label] = probability

# Add predicted class to the dataframe
predicted_class_labels = [class_labels[np.argmax(probabilities)] for probabilities in probability_list]
data_frame['predicted_label'] = predicted_class_labels

# Save predictions to a CSV file
data_frame.to_csv('predictions.csv', index=False)
