import pyaudio
import wave
import time
import os
import librosa as lb
import numpy as np
from scipy.io.wavfile import write
import pandas as pd
import os
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def recog():
    audio = pyaudio.PyAudio()

    filename = "recording.wav"
    filepath = os.path.join("C:\\Users\\sinha\\Desktop\\VB\\data\\last_try", filename)

    print("Recording started...")

    # Start recording
    stream = audio.open(format=pyaudio.paInt16, channels=2,
                        rate=44100, input=True,
                        frames_per_buffer=1024)
    frames = []

    for i in range(0, int(44100 / 1024 * 3)):
        data = stream.read(1024)
        frames.append(data)

    print("Recording finished.")

    # Stop recording
    stream.stop_stream()
    stream.close()

    # Save the recording to a WAV file
    waveFile = wave.open(filepath, 'wb')
    waveFile.setnchannels(2)
    waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    waveFile.setframerate(44100)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

    audio.terminate()

    print("Cretating csv from using mfcc")
    wav_dir = 'C:\\Users\\sinha\\Desktop\\VB\\data\\last_try'

    # Define the output CSV file
    csv_file = 'C:\\Users\\sinha\\Desktop\\VB\\data\\runtime.csv'

    # Initialize an empty DataFrame to store the MFCCs
    df = pd.DataFrame(columns=[str(i) for i in range(40)] + ["speaker"])

    # Initialize a counter for the row indices
    index = 0

    # Loop through the WAV files
    for filename in os.listdir(wav_dir):
        if filename.endswith(".wav"):
            # Load the WAV file
            audio, sample_rate = lb.load(os.path.join(wav_dir, filename))

            # Extract 40 MFCCs
            mfccs = lb.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)

            # Scale the MFCCs
            mfccs_scaled = np.mean(mfccs.T, axis=0)

            # Convert the MFCCs to a list
            lst = list(mfccs_scaled) + [1]

            # Append the MFCCs to the DataFrame
            df.loc[index] = lst

            # Increment the index
            index += 1

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=True)

    print("Created")

    print("Authenticating")
    
    # Load the CSV file for the training data
    train_df = pd.read_csv('C:\\Users\\sinha\\Desktop\\VB\\data\\mfccs.csv')

    # Normalize the feature values
    X_train = train_df.drop('speaker', axis=1)
    y_train = train_df['speaker']

    X_train_norm = (X_train - X_train.mean()) / X_train.std()

    # Train the MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(30, 20, 1), random_state=1, max_iter=300)
    clf.fit(X_train_norm, y_train)

    # Load the CSV file for the test data
    test_df = pd.read_csv('C:\\Users\\sinha\\Desktop\\VB\\data\\runtime.csv')

    # Normalize the feature values for the test data
    X_test = test_df.drop('speaker', axis=1)
    X_test_norm = (X_test - X_train.mean()) / X_train.std()

    # Make predictions on the test data
    y_pred = clf.predict(X_test_norm)

    # Evaluate the model on the test data
    print("Accuracy:", accuracy_score(test_df['speaker'], y_pred))

    # Print authentication results
    if np.all(y_pred == test_df['speaker']):
        print("Authenticated")
    else:
        print("Denied")

recog();