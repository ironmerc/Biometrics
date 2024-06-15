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

def record_audio():
    num_recordings = int(input("How many times would you like to record? "))

    audio = pyaudio.PyAudio()

    for i in range(num_recordings):
        filename = f"recording_{i+1}.wav"
        filepath = os.path.join("C:\\Users\\sinha\\Desktop\\VB\\data\\rec", filename)
        print(f"Recording {i+1} of {num_recordings}...")
        
        # Start recording
        stream = audio.open(format=pyaudio.paInt16, channels=2,
                            rate=44100, input=True,
                            frames_per_buffer=1024)
        print("Recording...")
        frames = []

        for i in range(0, int(44100 / 1024 * 3)):
            data = stream.read(1024)
            frames.append(data)

        print("Finished recording")

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

        time.sleep(1)  # Wait 1 second before the next recording

    audio.terminate()

#if __name__ == "__main__":
#    record_audio()


# Define the directory containing the WAV files
wav_dir = 'C:\\Users\\sinha\\Desktop\\VB\\data\\rec'

def create_mfcc_csv():
    # Define the output CSV file
    csv_file = 'C:\\Users\\sinha\\Desktop\\VB\\data\\mfccs.csv'

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



# Call the function
create_mfcc_csv()