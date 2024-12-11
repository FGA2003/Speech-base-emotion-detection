import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

## load audio data
data, sampling_rate = librosa.load(r'D:\Third year project\Dataset\RAVDESS-sorted by emotions\angry\03-01-05-01-01-01-01.wav')

#  Plot waveform fiagram
plt.figure(figsize=(15, 5))
librosa.display.waveshow(data, sr=sampling_rate)
plt.show()  

##D:\Third year project\Dataset\RAVDESS-sorted by emotions\angry\03-01-05-01-01-01-01.wav