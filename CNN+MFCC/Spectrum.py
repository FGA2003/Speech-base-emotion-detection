import librosa
import librosa.display
import numpy as np
import pandas as pd
import glob
import os, sys
import matplotlib.pyplot as plt
import scipy.io.wavfile
sr,x = scipy.io.wavfile.read(r'D:\Third year project\Dataset\RAVDESS-sorted by emotions\angry\03-01-05-01-01-01-01.wav')

#parameter setting
nstep = int(sr * 0.01)
nwin  = int(sr * 0.03)
nfft = nwin
window = np.hamming(nwin)
nn = range(nwin, len(x), nstep)
X = np.zeros( (len(nn), nfft//2) )
for i,n in enumerate(nn):
    xseg = x[n-nwin:n]
    z = np.fft.fft(window * xseg, nfft)
    X[i,:] = np.log(np.abs(z[:nfft//2]))
plt.imshow(X.T, interpolation='nearest',
    origin='lower',
    aspect='auto')
plt.show()