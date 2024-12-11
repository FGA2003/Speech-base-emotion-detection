import sounddevice as sd
from scipy.io.wavfile import writefs = 44100  # sample rate
seconds = 4  # duration
sd.wait()  # until finish
write('output.wav', fs, myrecording)  # save 