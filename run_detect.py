
import sys
import keras
import morse
import numpy as np
import cwmodel
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

fns = sys.argv[1:]

for fn in fns:
    sample_rate, samples = wavfile.read(fn)
    #print(fn, sample_rate, len(samples))
    frequencies, times, spectrogram = signal.spectrogram(samples, fs=sample_rate, nperseg=256, noverlap=192)

    checkpoint_fn = "best_model.h5"
    model = cwmodel.make_model()
    model.load_weights(checkpoint_fn)

    #print(spectrogram.shape)
    if spectrogram.shape[1] < cwmodel.max_samples:
        xy = np.pad(spectrogram, ((0,0), (0,cwmodel.max_samples-spectrogram.shape[1])))
    else:
        xy = spectrogram[:, 0:500]
    xy = (xy - np.min(xy)) / (np.max(xy) - np.min(xy))
    xy = np.reshape(xy, (xy.shape[0], xy.shape[1], 1))
    #print(xy.shape)
    p = model.predict(xy)
    #print(p)
    print(fn, np.argwhere(p > 0.5))
    #print([frequencies[i[0]] for i in np.argwhere(p > 0.5)])
    #print("---")

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')

plt.show()

