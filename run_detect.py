
import sys
import keras
import morse
import numpy as np
import cwmodel
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

fns = sys.argv[1:]

detect_checkpoint_fn = "best_model.h5"
detect_model = cwmodel.make_model()
detect_model.load_weights(detect_checkpoint_fn)

trans_checkpoint_fn = "best_trans_model.h5"
trans_model = cwmodel.make_trans_model()
trans_model.load_weights(trans_checkpoint_fn)

for fn in fns:
    sample_rate, samples = wavfile.read(fn)
    #print(fn, sample_rate, len(samples))
    frequencies, times, spectrogram = signal.spectrogram(samples, fs=sample_rate, nperseg=256, noverlap=192)

    #print(spectrogram.shape)
    if spectrogram.shape[1] < cwmodel.trans_samples:
        xy = np.pad(spectrogram, ((0,0), (0,cwmodel.trans_samples-spectrogram.shape[1])))
    else:
        xy = spectrogram[:, 0:cwmodel.trans_samples]
    xy = (xy - np.min(xy)) / (np.max(xy) - np.min(xy))
    xy = np.reshape(xy, (xy.shape[0], xy.shape[1], 1))
    #print(xy.shape)
    p = detect_model.predict(xy[:, 0:cwmodel.max_samples])
    #print(p)
    print(fn, np.argwhere(p > 0.5))
    #print([frequencies[i[0]] for i in np.argwhere(p > 0.5)])

    # iterate over all detected rows
    for y,i in np.argwhere(p > 0.5):
        row = xy[y]
        row = np.reshape(row, (1, row.shape[0], 1))
        t = trans_model.predict(row)[0]
        nbins = t.shape[0]
        # pick the best choice for each bin
        msg = ['.'] * nbins
        for j in range(0, nbins):
            k = np.argmax(t[j])
            if k>0:
                msg[j] = cwmodel.TOKENS[k]
        print(''.join(msg))

    plt.subplot(211, label=fn)
    plt.pcolormesh(times, frequencies, spectrogram, shading='auto')
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.subplot(212, label=fn)
    plt.plot(xy[19])

plt.show()

