
import sys
import keras
import morse
import numpy as np
import cwmodel
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

fns = sys.argv[1:]

cw = None

for fn in fns:
    sample_rate, samples = wavfile.read(fn)
    print(fn, sample_rate, len(samples))
    if cw is None:
        cw = cwmodel.CWDetectorTranslator(sample_rate, overlap=3/4)
    cw.clear()
    for i in range(0, len(samples), cw.nsamples):
        cw.add_samples(samples[i:i+cw.nsamples])
        cw.detect()
        reslist = cw.translate()
        for r in reslist:
            print(r[0], r[1])
        print('---')

    plt.subplot(211, label=fn)
    plt.pcolormesh(cw.spec, shading='auto')
    plt.imshow(cw.spec)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.subplot(212, label=fn)
    plt.plot(cw.xy[19])

plt.show()

