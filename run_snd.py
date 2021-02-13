
import sys
import numpy as np
import cwmodel
import sounddevice as sd

sample_rate = 8000
sd.default.samplerate = sample_rate #//2
sd.default.channels = 1

cw = cwmodel.CWDetectorTranslator(sample_rate)

while True:
    samples = sd.rec(cw.nsamples, blocking=True)
    samples = np.reshape(samples, (cw.nsamples,))
    cw.add_samples(samples)
    cw.detect()
    reslist = cw.translate()
    for r in reslist:
        print(r[0], r[1])
