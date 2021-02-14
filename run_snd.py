
import sys
import numpy as np
import cwmodel
import sounddevice as sd

sample_rate = 4000
sd.default.samplerate = sample_rate #//2
sd.default.channels = 1

cw = cwmodel.CWDetectorTranslator(sample_rate)
print("Listening...")

while True:
    samples = sd.rec(cw.nsamples, blocking=True)
    samples = np.reshape(samples, (samples.shape[0],))
    cw.add_samples(samples)
    cw.detect()
    reslist = cw.translate()
    if len(reslist):
        for r in reslist:
            print(r[0], r[1])
        print("---")

