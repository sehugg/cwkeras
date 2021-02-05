import rstr, math, random
import morse_talk as mtalk
import numpy as np

DEFAULT_SIGNAL_LENGTH = 4
MIN_SIGNAL_LENGTH = 2.5
MIN_SNR = 1
MAX_SNR = 10

def generate_morse_samples(msg, symlen, variance, drift):
    i = random.random()
    s = []
    sym2 = 0
    bits = mtalk.encode(msg, encoding_type='binary')
    for b in bits:
        sym = int(b)
        i2 = i + symlen[sym]
        l = int(i2 - int(i) + random.gauss(0, variance))
        if sym2 != sym:
            if sym: # 0 -> 1
                s.append(1-i+int(i))
            else: # 1 -> 0
                s.append(i-int(i))
            s.extend([sym] * (l-1))
        else:
            s.extend([sym] * l)
        symlen[sym] = max(MIN_SIGNAL_LENGTH, symlen[sym] + random.gauss(0, drift))
        i = i2
        sym2 = sym
    s.append(i-int(i))
    return np.array(s)
    
def generate_signal(msg):
    sl = DEFAULT_SIGNAL_LENGTH
    sv = random.gauss(0,1)
    symlen = [sl+random.gauss(0,sv), sl+random.gauss(0,sv)]
    variance = random.gauss(0,0.2)
    drift = random.gauss(0,0.1)
    samp = generate_morse_samples(msg, symlen, variance, drift)
    #blur = random.gauss(0,0.5)
    #samp = np.convolve(samp, [blur,1,blur])
    return samp

def generate_signoise(msg, MAXSAMP):
    sig = generate_signal(msg)
    siglen = len(sig)
    if siglen < MAXSAMP:
        # randomly center signal in window
        lpad = random.randrange(0, MAXSAMP-siglen)
        sig = np.pad(sig, (lpad, MAXSAMP-lpad-siglen), 'constant')
    else:
        # signal bigger than window, at least 50% must appear
        sig = np.pad(sig, (MAXSAMP//2, MAXSAMP//2), 'constant')
        ofs = random.randrange(0, siglen)
        sig = sig[ofs:ofs+MAXSAMP]
    assert(sig.shape == (MAXSAMP,))
    noise = np.random.normal(0, 1, (MAXSAMP,))
    snr = MIN_SNR + random.random() * (MAX_SNR-MIN_SNR)
    return sig * snr + noise

def generate_detection_training_sample(MAXSAMP, noempty=False):
    msg = ''
    r = random.random()
    if r > 0.5 or noempty:
        msg = rstr.xeger(r'\d?[A-Z]{1,2}\d{1,4}[A-Z]{1,4}') # [A-R][A-R][0-9][0-9]
        #msg = rstr.xeger(r'[A-Z0-9 ]{2,15}')
    y = generate_signoise(msg, MAXSAMP)
    normalized = (y-min(y))/(max(y)-min(y))
    return (msg, normalized)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    #y = generate_signoise('CQ', 300)
    fig, axs = plt.subplots(4, 4)
    for i in range(0,16):
        msg,y = generate_detection_training_sample(300, True)
        print(msg)
        axs[i%4,i//4].plot(y)
    plt.show()
