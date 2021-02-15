import rstr, math, random, re
import morse_talk as mtalk
import numpy as np

MIN_SIGNAL_LENGTH = 2
MAX_SIGNAL_LENGTH = 10
MIN_SNR = 2
MAX_SNR = 20

def generate_morse_samples(bits, symlen, variance, drift):
    i = random.random()
    s = []
    p = []
    sym2 = 0
    zerocnt = 0
    p.append(i)
    for k in range(0,len(bits)):
        b = bits[k]
        sym = int(b)
        symlen[sym] = min(MAX_SIGNAL_LENGTH, max(MIN_SIGNAL_LENGTH, symlen[sym] + random.gauss(0, drift)))
        if sym == 0:
            zerocnt += 1
            if zerocnt == 4: # is space? (7 bits)
                p.append(i-symlen[0*4]) # SWAG at start of space...
        else:
            if zerocnt >= 3: # inter-char? (3 bits)
                p.append(i)
            zerocnt = 0
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
        i = i2
        sym2 = sym
    s.append(i-int(i))
    p.append(i)
    return np.array(s), np.array(p)

def generate_signal(msg):
    sl = MIN_SIGNAL_LENGTH + random.random() * (MAX_SIGNAL_LENGTH - MIN_SIGNAL_LENGTH)
    sv = random.random()
    symlen = [sl+random.gauss(0,sv), sl+random.gauss(0,sv)]
    variance = random.gauss(0, 0.2*sv)
    drift = random.gauss(0, 0.1*sv)
    # fake msgs start with "~"
    isfake = re.match(r'^[~][01]+$', msg)
    if isfake:
        bits = msg[1:]
    else:
        bits = mtalk.encode(msg, encoding_type='binary')
    samp, posns = generate_morse_samples(bits, symlen, variance, drift)
    if not isfake and len(msg) > 0:
        assert(len(posns) == len(msg)+1)
    return samp, posns

def generate_signoise(msg, MAXSAMP):
    sig, posns = generate_signal(msg)
    siglen = len(sig)
    bpc = siglen / len(posns)  # bins per char
    if siglen < MAXSAMP-1:
        # randomly center signal in window
        lpad = random.randrange(0, MAXSAMP-siglen-1)
        sig = np.pad(sig, (lpad, MAXSAMP-lpad-siglen), 'constant')
        posns += lpad
    else:
        # signal bigger than window, at least 50% must appear
        sig = np.pad(sig, (MAXSAMP//2, MAXSAMP//2), 'constant')
        ofs = random.randrange(0, siglen)
        sig = sig[ofs:ofs+MAXSAMP]
        posns += MAXSAMP//2 - ofs
    assert(sig.shape == (MAXSAMP,))
    # generate random noise
    noise = np.random.normal(0, 1, (MAXSAMP,))
    # TODO: snr proportional to bitrate
    # mix signal and noises (multiplicative and additive)
    snr2 = MIN_SNR + random.random() * (MAX_SNR-MIN_SNR)
    multnoise = np.random.normal(1, 1.0/snr2, (MAXSAMP,))
    snr = MIN_SNR + random.random() * (MAX_SNR-MIN_SNR)
    signoise = sig * multnoise * snr + noise
    # simulate 0% - 50% windowing
    k = random.random() * 0.5
    signoise = np.convolve(signoise, [k/2, 1-k, k/2], mode='same')
    return signoise, posns

def normalize(y):
    return (y-min(y))/(max(y)-min(y))
    #normalized = (y - y.mean(axis=0)) / y.std(axis=0)

def generate_detection_training_sample(MAXSAMP, noempty=False):
    r = random.random()
    if r > 0.5 or noempty:
        # cq? callsign? grid square?
        if r > 0.9:
            msg = rstr.xeger(r'(CQ )?\d?[A-Z]{1,2}\d{1,4}[A-Z]{1,4}( [A-R][A-R][0-9][0-9])?')
        else:
            msg = rstr.xeger(r'[A-Z0-9]{3,12}( [A-Z0-9]{1,12})?( [A-Z0-9]{1,12})?')
    elif r > 0.25:
        # fake msg
        msg = '~' + rstr.xeger(r'[0-1]{30,300}')
    else:
        # no msg, just noise
        msg = ''
    # simulate undetectable too-long signals
    if r > 0.49 and r < 0.50:
        global MIN_SIGNAL_LENGTH, MAX_SIGNAL_LENGTH
        tmp = (MIN_SIGNAL_LENGTH, MAX_SIGNAL_LENGTH)
        MIN_SIGNAL_LENGTH = MAX_SIGNAL_LENGTH + 1
        MAX_SIGNAL_LENGTH = 30
        y, posns = generate_signoise(msg, MAXSAMP)
        MIN_SIGNAL_LENGTH = tmp[0]
        MAX_SIGNAL_LENGTH = tmp[1]
    else:
        y, posns = generate_signoise(msg, MAXSAMP)
    normalized = normalize(y)
    return (msg, normalized, posns)

def generate_translation_training_sample(MAXSAMP):
    r = random.random()
    if r > 0.9:
        msg = rstr.xeger(r'(CQ )?\d?[A-Z]{1,2}\d{1,4}[A-Z]{1,4}( [A-R][A-R][0-9][0-9])?')
    else:
        msg = rstr.xeger(r'[A-Z0-9]{1,12}( [A-Z0-9]{1,12})?( [A-Z0-9]{1,12})?')
    y, posns = generate_signoise(msg, MAXSAMP)
    normalized = normalize(y)
    return (msg, normalized, posns)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    if 1:
        nex = 25
        im = []
        for i in range(0,nex):
            msg,y,posns = generate_detection_training_sample(500)
            #msg,y,posns = generate_translation_training_sample(500)
            im.append(y)
            im.append(np.zeros(len(y)))
        im = np.array(im)
        plt.imshow(im, aspect='auto')
        plt.show()
    if 0:
        #y = generate_signoise('CQ', 300)
        fig, axs = plt.subplots(4, 4)
        for i in range(0,16):
            msg,y,posns = generate_detection_training_sample(500)
            print(msg, posns)
            ax = axs[i%4,i//4]
            ax.plot(y, linewidth=0.5)
            if len(msg) < 30:
                ax.set_title(msg)
            else:
                ax.set_title(str(len(msg))+' bits')
        plt.tight_layout()
        plt.show()
