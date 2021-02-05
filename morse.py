import rstr, math, random
import morse_talk as mtalk
import numpy as np

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
        symlen[sym] = max(2, symlen[sym] + random.gauss(0, drift))
        i = i2
        sym2 = sym
    s.append(i-int(i))
    return np.array(s)
    
def generate_signal(msg):
    sl = 4
    sv = random.gauss(0,1)
    symlen = [sl+random.gauss(0,sv), sl+random.gauss(0,sv)]
    variance = random.gauss(0,0.2)
    drift = random.gauss(0,0.1)
    blur = random.gauss(0,0.5)
    samp = generate_morse_samples(msg, symlen, variance, drift)
    #samp = np.convolve(samp, [blur,1,blur])
    return samp

def generate_noise(noise_shape):
    noise = np.random.normal(0, random.random() * 0.5, noise_shape)
    return noise

def generate_signoise(msg, MAXSAMP):
    sig = generate_signal(msg)
    siglen = len(sig)
    if siglen < MAXSAMP:
        lpad = random.randrange(0, MAXSAMP-siglen)
        sig = np.pad(sig, (lpad, MAXSAMP-lpad-siglen), 'constant')
    else:
        ofs = random.randrange(0, siglen-MAXSAMP)
        sig = sig[ofs:ofs+MAXSAMP]
    assert(sig.shape == (MAXSAMP,))
    noise = generate_noise((MAXSAMP,))
    return sig + noise

def generate_detection_training_sample(MAXSAMP):
    msg = ''
    r = random.random()
    if r > 0.5:
        msg = rstr.xeger(r'\d?[A-Z]{1,2}\d{1,4}[A-Z]{1,4}') # [A-R][A-R][0-9][0-9]
        #msg = rstr.xeger(r'[A-Z0-9 ]{2,15}')
    y = generate_signoise(msg, MAXSAMP)
    return (msg,y)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    y = generate_signoise('CQ', 300)
    #msg,y = generate_detection_training_sample(300)
    #print(msg)
    plt.plot(y)
    plt.show()
