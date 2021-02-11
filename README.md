
cwkeras: Decode Morse Code (CW) with Keras
===

Installation
---

You need Python 3.5-3.8 to run Tensorflow (last I checked)

~~~sh
python3.8 -m venv .
. bin/activate
pip install -r requirements.txt
python train_detect.py
python train_translate.py
python run_detect.py [.wav files...]
~~~


Training Data
---

We generate training samples of three types:
1. Random CW symbols + noise (50%)
2. Random 0s and 1s + noise (25%)
3. Just noise (25%)

Distinguishing between #1 and #3 is easy, between #1 and #2 is hard.

We assume 100 samples per second, which at 20 WPM gives a dit length of about 6 samples. We use a window of 500 samples for recognition, or 5 seconds.

If the morse signal is bigger than the window size, we crop it so that at least 50% of the signal remains. Otherwise we randomly place it in the window.

The pulse stream dit length and non-dit length varies in speed, and also may vary during the sample.




Detection Model
---

The detection model just answers the question "is there a Morse code signal at this frequency?"
We can run it in parallel on an entire 5-second window of spectrum.

The model uses multiple Conv1D layers with 64 x 7 filters.


Translation Model
--

For translation, we use Conv1D layers, the last layer having exactly as many filters as target characters (A-Z, 0-9, space), plus one for "no character", 39 values in all.

The translation model also uses multiple Conv1D layers, but 128 x 5 filters.
500 samples get downsampled to 62 bins, and symbols are decoded into the closest bin.
The final layer is a TimeDistributed Dense layer to classify each bin -> symbol, or 0 = no symbol.

It's uncommon that symbols would share the same bin, but if so, the later one is moved the adjacent bin.
We don't try to decode symbols that aren't completely contained within the window.


Older Notes
--

limited training set at first:
- [CQ] <callsign>
- <callsign> [73 | RRR]
- <callsign> <gridsquare>
- <callsign> [R][+-]<num>
- noise or random letters

<callsign>:
	\d?[A-Z]{1,2}\d{1,4}[A-Z]{1,4}
	(type 1/2 compound callsigns?)

input: normalized spectral data (do we need I/Q?)
what if signal spans 2 bins? (take most likely bin? use adjacent bins?)
50% overlap fft?

https://github.com/jj1bdx/wspr-cui

https://physics.princeton.edu//pulsar/K1JT/JT65.pdf


T = 1200/WPM = 17 msec ~= 20 msec
downsample to 10 msec (100 samples/sec)
- wspr is 375 frames/sec
random slice
need about 200 bits for input
handle up to 15 chars - 750?

output: [ A-Z0-9+-/]
0-15 chars

dot = 1/1
dash = 3/1
space = 3
word = 7

variability? maybe 0-30%? 

record actual hits to add to corpus, submit to server


