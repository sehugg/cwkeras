
cwkeras: Decode Morse Code (CW) with Keras
===

Installation
---
TODO
~~~sh
python3.8 -m venv .
. bin/activate
pip install -r requirements.txt
python train_detect.py
~~~

Training Data
---

We generate training samples of three types:
1. Random callsign in CW + noise (3 chars min)
2. Random 0s and 1s + noise
3. Just noise

Distinguishing between #1 and #3 is easy, between #1 and #2 is hard.

We assume 100 samples per second, which at 20 WPM gives a dit length of about 6 samples. We use a window of 500 samples for recognition, or 5 seconds.

If the morse signal is bigger than the window size, we crop it so that at least 50% of the signal remains. Otherwise we randomly place it in the window.

The pulse stream dit length and non-dit length varies in speed, and also may vary during the sample.


Model
---

For recognition (is there or is there not a Morse code signal?) we use multiple Conv1D layers.

For translation to text we'll use a LSTM or something.

---

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


