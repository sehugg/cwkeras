
# cwkeras: Decode Morse Code (CW) with Keras

## Installation

You need Python 3.5-3.8 to run Tensorflow (last I checked)

~~~sh
python3.8 -m venv .
. bin/activate
pip install -r requirements.txt
~~~

## Training

Train the detection model:
~~~sh
python train_detect.py
~~~
Train the translation model:
~~~sh
python train_translate.py
~~~

## Prediction

Detect and translate Morse code from .wav files:
~~~sh
python run_detect.py [.wav files...]
~~~

Or in real time from the default audio-in device:
~~~sh
python run_snd.py [.wav files...]
~~~

Whenever CW is detected, it'll output the bin number and predicted translation:
~~~
G22R49Z6GYQVM9ZOAM8N.wav 8000 70137
19 ...G..2...2.....4...99..Z..6..G..YY..Q..V..M..99..Z..O..A.M...
~~~
When translating in real-time, audio is captured in a 5-second window, which shifts every 2.5 seconds.


## Training Data

We generate training samples of three types:
1. Random CW symbols + noise (50%)
2. Random 0s and 1s + noise (25%)
3. Just noise (25%)

Distinguishing between #1 and #3 is easy, between #1 and #2 is hard.

We assume 100 samples per second, which at 20 WPM gives a dit length of about 6 samples. We use a window of 500 samples for recognition, or 5 seconds.

If the morse signal is bigger than the window size, we crop it so that at least 50% of the signal remains. Otherwise we randomly place it in the window.

The pulse stream dit length and non-dit length varies in speed, and also may vary during the sample.


## Detection Model

The detection model just answers the question "is there a Morse code signal at this frequency?"
We can run it in parallel on an entire 5-second window of spectrum.

The model uses multiple Conv1D layers with 64 x 7 filters.


## Translation Model

For translation, we use Conv1D layers, the last layer having exactly as many filters as target characters (A-Z, 0-9, space), plus one for "no character", 39 values in all.

The translation model also uses multiple Conv1D layers, but 96 x 7 filters.
500 samples get downsampled to 62 bins, and symbols are decoded into the closest bin.
The final layer is a TimeDistributed Dense layer to classify each bin -> symbol, or 0 = no symbol.

It's uncommon that symbols would share the same bin, but if so, the later one is moved the adjacent bin.
We don't try to decode symbols that aren't completely contained within the window.

Before translation, if we see CW data in two adjacent bins (frequencies) we will merge them.

