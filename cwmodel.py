
import morse, keras
import numpy as np

channels = 1
samples_per_sec = 100
max_seconds = 5
max_samples = max_seconds * samples_per_sec

latent_dim = 20
num_decoder_tokens = 26+10+1
TOKENS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size=128):
        'Initialization'
        self.dim = (128,max_samples,channels)
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1000

    def __getitem__(self, index):
        'Generate one batch of data'
        x_train = []
        y_train = []
        for i in range(0,self.batch_size):
            msg, x = morse.generate_detection_training_sample(max_samples)
            x = np.reshape(x, (-1,1))
            x_train.append(x)
            y_train.append(len(msg) > 0 and msg[0] != '~')
        return np.array(x_train), np.array(y_train)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

def make_model_lstm(input_shape = (max_samples,channels)):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=32, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    lstm = keras.layers.LSTM(100, dropout=0.2)(conv2)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(lstm)
    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def make_model_conv(input_shape = (max_samples,channels)):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.MaxPooling1D()(conv1)
    conv1 = keras.layers.Dropout(0.2)(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.MaxPooling1D()(conv2)
    conv2 = keras.layers.Dropout(0.2)(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.MaxPooling1D()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    conv4 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.MaxPooling1D()(conv4)
    conv4 = keras.layers.ReLU()(conv4)

    gap = keras.layers.GlobalAveragePooling1D()(conv4)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

make_model = make_model_conv
