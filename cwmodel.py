
import morse, keras
import numpy as np

channels = 1
samples_per_sec = 100
max_seconds = 5
max_samples = max_seconds * samples_per_sec
trans_seconds = 15
trans_samples = trans_seconds * samples_per_sec
latent_dim = 100
TOKENS = "$^0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
num_decoder_tokens = len(TOKENS)
target_token_index = dict([(char, i) for i, char in enumerate(TOKENS)])

# detection model
def make_model(input_shape = (max_samples,channels)):
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

# translation model
# https://keras.io/examples/nlp/lstm_seq2seq/
def make_trans_model(input_shape = (trans_samples,channels)):
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

    conv5 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.MaxPooling1D()(conv5)
    conv5 = keras.layers.Dropout(0.1)(conv5)
    conv5 = keras.layers.ReLU()(conv5)

    conv6 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv5)
    conv6 = keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.MaxPooling1D()(conv6)
    conv6 = keras.layers.ReLU()(conv6)

    decoder_lstm = keras.layers.LSTM(latent_dim, dropout=0.1, return_sequences=True)(conv6)
    encoder_lstm = keras.layers.LSTM(num_decoder_tokens, return_sequences=True)(decoder_lstm)
    return keras.models.Model(inputs=input_layer, outputs=encoder_lstm)

    #encoder = keras.layers.LSTM(latent_dim, return_state=True, dropout=0.2)
    #encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    #encoder_states = [state_h, state_c]
    # Set up the decoder, using `encoder_states` as initial state.
    #decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    #decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    #decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    #decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
    #decoder_outputs = decoder_dense(decoder_outputs)
    #return keras.models.Model(inputs=[input_layer, decoder_inputs], outputs=decoder_outputs)

class DataGenerator(keras.utils.Sequence):
    'Generates detection data for Keras'
    def __init__(self, batch_size=128):
        'Initialization'
        self.dim = (batch_size,max_samples,channels)
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
            y = len(msg) > 0 and msg[0] != '~'
            x_train.append(x)
            y_train.append(y)
        return np.array(x_train), np.array(y_train)

    def on_epoch_end(self):
        'Updates indexes after each epoch'


class TranslationGenerator(keras.utils.Sequence):
    'Generates detection data for Keras'
    def __init__(self, batch_size=64):
        'Initialization'
        self.dim = (batch_size,trans_samples,channels)
        self.batch_size = batch_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return 1000

    def __getitem__(self, index):
        'Generate one batch of data'
        x_train = []
        y_train = []
        for i in range(0, self.batch_size):
            msg, x, posns = morse.generate_translation_training_sample(trans_samples)
            x = np.reshape(x, (-1,1))
            y = np.zeros((23, num_decoder_tokens))
            msg += '$'
            for t, char in enumerate(msg):
                y[t, target_token_index[msg[t]]] = 1.0
            x_train.append(x)
            y_train.append(y)
        return np.array(x_train), np.array(y_train)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

