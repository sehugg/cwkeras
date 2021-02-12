
import morse, keras
import numpy as np

channels = 1
samples_per_sec = 100
max_seconds = 5
max_samples = max_seconds * samples_per_sec
trans_seconds = 5
trans_samples = trans_seconds * samples_per_sec
latent_dim = 100
TOKENS = "$^0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
num_decoder_tokens = len(TOKENS)
target_token_index = dict([(char, i) for i, char in enumerate(TOKENS)])
max_translation_length = 62
use_lstm = False

# detection model
def make_model(input_shape = (max_samples,channels)):
    input_layer = keras.layers.Input(input_shape)
    nf = 64
    ks = 7

    conv1 = keras.layers.Conv1D(filters=nf, kernel_size=ks, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.MaxPooling1D()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.Dropout(0.3)(conv1)

    conv2 = keras.layers.Conv1D(filters=nf, kernel_size=ks, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.MaxPooling1D()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.Dropout(0.2)(conv2)

    conv3 = keras.layers.Conv1D(filters=nf, kernel_size=ks, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.MaxPooling1D()(conv3)
    conv3 = keras.layers.ReLU()(conv3)
    conv3 = keras.layers.Dropout(0.1)(conv3)

    conv4 = keras.layers.Conv1D(filters=nf, kernel_size=ks, padding="same")(conv3)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.MaxPooling1D()(conv4)
    conv4 = keras.layers.ReLU()(conv4)

    conv5 = keras.layers.Conv1D(filters=nf, kernel_size=ks, padding="same")(conv4)

    gap = keras.layers.GlobalAveragePooling1D()(conv5)

    output_layer = keras.layers.Dense(1, activation="sigmoid")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

# translation model
# https://keras.io/examples/nlp/lstm_seq2seq/
def make_trans_model(input_shape = (trans_samples,channels)):
    input_layer = keras.layers.Input(input_shape)
    nf = 96
    ks = 7

    conv1 = keras.layers.Conv1D(filters=nf, kernel_size=ks, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.AveragePooling1D()(conv1)
    conv1 = keras.layers.ReLU()(conv1)
    conv1 = keras.layers.Dropout(0.2)(conv1)

    conv2 = keras.layers.Conv1D(filters=nf, kernel_size=ks, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.AveragePooling1D()(conv2)
    conv2 = keras.layers.ReLU()(conv2)
    conv2 = keras.layers.Dropout(0.1)(conv2)

    conv3 = keras.layers.Conv1D(filters=nf, kernel_size=ks, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.AveragePooling1D()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    if use_lstm:
        lstm1 = keras.layers.LSTM(nf, go_backwards=True, return_sequences=True)(conv3)
        lstm2 = keras.layers.LSTM(num_decoder_tokens, return_sequences=True)(lstm1)
        return keras.models.Model(inputs=input_layer, outputs=lstm2)
    else:
        conv7 = keras.layers.Conv1D(filters=nf, kernel_size=ks, activation="relu", padding="same")(conv3)
        #conv7 = keras.layers.BatchNormalization()(conv7)
        #conv8 = keras.layers.Conv1D(filters=num_decoder_tokens*2, kernel_size=3, activation="relu", padding="same")(conv7)
        conv8 = keras.layers.TimeDistributed(keras.layers.Dense(nf*2, activation="relu"))(conv7)
        dense = keras.layers.TimeDistributed(keras.layers.Dense(num_decoder_tokens, activation="softmax"))(conv8)
        return keras.models.Model(inputs=input_layer, outputs=dense)

    #encoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, dropout=0.1)(conv5)
    #decoder_lstm = keras.layers.LSTM(num_decoder_tokens, dropout=0.1, return_sequences=True)(encoder_lstm)
    #return keras.models.Model(inputs=input_layer, outputs=decoder_lstm)

    #encoder_inputs = conv5
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
            msg, x, posns = morse.generate_detection_training_sample(max_samples)
            x = np.reshape(x, (-1,1))
            y = len(msg) > 0 and msg[0] != '~'
            x_train.append(x)
            y_train.append(y)
        return np.array(x_train), np.array(y_train)

    def on_epoch_end(self):
        'Updates indexes after each epoch'


class TranslationGenerator(keras.utils.Sequence):
    'Generates detection data for Keras'
    def __init__(self, batch_size=128):
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
            assert(len(posns) == len(msg)+1)
            x = np.reshape(x, (-1,1))
            y = np.zeros((max_translation_length, num_decoder_tokens))
            str = ['.'] * max_translation_length
            # iterate over all bins
            for i, char in enumerate(msg):
                if use_lstm:
                    # lstm, bin goes at end of symbol
                    pos = posns[i+1] / trans_samples * max_translation_length
                    ofs = int(round(pos))+1
                else:
                    # put bin smack dab in middle of the feature
                    pos = ((posns[i] + posns[i+1]) / trans_samples / 2) * max_translation_length
                    ofs = int(round(pos))
                # is this symbol in the window?
                if ofs > 0 and ofs < max_translation_length-1 and posns[i] > 0 and posns[i+1] < trans_samples:
                    tti = target_token_index[msg[i]]
                    y[ofs-1, tti] = 1/4
                    y[ofs+0, tti] = 1/1
                    y[ofs+1, tti] = 1/4
                    str[ofs] = char
            # set "no symbol" probability for bins
            for ofs in range(0,max_translation_length):
                y[ofs, 0] = max(0.0, 1.0 - np.sum(y[ofs]))
            x_train.append(x)
            y_train.append(y)
            #print(''.join(str))
        return np.array(x_train), np.array(y_train)

    def on_epoch_end(self):
        'Updates indexes after each epoch'

