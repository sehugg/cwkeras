
import keras
import morse
import numpy as np

channels = 1
samples_per_sec = 50
max_seconds = 20
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
        return 64

    def __getitem__(self, index):
        'Generate one batch of data'
        x_train = []
        y_train = []
        for i in range(0,self.batch_size):
            msg, x = morse.generate_detection_training_sample(max_samples)
            x = np.reshape(x, (-1,1))
            x_train.append(x)
            y_train.append(len(msg) > 0)
        return np.array(x_train), np.array(y_train)

    def on_epoch_end(self):
        'Updates indexes after each epoch'


def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(2, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)


model = make_model(input_shape=(max_samples,channels))
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.h5", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

training_generator = DataGenerator()
validation_generator = DataGenerator()

epochs = 500

history = model.fit(
    x=training_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)


