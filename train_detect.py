
import keras
import morse
import numpy as np
import cwmodel

checkpoint_fn = "weights_detect.h5"

try:
    from google.colab import drive
    drive.mount('/content/drive')
    checkpoint_fn = '/content/drive/MyDrive/Colab Notebooks/' + checkpoint_fn
except:
    print("Couldn't mount Google Colab Drive")

model = cwmodel.make_model()
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        checkpoint_fn, save_best_only=True, monitor="binary_accuracy"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="binary_accuracy", factor=0.5, patience=10, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="binary_accuracy", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
try:
    model.load_weights(checkpoint_fn)
except:
    print("could not load weights", checkpoint_fn)

training_generator = cwmodel.DataGenerator()
validation_generator = cwmodel.DataGenerator()

epochs = 500

history = model.fit(
    x=training_generator,
    #validation_data=validation_generator,
    #validation_steps=100,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)


