
import keras
import morse
import numpy as np
import cwmodel

checkpoint_fn = "best_model.h5"

model = cwmodel.make_model()
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        checkpoint_fn, save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
)
try:
    model.load_weights(checkpoint_fn)
except:
    print("could not load weights")

training_generator = cwmodel.DataGenerator()
validation_generator = cwmodel.DataGenerator()

epochs = 500

history = model.fit(
    x=training_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)


