
import keras
import morse
import numpy as np
import cwmodel

checkpoint_fn = "best_trans_model.h5"

model = cwmodel.make_trans_model()
model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint(
        checkpoint_fn, save_best_only=True, monitor="loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="loss", patience=50, verbose=1),
]
#opt = keras.optimizers.SGD(lr=0.01)
model.compile(
    optimizer="adam",
    loss=keras.losses.kullback_leibler_divergence, #"categorical_crossentropy",
    metrics=["accuracy"],
)
try:
    model.load_weights(checkpoint_fn)
except:
    print("could not load weights")

training_generator = cwmodel.TranslationGenerator()
validation_generator = cwmodel.TranslationGenerator()

epochs = 500

history = model.fit(
    x=training_generator,
    #validation_data=validation_generator,
    #validation_steps=100,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)


