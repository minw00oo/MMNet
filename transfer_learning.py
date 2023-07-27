from tensorflow.keras import Model, callbacks, optimizers
from tensorflow.keras.models import load_model

a = 'after first_encoder_block'
b = 'befor last_decoder_block'

model = load_model('pre_trained model.h5')
model.summary()

for layer in model.layers[a:b]:
    layer.trainable = False
model.summary()

adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam,
              loss="mse",
              metrics=["mse"])
