from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, Concatenate, Dropout
from tensorflow.keras import Model, optimizers, callbacks
from tensorflow.keras.models import load_model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Dropout(0.2)(x)
    
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x
#%%
inputs = Input((64, 64, 3))

s1, p1 = encoder_block(inputs, 16)
s2, p2 = encoder_block(p1, 32)
s3, p3 = encoder_block(p2, 64)
s4, p4 = encoder_block(p3, 128)

b1 = conv_block(p4, 256)

d1 = decoder_block(b1, s4, 128)
d2 = decoder_block(d1, s3, 64)
d3 = decoder_block(d2, s2, 32)
d4 = decoder_block(d3, s1, 16)

outputs = Conv2D(1, 1, padding="same", activation="linear")(d4)

model = Model(inputs=[inputs], outputs=[outputs])

adam = optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer=adam,
              loss="mse",
              metrics=["mse"])
model.summary()
#%%
import numpy as np
import seaborn as sns
import random
import matplotlib.pyplot as plt

data2 = np.load('dataset2.npy')

target2 = np.load('target2.npy')
#%%
train_data = data2[:5000]
test_data = data2[5000:]

target2 = target2.reshape(6000, 64, 64, 1)
train_fem = target2[:5000]
test_fem = target2[5000:]
#%%
callbacks = callbacks.EarlyStopping(patience=30, monitor='val_loss')

results = model.fit(train_data, train_fem, validation_split=0.7, batch_size=100, epochs=1000, callbacks=callbacks)
#%%
pred_fem = model.predict(test_data)

test_fem = test_fem.reshape(3000, 64, 64)
pred_fem = pred_fem.reshape(3000, 64, 64)