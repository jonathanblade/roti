from tensorflow import GradientTape
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    ConvLSTM2D,
    Input,
    LeakyReLU,
    TimeDistributed,
    ZeroPadding2D,
)

CONFIG = {
    "encoder": [
        ["conv", 16, (3, 3), (1, 1), (1, 1)],
        ["convLSTM", 32, (5, 5), (1, 1), (2, 2)],
        ["conv", 32, (3, 3), (2, 2), (1, 1)],
        ["convLSTM", 64, (5, 5), (1, 1), (2, 2)],
        ["conv", 64, (3, 3), (2, 2), (1, 1)],
        ["convLSTM", 64, (5, 5), (1, 1), (2, 2)],
    ],
    "decoder": [
        ["convLSTM", 64, (5, 5), (1, 1), (2, 2)],
        ["deconv", 64, (3, 3), (2, 2), (1, 1)],
        ["convLSTM", 64, (5, 5), (1, 1), (2, 2)],
        ["deconv", 64, (4, 3), (2, 2), (1, 1)],
        ["convLSTM", 32, (5, 5), (1, 1), (2, 2)],
        ["conv", 16, (4, 3), (1, 1), (1, 1)],
        ["conv", 1, (1, 1), (1, 1), (0, 0)],
    ],
}

def add_layer(model, layer, use_normalization=False, use_activation=False):
    layer_name = layer[0]
    filters = layer[1]
    kernel_size = layer[2]
    strides = layer[3]
    padding = layer[4]

    if (layer_name == "conv"):
        # model.add(TimeDistributed(ZeroPadding2D(padding=padding)))
        model.add(TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")))
    if (layer_name == "convLSTM"):
        # model.add(TimeDistributed(ZeroPadding2D(padding=padding)))
        model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", return_sequences=True))
    if (layer_name == "deconv"):
        # model.add(TimeDistributed(ZeroPadding2D(padding=padding)))
        model.add(TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")))

    if use_normalization:
        model.add(BatchNormalization())

    if use_activation:
        model.add(LeakyReLU())

    return model

def build_model(input_shape=(3, 20, 180, 1)):
    model = Sequential(name="LeiLiuAE")
    model.add(Input(shape=input_shape))
    for layer in CONFIG.get("encoder"):
        model = add_layer(model, layer)
    for layer in CONFIG.get("decoder"):
        model = add_layer(model, layer)
    model.compile(optimizer=Adam(1e-3))
    return model

def update_weights(model, x, y):
    with GradientTape() as tape:
        pred_y = model(x)[0,0:1,:,:,0]
        # Compute loss
        loss = mean_absolute_error(y, pred_y)
    # Compute gradients
    gradients = tape.gradient(loss, model.trainable_variables)
    # Update weights
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def train(model, train_generator, epochs):
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}")
        for train_x, train_y in train_generator:
            update_weights(model, train_x, train_y)
    return model
