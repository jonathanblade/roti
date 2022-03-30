from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, Conv2DTranspose, ConvLSTM2D, BatchNormalization, LeakyReLU

def get_model_config():
    return {
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
        ]
    }

def add_layer(model, layer, use_normalization=True, use_activation=True):
    layer_name = layer[0]
    filters = layer[1]
    kernel_size = layer[2]
    strides = layer[3]
    padding = layer[4]

    if (layer_name == "conv"):
        model.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding))
    if (layer_name == "convLSTM"):
        model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding))
    if (layer_name == "deconv"):
        model.add(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding))

    if use_normalization:
        model.add(BatchNormalization())

    if use_activation:
        model.add(LeakyReLU())

    return model

def build_model(input_shape):
    config = get_model_config()

    encoder = Sequential()
    encoder.add(InputLayer(input_shape=input_shape))
    for layer in config["encoder"]:
        encoder = add_layer(encoder, layer)

    decoder = Sequential()
    decoder.add(InputLayer(input_shape=encoder.layers[-1].output_shape))
    for layer in config["decoder"]:
        decoder = add_layer(decoder, layer)

    return (encoder, decoder)
