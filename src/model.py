from keras import Model
from keras.models import Sequential
from keras.layers import Input, Conv2D, Conv2DTranspose, ConvLSTM2D, BatchNormalization, LeakyReLU, ZeroPadding2D, TimeDistributed

class LeiLiuAE(Model):
    def __init__(self, input_shape):
        super(LeiLiuAE, self).__init__(name="Lei Liu AE")
        self.encoder = self._build_encoder(input_shape)
        encoder_shape = self.encoder.layers[-1].output_shape
        self.decoder = self._build_decoder(encoder_shape)

    def call(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @property
    def config(self):
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
            ],
        }

    def _add_layer(self, model, layer, use_normalization=False, use_activation=False):
        layer_name = layer[0]
        filters = layer[1]
        kernel_size = layer[2]
        strides = layer[3]
        padding = layer[4]

        if (layer_name == "conv"):
            # model.add(ZeroPadding2D(padding=padding))
            model.add(TimeDistributed(Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")))
        if (layer_name == "convLSTM"):
            # model.add(ZeroPadding2D(padding=padding))
            model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size, strides=strides, padding="same", return_sequences=True))
        if (layer_name == "deconv"):
            # model.add(ZeroPadding2D(padding=padding))
            model.add(TimeDistributed(Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding="same")))

        if use_normalization:
            model.add(BatchNormalization())

        if use_activation:
            model.add(LeakyReLU())

        return model

    def _build_encoder(self, input_shape):
        encoder = Sequential(name="Encoder")
        encoder.add(Input(shape=input_shape))
        for layer in self.config.get("encoder"):
            encoder = self._add_layer(encoder, layer)
        return encoder

    def _build_decoder(self, input_shape):
        decoder = Sequential(name="Decoder")
        decoder.add(Input(shape=input_shape[1:]))
        for layer in self.config.get("decoder"):
            decoder = self._add_layer(decoder, layer)
        return decoder