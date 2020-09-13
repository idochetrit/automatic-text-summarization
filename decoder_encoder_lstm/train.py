import data_util
import numpy as np
from .keras_model import init_keras_models
from .utils import predict_sequence, one_shot_decode, preprocessing
from keras.optimizers import RMSprop

# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()


def train(train_data):
    input_texts, target_texts, input_sequences, target_sequences = train_data

    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])
    input_token_index = dict([(char, i) for i, char in enumerate(input_sequences)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_sequences)])
    reverse_input_token_index = dict((i, char) for char, i in input_token_index.items())
    reverse_target_token_index = dict(
        (i, char) for char, i in target_token_index.items()
    )

    X1, X2, y = preprocessing(train_data)
    # generate training dataset
    print(X1.shape, X2.shape, y.shape)
    n_input_tokens, n_output_tokens = (X1.shape[2], X2.shape[2])
    latent_dim = 128

    model, encoder_layer, decoder_layer = init_keras_models(
        n_input_tokens, n_output_tokens, latent_dim
    )
    rmsprops = RMSprop(lr=0.005, clipnorm=2.0)
    model.compile(optimizer=rmsprops,metrics=['accuracy'], loss="categorical_crossentropy")

    # train phase
    model.fit([X1, X2], y, epochs=10, batch_size=32, validation_split=0.3)

    return model, encoder_layer, decoder_layer, n_output_tokens