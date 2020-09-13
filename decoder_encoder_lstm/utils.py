import numpy as np
from random import randint
from numpy import argmax
from numpy import array

def predict_sequence(encoder_model, decoder_model, input_seq, n_output_tokens, reverse_target_token_index, max_decoder_seq_length):
  padded_seq = np.pad(input_seq, [(0,0), (0,0), (encoder_model.input_shape[2] - input_seq.shape[2],0)])
  states_value = encoder_model.predict(padded_seq)

  target_seq = np.zeros((1, 1, n_output_tokens))

  stop_condition = False
  predict_sequence = ''
  while True:
    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    sampled_token_index = argmax(output_tokens[0, -1, :])
    sampled_string = reverse_target_token_index[sampled_token_index]
    predict_sequence += sampled_string + " "

    if (sampled_string == '\n' or len(predict_sequence) > max_decoder_seq_length):
      break

    target_seq = np.zeros((1, 1, n_output_tokens))
    target_seq[0, 0, sampled_token_index] = 1.0
    states_value = [h, c]

  return predict_sequence

def one_shot_decode(encoded_seq):
  return [argmax(vector) for vector in encoded_seq]


def preprocessing(data):
    input_texts, target_texts, input_sequences, target_sequences = data
    num_encoder_tokens = len(input_sequences)
    num_decoder_tokens = len(target_sequences)
    print("Number of unique input tokens:", num_encoder_tokens)
    print("Number of unique output tokens:", num_decoder_tokens)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    input_token_index = dict([(char, i) for i, char in enumerate(input_sequences)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_sequences)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, seq in enumerate(input_text):
            encoder_input_data[i, t, input_token_index[seq]] = 1.0
        for t, seq in enumerate(target_text):
            decoder_input_data[i, t, target_token_index[seq[:-1]]] = 1.0
            if t > 0:
                decoder_target_data[i, t - 1, target_token_index[seq[:-1]]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data