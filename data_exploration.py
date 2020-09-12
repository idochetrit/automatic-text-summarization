import data_util
import numpy as np

batch_size = 64
epochs = 110
latent_dim = 256
num_samples = 10000
train_data, val_data = data_util.get_datasets();

input_texts, target_texts, input_sequences, target_sequences = train_data
print(input_texts[0])



num_encoder_tokens = len(input_sequences)
num_decoder_tokens = len(target_sequences)

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

input_token_index = dict([(char, i) for i, char in enumerate(input_sequences)])
target_token_index = dict([(char, i) for i, char in enumerate(target_sequences)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

print("decoder_target_data shape", decoder_input_data.shape)

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, seq in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[seq]] = 1.0
    for t, seq in enumerate(target_text):
        decoder_input_data[i, t, target_token_index[seq[:-1]]] = 1.0

print("encoder input shape", encoder_input_data.shape)
print("decoder input shape", decoder_input_data.shape)