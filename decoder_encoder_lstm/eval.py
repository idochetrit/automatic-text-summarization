from numpy import array_equal
import numpy as np
from rouge import Rouge 
from .utils import predict_sequence, one_shot_decode, preprocessing

def evaluate(val_data, train_data, model, encoder_layer, decoder_layer, n_output_tokens):
    rouge = Rouge()
    input_texts, target_texts, input_sequences, target_sequences = train_data
    
    train_target_texts, _, train_target_sequences, _ = train_data
    max_decoder_seq_length = max([len(txt) for txt in train_target_texts])
    target_token_index = dict([(char, i) for i, char in enumerate(train_target_sequences)])
    reverse_target_token_index = dict(
        (i, char) for char, i in target_token_index.items()
    )

    X1_test, X2_test, y_test = preprocessing(train_data)

   # example runs
    for idx in range(3):
        input_seq = X1_test[idx : idx + 1]
        predicted = predict_sequence(
            encoder_layer,
            decoder_layer,
            input_seq,
            n_output_tokens,
            reverse_target_token_index,
            max_decoder_seq_length,
        )

        print("- -- -")
        # print("Input sentence:", "\n".join(input_texts[idx]))
        print("Ground truth sentence:", "\n".join(target_texts[idx]))
        print("Decoded sentence:", predicted)
        print("- -- -")
    
    # evaluate LSTM
    total = len(X1_test)
    refs = []
    hyps = []
    for idx in range(total):
        input_seq = X1_test[idx : idx + 1]
        predicted_summ = predict_sequence(
            encoder_layer,
            decoder_layer,
            input_seq,
            n_output_tokens,
            reverse_target_token_index,
            max_decoder_seq_length,
        )
        refs.append("\n".join(input_texts[idx]))
        hyps.append(predicted_summ)

    rouge_scores = rouge.get_scores(hyps, refs, avg=True)
    print("Average ROUGE Score: {}".format(rouge_scores))

  