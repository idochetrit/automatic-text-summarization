from decoder_encoder_lstm.train import train as train
from decoder_encoder_lstm.eval import evaluate as evaluate
import data_util

if __name__ == '__main__':
    train_data, val_data = data_util.get_datasets()

    model, encoder_layer, decoder_layer, n_output_tokens = train(train_data)
    model.save("./saved_models/last_changed.h5")
    
    evaluate(val_data, train_data, model, encoder_layer, decoder_layer, n_output_tokens)


