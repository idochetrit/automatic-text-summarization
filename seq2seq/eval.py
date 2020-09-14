import pandas as pd
from .model import Seq2SeqSummarizer
from .. import data_util
from ..decoder_encoder_lstm import utils
import numpy as np


def main():
    np.random.seed(42)
    data_dir_path = './data'
    model_dir_path = '../saved_models'

    train_data, val_data = data_util.get_datasets()
    X1_val, X2_val, y_val = utils.preprocessing(val_data)

    config = np.load(Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)).item()

    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for i in np.random.permutation(np.arange(len(X1_val)))[0:20]:
        combined_higlights = y_val[i]
        headline = summarizer.summarize(X1_val[i])

        print('Generated Summary: ', headline)
        print('Original Summary: ', combined_higlights)


if __name__ == '__main__':
    main()