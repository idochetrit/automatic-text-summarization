import pandas as pd
from .model import Seq2SeqSummarizer
import data_util
import numpy as np


def main():
    np.random.seed(42)
    model_dir_path = './saved_models'

    train_data, val_data = data_util.get_datasets()
    input_texts_val, target_texts_val, _, _ = val_data

    config_file = Seq2SeqSummarizer.get_config_file_path(model_dir_path=model_dir_path)
    config = np.load(config_file, allow_pickle=True).item()

    summarizer = Seq2SeqSummarizer(config)
    summarizer.load_weights(weight_file_path=Seq2SeqSummarizer.get_weight_file_path(model_dir_path=model_dir_path))

    print('start predicting ...')
    for i in np.random.permutation(np.arange(len(input_texts_val)))[0:20]:
        combined_higlights = "\n".join(target_texts_val[i])
        summary = summarizer.summarize("".join(input_texts_val[i]))

        print('Input text: ', "".join(input_texts_val[i]))
        print('Generated Summary: ', summary)
        print('Original Summary: ', combined_higlights)


if __name__ == '__main__':
    main()