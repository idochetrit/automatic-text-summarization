import pandas as pd
import data_util
from .plt import plot_and_save_history
from .model import Seq2SeqSummarizer
from .config import fit_text
import numpy as np


def main():
    np.random.seed(42)
    report_dir_path = '../reports'
    model_dir_path = '../saved_models'

    train_data, val_data = data_util.get_datasets()
    input_texts_train, target_texts_train, _, _ = train_data
    input_texts_val, target_texts_val, _, _ = val_data
    
    config = fit_text(input_texts_train, target_texts_train)
    summarizer = Seq2SeqSummarizer(config)

    print('demo size: ', len(input_texts_train))
    print('testing size: ', len(input_texts_train))

    print('start fitting ...')
    history = summarizer.fit(input_texts_train, target_texts_train, input_texts_val, target_texts_val, epochs=10)

    history_plot_file_path = report_dir_path + '/' + Seq2SeqSummarizer.model_name + '-history.png'
    plot_and_save_history(history, summarizer.model_name, history_plot_file_path, metrics={'loss', 'acc'})


if __name__ == '__main__':
    main()