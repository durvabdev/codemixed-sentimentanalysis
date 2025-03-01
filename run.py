import argparse, os

import numpy as np

from formdataset import *
import WordCNN_LSTM as EBCNNLSTM
import WordCNNLSTM as SWCNNLSTM
from collections import Counter
import Word_BiLSTM as BiLSTM
embeddingpath = 'embeddings'
"""

this code is the starting point & takes inputs to run code, processes data
run it as
    example: python3 run.py -d test
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'arguments to run code')
    parser.add_argument('-d', '--datasets', nargs = '+', type = str, default = 'test', help = 'input datasets')
    parser.add_argument('-e', '--embedding', type = str, choices = ['mbert', 'fasttext', 'bpemb'],default = 'bpemb', help = 'input embeddings')
    args = parser.parse_args()
    assert args.datasets is not None, "mention datasets"

    print("loading datasets")
    if (os.path.exists(f'{embeddingpath}/{args.embedding}_X.npy') \
    and os.path.exists(f'{embeddingpath}/{args.embedding}_Y.npy')):
        print(f"files exist. loading from {embeddingpath}")
        x_array, y_array = np.load(f'{embeddingpath}/{args.embedding}_X.npy')[:99200], np.load(f'{embeddingpath}/{args.embedding}_Y.npy')[:99200]
        count = Counter(y_array)
        # Print the distribution
        for value, freq in count.items():
            print(f"Value: {value}, Frequency: {freq}")
        #print(x_array.shape)
        if args.embedding == 'bpemb':
            SWCNNLSTM.wrapper(x_array.astype(np.float32), y_array.astype(np.int64))
        else:
            #EBCNNLSTM.wrapper(x_array, y_array)
            BiLSTM.wrapper(x_array, y_array)
    else:
        data = returndataset(args.datasets)
        x_array, y_array = process(data, args.embedding)
        if args.embedding == 'bpemb':
            SWCNNLSTM.wrapper(x_array, y_array)
        else:
            BiLSTM.wrapper(x_array, y_array)
            #EBCNNLSTM.wrapper(x_array, y_array)
