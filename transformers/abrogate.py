
import string
import json
import jsonlines
import numpy as np
import pandas as pd


if __name__ == '__main__':

    file_path = '../data/tldr-training-data.jsonl'
    batch_size = 100
    reader = pd.read_json(file_path, precise_float=True, dtype=False, lines=True, chunksize=batch_size)

    num_sentences, num_large_sentences = 0, 0
    summaries = []
    large_summaries = []
    i = 0
    num_excluded = 0

    
    for section in reader:
        train_batch = section['normalizedBody'].tolist()
        test_batch = section['summary'].tolist()


        for train, test in zip(train_batch, test_batch):

            #print("train\n", train, "test\n", test)

            if '.' in test:
                # normalize
                punc_mapping = str.maketrans('', '', string.punctuation)
                train_words = [ w.translate(punc_mapping).lower() for w in train ]
                test_words = [ w.translate(punc_mapping).lower() for w in test ]
                json_out = {'i': i, 'train': ''.join(train_words), 'test': ''.join(test_words) }
                i += 1
                
                with jsonlines.open('cleaned_data.json', mode='a') as writer:
                    writer.write(json_out)
                



