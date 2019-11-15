import numpy as np
import tensorflow as tf
import numpy as np
import pandas as pd
#import ijson
#from functools import reduce


##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
FRENCH_WINDOW_SIZE = 14
ENGLISH_WINDOW_SIZE = 14
##########DO NOT CHANGE#####################

def pad_corpus(french, english):
    """
    DO NOT CHANGE:

    arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
    text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
    the end.

    :param french: list of French sentences
    :param english: list of English sentences
    :return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
    """
    FRENCH_padded_sentences = []
    FRENCH_sentence_lengths = []
    for line in french:
        padded_FRENCH = line[:FRENCH_WINDOW_SIZE]
        padded_FRENCH += [STOP_TOKEN] + [PAD_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_FRENCH)-1)
        FRENCH_padded_sentences.append(padded_FRENCH)

    ENGLISH_padded_sentences = []
    ENGLISH_sentence_lengths = []
    for line in english:
        padded_ENGLISH = line[:ENGLISH_WINDOW_SIZE]
        padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN] + [PAD_TOKEN] * (ENGLISH_WINDOW_SIZE - len(padded_ENGLISH)-1)
        ENGLISH_padded_sentences.append(padded_ENGLISH)

    return FRENCH_padded_sentences, ENGLISH_padded_sentences

def build_vocab(sentences):
    """
    DO NOT CHANGE

    Builds vocab from list of sentences

    :param sentences:  list of sentences, each a list of words
    :return: tuple of (dictionary: word --> unique index, pad_token_idx)
    """
    tokens = []
    for s in sentences: tokens.extend(s)
    all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN] + tokens)))

    vocab =  {word:i for i,word in enumerate(all_words)}

    return vocab,vocab[PAD_TOKEN]

def convert_to_id(vocab, sentences):
    """
    DO NOT CHANGE

    Convert sentences to indexed 

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

  
def read_data(file_name):
    """
    DO NOT CHANGE

    Load text data from file

    :param file_name:  string, name of data file
    :return: list of sentences, each a list of words split on whitespace
    """
    '''
    text = []
    with open(file_name, 'rt', encoding='latin') as data_file:
        for line in data_file: text.append(line.split())
    '''
    train, test = [], []
    reader = pd.read_json(file_name, lines=True, chunksize=100)
    counter = 0
    for section in reader:
        train = section['normalizedBody']
        test = section['summary']
        #print("train, test", train, "\n", test)
        counter += 1

    print("number of counters in data", counter)
    return train, test


def get_data(json_file):
    """
    Use the helper functions in this file to read and parse training and test data, then pad the corpus.
    Then vectorize your train and test data based on your vocabulary dictionaries.

    :param french_training_file: Path to the french training file.
    :param english_training_file: Path to the english training file.
    :param french_test_file: Path to the french test file.
    :param english_test_file: Path to the english test file.

    :return: Tuple of train containing:
    (2-d list or array with english training sentences in vectorized/id form [num_sentences x 15] ),
    (2-d list or array with english test sentences in vectorized/id form [num_sentences x 15]),
    (2-d list or array with french training sentences in vectorized/id form [num_sentences x 14]),
    (2-d list or array with french test sentences in vectorized/id form [num_sentences x 14]),
    english vocab (Dict containg word->index mapping),
    french vocab (Dict containg word->index mapping),
    english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
    """
    # step 1 read text and summary data
    '''
    train_french_text = read_data(french_training_file)
    train_english_text = read_data(english_training_file)
    test_french_text = read_data(french_test_file)
    test_english_text = read_data(english_test_file)

    #2) Pad training data (see pad_corpus)
    train_french_text, train_english_text = pad_corpus(train_french_text, train_english_text)

    #3) Pad testing data (see pad_corpus)
    test_french_text, test_english_text = pad_corpus(test_french_text, test_english_text)

    #4) Build vocab for french (see build_vocab)
    french_vocab, french_pad_idx = build_vocab(train_french_text)

    #5) Build vocab for english (see build_vocab)
    english_vocab, english_pad_idx = build_vocab(train_english_text)

    #7) Convert training and testing french sentences to list of IDS (see convert_to_id)
    train_french = convert_to_id(french_vocab, train_french_text)
    test_french = convert_to_id(french_vocab, test_french_text)
    #print("train french\n", train_french[0], "\n", train_french[1], "\n", train_french[2], "\n")
    
    #6) Convert training and testing english sentences to list of IDS (see convert_to_id)
    train_english = convert_to_id(english_vocab, train_english_text)
    test_english = convert_to_id(english_vocab, test_english_text)

    return train_english, test_english, train_french, test_french, english_vocab, french_vocab, english_pad_idx
    '''
    train, test = read_data(json_file)
	

if __name__ == '__main__':
    get_data('../data/tldr-training-data.jsonl')


