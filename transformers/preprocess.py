
import string
import numpy as np
import pandas as pd

'''
vocab size:          3053570 
reduced vocab size:  717515
reduction %:         0.765
filtered words on showing up atleast twice in vocabulary

longest paragraph:   1015
longest summary:     397

num examples:        30845 * 100 ~= 3,084,500
'''

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
#PARAGRAPH_WINDOW_SIZE = 1024  # window size is the largest sequnese we want to read
#SUMMARY_WINDOW_SIZE = 512
PARAGRAPH_WINDOW_SIZE = 16
SUMMARY_WINDOW_SIZE = 16
##########DO NOT CHANGE#####################

def build_vocab_set(file_name):
    
    vocab = set()
    vocab = {}
    reduced_vocab = {}
    # pd json reader
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html
    reader = pd.read_json(file_name, precise_float=True, dtype=False, lines=True, chunksize=100)
    counter = 0
    for section in reader:
        #print("section['normalizedBody']: \n", section['normalizedBody'].tolist(), len(section['normalizedBody'].tolist()))
        #vocab.add(frozenset(section['normalizedBody'].tolist()))
        #vocab.add(frozenset(section['summary'].tolist()))
        for body in section['normalizedBody']:
            words = body.split()
            punc_mapping = str.maketrans('', '', string.punctuation)
            words = [ w.translate(punc_mapping).lower() for w in words ]
            for word in words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 0
        
        for body in section['summary']:
            words = body.split()
            punc_mapping = str.maketrans('', '', string.punctuation)
            words = [ w.translate(punc_mapping).lower() for w in words ]
            for word in words:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 0

    identifier = 0
    for word, count in vocab.items():
        if count > 1:
            reduced_vocab[word] = identifier
            identifier += 1
    
    print("vocab size", len(vocab), len(reduced_vocab))

    with open('vocab.csv', 'w') as f:
        f.write("word,index\n")
        for word, i in reduced_vocab.items():
            f.write("%s,%s\n"%(word, i))
    #pd.DataFrame(vocab).to_csv('vocab.csv')

def initialize_vocab(file_path):

    vocab = {}
    counter = 0
    with open(file_path, 'r') as f:
        next(f) # skip first line
        for line in f:
            line = line.strip().split(',')
            #vocab[line[0]] = line[1]
            vocab[line[0]] = counter
            counter += 1

    additional_tokens = [START_TOKEN, STOP_TOKEN, PAD_TOKEN, UNK_TOKEN]
    for token in additional_tokens:
        vocab[token] = counter
        counter += 1

    return vocab

def identify_longest_paragraph(file_name):
    
    reader = pd.read_json(file_name, precise_float=True, dtype=False, lines=True, chunksize=100)
    longest_paragraph = 0
    longest_summary = 0
    
    for section in reader:
        
        for paragraph, summary in zip(section['normalizedBody'], section['summary']):

            paragraph_len = len(paragraph.split())
            summary_len = len(summary.split())
            if paragraph_len > longest_paragraph:
                longest_paragraph = paragraph_len
            if summary_len > longest_summary:
                longest_summary = summary_len

    print("The longest paragraph:", longest_paragraph, "The longest summary:", longest_summary)


def identify_num_sentences(file_name):
    
    reader = pd.read_json(file_name, precise_float=True, dtype=False, lines=True, chunksize=100)
    num_sentences, num_large_sentences = 0, 0
    summaries = []
    large_summaries = []
    i = 0
    
    for section in reader:
        i += 1
        for summary in section['summary']:
            summary_len = len(summary.split('.'))
            if summary_len > 1:
                num_large_sentences += 1
                large_summaries.append(summary_len)
            num_sentences += 1
            summaries.append(summary_len)
    print(i, "steps with batch size", 100)
    print("average summary num sentences:", np.mean(summaries))
    print("average large summary num sentences:", np.mean(large_summaries))
    print("number of sentences", num_sentences)
    print("number of large sentences", num_large_sentences)
    print("average", num_large_sentences / num_sentences)


    
def convert_to_id(vocab, sentences):
    """
    DO NOT CHANGE

    Convert sentences to indexed 

    :param vocab:  dictionary, word --> unique index
    :param sentences:  list of lists of words, each representing padded sentence
    :return: numpy array of integers, with each row representing the word indeces in the corresponding sentences
    """
    #print("sentences shape:", len(sentences), len(sentences[0]))
    for sentence in sentences:
        if len(sentence) != SUMMARY_WINDOW_SIZE:
            print("sentence not in right shape:", len(sentence))
    return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])


def pad_corpus(sentences, window_size):
    """
    
    :param sentences: dim(num_batches, variable_text_length)
    :param french: list of French sentences
    :param english: list of English sentences
    :return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
    """
    '''
    padded_sentences = []
    for line in sentences:
        padded_FRENCH = line[:FRENCH_WINDOW_SIZE]
        padded_FRENCH += [STOP_TOKEN] + [PAD_TOKEN] * (FRENCH_WINDOW_SIZE - len(padded_FRENCH)-1)
        FRENCH_padded_sentences.append(padded_FRENCH)
    '''
    #print("length of sentences", len(sentences), len(sentences[0].split()), "\n", sentences[0])
    # text can be a body paragraph or a summary
    pad_sentences = []
    i = 0
    for text in sentences:
        text = text.split()
        #print("length of pre-padded text", len(text), end="\t")
        pad_sentences.append(text[:window_size-1] + [STOP_TOKEN] + [PAD_TOKEN] * (window_size - len(text)-1))
        #print("length of post padded text", len(pad_sentences[i]))
        i += 1
    #print("length of post-padded text", len(pad_sentences[0]))
    return pad_sentences



if __name__ == '__main__':
    # get_data('../data/tldr-training-data.jsonl')
    #build_vocab_set('../data/tldr-training-data.jsonl')
    print("loading vocab...")
    #vocab = initialize_vocab('../data/reduced_vocab.csv')

    #identify_longest_paragraph('../data/tldr-training-data.jsonl')
    #identify_num_sentences('../data/tldr-training-data.jsonl')
    vocab = initialize_vocab('../data/reduced_vocab.csv')
    print("vocab length:", len(vocab), "vocab unk", vocab[UNK_TOKEN])
    
