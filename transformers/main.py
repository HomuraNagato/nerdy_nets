import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
import sys
import time

PARAGRAPH_WINDOW_SIZE = 1024  # window size is the largest sequnese we want to read
SUMMARY_WINDOW_SIZE = 1024


def train(model, file_name, vocab, paragraph_window_size, summary_window_size, eng_padding_index):
    """
    Train transformer model

    :param model: the initilized model to use for forward and backward pass
    :param train: train data  of shape (num_sentences, 14)
    :param test: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :return: None
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    reader = pd.read_json(file_name, precise_float=True, dtype=False, lines=True, chunksize=100)
    train_steps = 0
    
    for section in reader:
        #print("section['normalizedBody']: \n", section['normalizedBody'].tolist(), len(section['normalizedBody'].tolist()))

        # get paragraph and summary
        train_batch = section['normalizedBody'].tolist()
        test_batch = section['summary'].tolist()

        # normalize
        punc_mapping = str.maketrans('', '', string.punctuation)
        #train_words = [ w.translate(punc_mapping).lower() for paragraph in train_batch for w in paragraph.split() ]
        train_words = [ w.translate(punc_mapping).lower() for w in train_batch ]
        test_words = [ w.translate(punc_mapping).lower() for w in test_batch ]

        start_print, stop_print = 96, 98
        

        print("cleaned words")
        for paragraph, summary in zip(train_words[start_print:stop_print], test_words[start_print:stop_print]):
            print("\nparagraph\n", np.array(paragraph))
            print("summary\n", np.array(summary))

        
        # fix length
        train_words = pad_corpus(train_words, paragraph_window_size)
        test_words = pad_corpus(test_words, summary_window_size)
        '''
        print("pad words")
        for paragraph, summary in zip(train_words[start_print:stop_print], test_words[start_print:stop_print]):
            print("\nparagraph\n", len(paragraph))
            print("summary\n", len(summary))
        '''
            
        # dict lookup
        train_words = convert_to_id(vocab, train_words)
        test_words = convert_to_id(vocab, test_words)

        print("id words")
        for paragraph, summary in zip(train_words[start_print:stop_print], test_words[start_print:stop_print]):
            print("\nparagraph\n", np.array(paragraph))
            print("summary\n", np.array(summary))

        
        #mask = np.not_equal(loss_english, eng_padding_index)

        train_words = tf.convert_to_tensor(train_words, dtype=tf.int32)
        test_words = tf.convert_to_tensor(test_words, dtype=tf.int32)        
        '''
        print("id words as tensor")
        for paragraph, summary in zip(train_words[10:12], test_words[10:12]):
            print("\nparagraph\n", paragraph)
            print("summary\n", summary)
        '''

        # Implement backprop:
        print("section step", train_steps)

        with tf.GradientTape() as tape:
            probs  = model(train_words, test_words)
            print("probs", probs)
            loss = 0
            exit(0)
            '''
            loss = model.loss_function(probs, test_words)

            if i % 10000 == 0:
                perplexity = tf.exp(loss)
                print("training steps: {}. model loss: {}. perplexity: {}".format(i, loss, perplexity))

                if i % 100 == 0:
                    break
            '''

        '''
        trainable_variables = model.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))
        '''


    return 0

def test(model, test_french, test_english, eng_padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initilized model to use for forward and backward pass
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :returns: perplexity of the test set, per symbol accuracy on test set
    """

    # Note: Follow the same procedure as in train() to construct batches of data!
    loss = 0
    accuracy = 0
    #accuracy_mult = 0    
    #num_non_padded = 0    
    num_batches = 0
    
    print("testing model")
    for i in range(0, len(test_french), model.batch_size):

        num_batches += 1
        batch_english = test_english[i:i+model.batch_size]
        encoder_english = tf.convert_to_tensor([ x[:-1] for x in batch_english ])
        loss_english = tf.convert_to_tensor([ x[1:] for x in batch_english ], dtype=tf.int64)
        batch_french = test_french[i:i+model.batch_size]
        mask = np.not_equal(loss_english, eng_padding_index)

        #batch_padded = np.sum(mask)
        #num_non_padded += batch_padded
        probs  = model(batch_french, encoder_english)
        loss += model.loss_function(probs, loss_english, mask) 
        #batch_accuracy = model.accuracy_function(probs, loss_english, mask)
        accuracy += model.accuracy_function(probs, loss_english, mask)
        #accuracy_mult += batch_accuracy * batch_padded

    loss = loss / num_batches
    accuracy = accuracy / num_batches
    return loss, accuracy

def main():	

    start_time = time.time()

    vocab = initialize_vocab('../data/reduced_vocab.csv')
    print("vocab length:", len(vocab), "vocab unk", vocab[UNK_TOKEN])
    
    padding_index = vocab[PAD_TOKEN]
    
    model = Transformer_Seq2Seq(len(vocab), PARAGRAPH_WINDOW_SIZE, SUMMARY_WINDOW_SIZE)

    print("training model")
    train(model, '../data/tldr-training-data.jsonl', vocab, PARAGRAPH_WINDOW_SIZE, SUMMARY_WINDOW_SIZE, padding_index)
    '''
    loss, accuracy = test(model, test_french, test_english, eng_padding_index)
    perplexity = np.exp(loss)
    print("model test perplexity: {}. model test accuracy: {}".format(perplexity, accuracy))
    end_time = time.time()
    print("The model took:", end_time-start_time,"seconds to train")
    '''

if __name__ == '__main__':
    main()


