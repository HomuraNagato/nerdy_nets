import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
import sys
import time

tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# largest paragraph: 1024. Largest Summary: 400.
PARAGRAPH_WINDOW_SIZE = 16  # window size is the largest sequnese we want to read
SUMMARY_WINDOW_SIZE = 16


def train(model, file_name, vocab, reverse_vocab, paragraph_window_size, summary_window_size, eng_padding_index):
    """
    Train transformer model

    :param model: the initilized model to use for forward and backward pass
    :param train: train data  of shape (num_sentences, 14)
    :param test: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :return: None
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    reader = pd.read_json(file_name, precise_float=True, dtype=False, lines=True, chunksize=10)
    train_steps = 0
    step_start_time = time.time()
    training_time = [step_start_time]
    
    for section in reader:
        #print("section['normalizedBody']: \n", section['normalizedBody'].tolist(), len(section['normalizedBody'].tolist()))
        train_steps += 1
        
        # get paragraph and summary
        train_batch = section['normalizedBody'].tolist()
        test_batch = section['summary'].tolist()
        # normalize
        punc_mapping = str.maketrans('', '', string.punctuation)
        #train_words = [ w.translate(punc_mapping).lower() for paragraph in train_batch for w in paragraph.split() ]
        train_words = [ w.translate(punc_mapping).lower() for w in train_batch ]
        test_words = [ w.translate(punc_mapping).lower() for w in test_batch ]
        ori_train_words, ori_test_words = np.array(train_words), np.array(test_words)

        start_print, stop_print = 6, 8
        '''
        print("cleaned words", ori_train_words.shape)
        for paragraph, summary in zip(ori_train_words[start_print:stop_print], ori_test_words[start_print:stop_print]):
            print("\nparagraph\n", np.array(paragraph))
            print("summary\n", np.array(summary))
        '''
        
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
        '''
        print("id words")
        for paragraph, summary in zip(train_words[start_print:stop_print], test_words[start_print:stop_print]):
            print("\nparagraph\n", np.array(paragraph))
            print("summary\n", np.array(summary))
        '''
        
        mask = np.not_equal(test_words, eng_padding_index)

        train_words = tf.convert_to_tensor(train_words, dtype=tf.int32)
        test_words = tf.convert_to_tensor(test_words, dtype=tf.int32)        
        '''
        print("id words as tensor")
        for paragraph, summary in zip(train_words[10:12], test_words[10:12]):
            print("\nparagraph\n", paragraph)
            print("summary\n", summary)
        '''

        # Implement backprop:
        #print("section step", train_steps)
        print("step %02d" % (train_steps), end='\r')


        with tf.GradientTape() as tape:
            probs  = model(train_words, test_words)
            #print("probs", probs)

            loss = model.loss_function(probs, test_words, mask)

            if train_steps % 10 == 0:
                
                step_inter_time = time.time()
                training_time.append(step_inter_time)
                average_training_time = np.mean([ training_time[i+1] - training_time[i] for i, x in enumerate(training_time[-11:-1]) ])
                perplexity = tf.exp(loss)
                print("current_time: {} average_training_time: {} model loss: {}. perplexity: {}".format(step_inter_time-step_start_time, average_training_time, loss, perplexity))
                
                model.produce_sentence(np.array(ori_train_words[0]), np.array(test_words[0]), probs[0], reverse_vocab, SUMMARY_WINDOW_SIZE)

                '''
                if i % 100 == 0:
                    break

                '''
        trainable_variables = model.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))


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
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    print("vocab length:", len(vocab), "vocab unk", vocab[UNK_TOKEN])
    
    padding_index = vocab[PAD_TOKEN]
    
    model = Transformer_Seq2Seq(len(vocab), PARAGRAPH_WINDOW_SIZE, SUMMARY_WINDOW_SIZE)

    print("training model")
    train(model, '../data/tldr-training-data.jsonl', vocab, reverse_vocab, PARAGRAPH_WINDOW_SIZE, SUMMARY_WINDOW_SIZE, padding_index)
    '''
    loss, accuracy = test(model, test_french, test_english, eng_padding_index)
    perplexity = np.exp(loss)
    print("model test perplexity: {}. model test accuracy: {}".format(perplexity, accuracy))
    end_time = time.time()
    print("The model took:", end_time-start_time,"seconds to train")
    '''

if __name__ == '__main__':
    main()


