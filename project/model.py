import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from transformer_model import Transformer_Seq2Seq
from rnn_model import RNN_Seq2Seq
import sys
import time

def train(model, train_french, train_english, eng_padding_index):
    """
    Train transformer model

    :param model: the initilized model to use for forward and backward pass
    :param train_french: french train data (all data for training) of shape (num_sentences, 14)
    :param train_english: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :return: None
    """

    # NOTE: For each training step, you should pass in the french sentences to be used by the encoder, 
    # and english sentences to be used by the decoder
    # - The english sentences passed to the decoder have the last token in the window removed:
    #	 [STOP CS147 is the best class. STOP *PAD*] --> [STOP CS147 is the best class. STOP] 
    # 
    # - When computing loss, the decoder labels should have the first word removed:
    #	 [STOP CS147 is the best class. STOP] --> [CS147 is the best class. STOP]

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    for i in range(0, len(train_french), model.batch_size):
        
        batch_english = train_english[i:i+model.batch_size]
        encoder_english = tf.convert_to_tensor([ x[:-1] for x in batch_english ])
        loss_english = tf.convert_to_tensor([ x[1:] for x in batch_english ], dtype=tf.float32)
        batch_french = train_french[i:i+model.batch_size]
        mask = np.not_equal(loss_english, eng_padding_index)

        # Implement backprop:
        with tf.GradientTape() as tape:
            probs  = model(batch_french, encoder_english)
            loss = model.loss_function(probs, loss_english, mask)

            if i % 10000 == 0:
                perplexity = tf.exp(loss)
                print("training steps: {}. model loss: {}. perplexity: {}".format(i, loss, perplexity))
                '''
                if (i % 10000) or (i >= 10000) == 0:
                    break
                '''
        trainable_variables = model.trainable_variables
        
        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))


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
    if len(sys.argv) != 2 or sys.argv[1] not in {"RNN","TRANSFORMER"}:
        print("USAGE: python assignment.py <Model Type>")
        print("<Model Type>: [RNN/TRANSFORMER]")
        exit()

    start_time = time.time()
    print("Preprocessing...")
    train_english,test_english, train_french,test_french, english_vocab,french_vocab,eng_padding_index = get_data('data/fls.txt','data/els.txt','data/flt.txt','data/elt.txt')
    print("Preprocessing complete.")

    model_args = (FRENCH_WINDOW_SIZE,len(french_vocab),ENGLISH_WINDOW_SIZE, len(english_vocab))
    if sys.argv[1] == "RNN":
        model = RNN_Seq2Seq(*model_args)
    elif sys.argv[1] == "TRANSFORMER":
        model = Transformer_Seq2Seq(*model_args) 

    # TODO:
    # Train and Test Model for 1 epoch.
    train(model, train_french, train_english, eng_padding_index)
    loss, accuracy = test(model, test_french, test_english, eng_padding_index)
    perplexity = np.exp(loss)
    print("model test perplexity: {}. model test accuracy: {}".format(perplexity, accuracy))
    end_time = time.time()
    print("The model took:", end_time-start_time,"seconds to train")

if __name__ == '__main__':
    main()


