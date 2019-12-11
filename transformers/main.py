
import os
import numpy as np
import tensorflow as tf
import numpy as np
import sys
import time

from preprocess import *
from transformer_model import Transformer_Seq2Seq
from gpt2_model import *



#tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# largest paragraph: 1024. Largest Summary: 400.
PARAGRAPH_WINDOW_SIZE = 32  # window size is the largest sequnese we want to read
SUMMARY_WINDOW_SIZE = 32

break_line = 1000
print_line = 20
gpt_line = 10

def train(model, torch_model, tokenizer, file_name, vocab, reverse_vocab, paragraph_window_size, summary_window_size, eng_padding_index):
    """
    Train transformer model

    :param model: the initilized model to use for forward and backward pass
    :param train: train data  of shape (num_sentences, 14)
    :param test: english train data (all data for training) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :return: None
    """
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
    result_file = open("result_file.txt", "w+")
    reader = pd.read_json(file_name, precise_float=True, dtype=False, lines=True, chunksize=10)
    train_steps = 0
    step_start_time = time.time()
    training_time = [step_start_time]
    
    for section in reader:
        #print("section['normalizedBody']: \n", section['normalizedBody'].tolist(), len(section['normalizedBody'].tolist()))
        train_steps += 1
        generate_index = 0
        # get paragraph and summary
        train_words = section['normalizedBody'].tolist()
        test_words = section['summary'].tolist()

        # save first entry from batch for testing on gpt2
        origin_paragraph = train_words[generate_index]
        origin_summary = test_words[generate_index]
        
        # normalize
        punc_mapping = str.maketrans('', '', string.punctuation)
        #train_words = [ w.translate(punc_mapping).lower() for paragraph in train_batch for w in paragraph.split() ]
        train_words = [ w.translate(punc_mapping).lower() for w in train_words ]
        test_words = [ w.translate(punc_mapping).lower() for w in test_words ]
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

        train_words = tf.convert_to_tensor(train_words, dtype=tf.int64)
        test_words = tf.convert_to_tensor(test_words, dtype=tf.int64)        
        '''
        print("id words as tensor")
        for paragraph, summary in zip(train_words[10:12], test_words[10:12]):
            print("\nparagraph\n", paragraph)
            print("summary\n", summary)
        '''

        # Implement backprop:
        #print("section step", train_steps)
        print("step %02d / %02d" % (train_steps, break_line), end='\r')

        with tf.GradientTape() as tape:
            probs  = model(train_words, test_words)
            #print("probs", probs)

            loss = model.loss_function(probs, test_words, mask)

            if train_steps % gpt_line == 0:
                
                generated_summary = model.produce_sentence(probs[generate_index], reverse_vocab)
                #gpt_loss, generated = sample_sentence(torch_model, tokenizer, origin_paragraph, origin_summary)
                gpt_loss, generated = sample_sentence(torch_model, tokenizer, origin_paragraph, generated_summary)
                
                perplexity = tf.exp(loss)
                model_loss = loss
                loss += gpt_loss
                
                
                if train_steps % print_line == 0:

                    step_inter_time = time.time()
                    training_time.append(step_inter_time)
                    average_training_time = np.mean([ training_time[i+1] - training_time[i] for i, x in enumerate(training_time[-11:-1]) ])
                    #decoded_ori_paragraph = " ".join([ reverse_vocab[x] for x in train_words[generate_index].numpy() ])
                    #print("decoded original paragraph", decoded_ori_paragraph)
                    print("original paragraph\n", origin_paragraph, "\n")
                    print("generated paragraph from gpt2\n", generated, "\n")
                    print("original summary\n", origin_summary, "\n")
                    print("generated summary from transformers\n", generated_summary, "\n")

                    print("current_time: {} average_training_time: {} model loss: {}. perplexity: {}. gpt_loss: {}".format(step_inter_time-step_start_time, average_training_time, model_loss, perplexity, gpt_loss))
                    result_file.write("\n********************************\n")
                    result_file.write("iteration: {} current_time: {} average_training_time: {} model loss: {}. perplexity: {}. gpt_loss: {}".format(train_steps, step_inter_time-step_start_time, average_training_time, model_loss, perplexity, gpt_loss))
                    result_file.write("original paragraph\n", origin_paragraph, "\n")
                    result_file.write("generated paragraph from gpt2\n", generated, "\n")
                    result_file.write("original summary\n", origin_summary, "\n")
                    result_file.write("generated summary from transformers\n", generated_summary, "\n")
                    exit(1)
                    

        if train_steps % break_line == 0:
            result_file.close()
            break
                
        trainable_variables = model.trainable_variables

        gradients = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))


    return 0

def test(model, file_name, vocab, reverse_vocab, paragraph_window_size, summary_window_size, eng_padding_index):
    """
    Runs through one epoch - all testing examples.

    :param model: the initilized model to use for forward and backward pass
    :param test_french: french test data (all data for testing) of shape (num_sentences, 14)
    :param test_english: english test data (all data for testing) of shape (num_sentences, 15)
    :param eng_padding_index: the padding index, the id of *PAD* token. This integer is used to mask padding labels.
    :returns: perplexity of the test set, per symbol accuracy on test set
    """

    loss = 0
    accuracy = 0
    test_steps = 0
    start_time = time.time()
    reader = pd.read_json(file_name, precise_float=True, dtype=False, lines=True, chunksize=10)
    
    print("testing model")
    for section in reader:

        test_steps += 1
        
        # get paragraph and summary
        train_words = section['normalizedBody'].tolist()
        test_words = section['summary'].tolist()
        # normalize
        punc_mapping = str.maketrans('', '', string.punctuation)
        train_words = [ w.translate(punc_mapping).lower() for w in train_words ]
        test_words = [ w.translate(punc_mapping).lower() for w in test_words ]
        ori_train_words, ori_test_words = np.array(train_words), np.array(test_words)
        # fix length
        train_words = pad_corpus(train_words, paragraph_window_size)
        test_words = pad_corpus(test_words, summary_window_size)
        # dict lookup
        train_words = convert_to_id(vocab, train_words)
        test_words = convert_to_id(vocab, test_words)
        mask = np.not_equal(test_words, eng_padding_index)
        # tensor
        train_words = tf.convert_to_tensor(train_words, dtype=tf.int64)
        test_words = tf.convert_to_tensor(test_words, dtype=tf.int64)
        # test probability
        probs  = model(train_words, test_words)
        # loss and accuracy
        loss += model.loss_function(probs, test_words, mask)
        accuracy += model.accuracy_function(probs, test_words, mask)
        if test_steps % print_line/10 == 0:
            inter_time = time.time()
            inter_perplexity = tf.exp(loss / test_steps)
            inter_accuracy = accuracy / test_steps
            print("test_step {}. test_time: {}. perplexity: {}. accuracy: {}".format(test_steps, inter_time-start_time, inter_perplexity, inter_accuracy))

            model.produce_sentence(np.array(ori_train_words[0]), np.array(test_words[0]), probs[0], reverse_vocab, SUMMARY_WINDOW_SIZE)

    loss = loss / test_steps
    accuracy = accuracy / test_steps
    return loss, accuracy

def main():	

    start_time = time.time()

    vocab = initialize_vocab('../data/reduced_vocab.csv')
    reverse_vocab = {idx:word for word, idx in vocab.items()}
    
    padding_index = vocab[PAD_TOKEN]
    
    model = Transformer_Seq2Seq(len(vocab), PARAGRAPH_WINDOW_SIZE, SUMMARY_WINDOW_SIZE)
    
    torch_model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


    print("training model")
    train(model, torch_model, tokenizer, '../data/tldr_train80.jsonl', vocab, reverse_vocab, PARAGRAPH_WINDOW_SIZE, SUMMARY_WINDOW_SIZE, padding_index)

    loss, accuracy = test(model, '../data/tldr_test20.jsonl', vocab, reverse_vocab, PARAGRAPH_WINDOW_SIZE, SUMMARY_WINDOW_SIZE, padding_index)
    perplexity = np.exp(loss)
    print("model test perplexity: {}. model test accuracy: {}".format(perplexity, accuracy))
    end_time = time.time()
    print("The model took:", end_time-start_time,"seconds to train")


if __name__ == '__main__':
    main()


