
import sys
import time
#sys.path.append('../gpt-2/src/')
#import callable_gpt2 as gpt2
#import model
#import encoder
#import sample

import torch
import numpy as np
import pandas as pd

from transformers import *

#tf.keras.backend.set_floatx('float64')
'''
error in tensorflow dtype init if tensorflow version = 2.0.0beta1
pip uninstall tensorflow 
pip install tensorflow==2.0.0

unhashable
tf.compat.v1.enable_tensor_equality()
'''

train_window_size = 400
test_window_size = 100

def pad_corpus_ori(sentences, pad_token, window_size):
    pad_sentences = []
    i = 0
    for text in sentences:
        text = text.split()
        extended_text = text[:window_size-1] + [pad_token] * (window_size - len(text)-1)
        
        pad_sentences.append(extended_text)
        i += 1

    return pad_sentences


def pad_corpus(sentences, start_token, end_token, pad_token, window_size):
    pad_sentences = []
    i = 0
    for text in sentences:
        text = text.split()
        extended_text = text[:window_size-1] + [pad_token] * (window_size - len(text)-1) + [end_token]
        
        pad_sentences.append(extended_text)
        i += 1

    return pad_sentences

def pad_corpus_simul(train_sentences, test_sentences, special_tokens, train_window_size, test_window_size):
    pad_sentences = []
    i = 0
    for train_text, test_text in zip(train_sentences, test_sentences):
        train_text, test_text = train_text.split(), test_text.split()
        extended_text = [special_tokens['bos_token']] + train_text[:train_window_size-1] + [special_tokens['pad_token']] * (train_window_size - len(train_text)-1) + [special_tokens['cls_token']] + \
                        test_text[:test_window_size-1] + [special_tokens['pad_token']] * (test_window_size - len(test_text)-1) + [special_tokens['eos_token']]
                                                                                          
        pad_sentences.append(extended_text)
        i += 1
    #print("length of post-padded text", len(pad_sentences[0]))
    return pad_sentences



def train(model, tokenizer, file_path, special_tokens):

    optimizer = AdamW(model.parameters())
    
    batch_size = 10
    reader = pd.read_json(file_path, precise_float=True, dtype=False, lines=True, chunksize=batch_size)
    train_steps = 0
    step_start_time = time.time()
    training_time = [step_start_time]
    
    for section in reader:
        #print("section['normalizedBody']: \n", section['normalizedBody'].tolist(), len(section['normalizedBody'].tolist()))
        train_steps += 1
        
        # get paragraph and summary
        train_words = section['normalizedBody'].tolist()
        test_words = section['summary'].tolist()

        #words = np.array(words)
        #print("words as numpy\n", words, "\n", words.shape)
        #print("concatted words", words)

        words = pad_corpus_simul(train_words, test_words, special_tokens, train_window_size, test_window_size)
        #test_words = pad_corpus(test_words, special_tokens['bos_token'], special_tokens['eos_token'], special_tokens['pad_token'], test_window_size)
        #train_words = pad_corpus_ori(train_words, special_tokens['pad_token'], train_window_size)
        #test_words = pad_corpus_ori(test_words, special_tokens['pad_token'], test_window_size)
        '''
        train_words = pad_corpus_ori(train_words, '', train_window_size)
        words = []
        for train, test in zip(train_words, test_words):
            words.append(train + test)
        '''
        words = [ tokenizer.encode(text) for text in words ]
        words = torch.tensor(words)
        test_words = pad_corpus_ori(test_words, special_tokens['pad_token'], test_window_size)
        test_words = [ tokenizer.encode(text) for text in test_words ]
        test_words = torch.tensor(test_words)
        #words = tf.reshape(words, [batch_size, -1])
        print("words as tensor\n", words, "\n", words.shape)

        
        outputs  = model(words, labels=words) # last hidden states
        loss, logits = outputs[0], outputs[1]
        #logit = logit.numpy()
        predicted_index = torch.argmax(logits[0, -1, :])
        #loss = torch.keras.losses.sparse_categorical_crossentropy(words, logits, from_logits=True)
        #print("loss\n", loss, "\n", loss.shape)
        print("probs first\n", logits.shape)
        print("encoded prediction", predicted_index)
        print("decoded sentence first\n", tokenizer.decode(predicted_index))
        exit(0)


if __name__ == '__main__':

    # convert to pretrained to use transformers
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    #print("loaded model", model)
    #add the special tokens into the vocabulary
    special_tokens = {'bos_token': '<bos>', 'cls_token': '<cls>', 'eos_token': '<eos>',
                      'pad_token': '<pad>'}
    # tf transformers module not currently implemented resize_token_embeddings function :(
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    #num_train_optimization_steps = len(self.train_dataset) * options['num_epochs'] // options['train_batchsize']

    #train(model, '../data/tldr_train80.jsonl', special_tokens)
    train(model, tokenizer, '../data/tldr-training-data.jsonl', special_tokens)
    
