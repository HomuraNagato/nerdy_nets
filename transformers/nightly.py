
import sys
import time
sys.path.append('../gpt-2/src/')
import callable_gpt2 as gpt2
import model
import encoder
import sample

import tensorflow as tf
import torch 
import numpy as np
import pandas as pd

from transformers import *

#tf.keras.backend.set_floatx('float64')
'''
error in tensorflow dtype init if tensorflow version = 2.0.0beta1
pip uninstall tensorflow 
pip install tensorflow==2.0.0
'''

train_window_size = 400
test_window_size = 100

def pad_corpus(train_sentences, test_sentences, special_tokens, train_window_size, test_window_size):
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


def train(model, file_path, special_tokens):

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    
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

        ''''
        words = []
        for train_word, test_word in zip(train_words, test_words):
            text = special_tokens['bos_token'] + ' ' + train_word + ' ' +  \
                   special_tokens['cls_token'] + ' ' + test_word + ' ' + special_tokens['eos_token']
            words.append(text)
        '''
        #words = np.array(words)
        #print("words as numpy\n", words, "\n", words.shape)
        #print("concatted words", words)

        words = pad_corpus(train_words, test_words, special_tokens, train_window_size, test_window_size)
        
        words = [ tokenizer.encode(text) for text in words ]
        words = tf.convert_to_tensor(words, dtype=tf.float).unsqueeze(0)
        #words = tf.reshape(words, [batch_size, -1])
        print("words as tensor\n", words, "\n", words.shape)

        
        outputs  = model(words) # last hidden states
        logits = outputs[0]
        print("probs\n", logits)
        exit(0)


if __name__ == '__main__':
    '''
    # call gpt2-tf1 directly
    sentences = ['a crying wish', 'When the world stopped turning']
    model_name, models_dir = '124M', '../gpt-2/models/'
    seed = 13
    n_ctx = 1024
    nsamples, batch_size, length = 1, 1, n_ctx // 2
    temperature, top_k, top_p = 1, 0, 1
    
    gpt_text = gpt2.interact_model(model_name=model_name, seed=seed, nsamples=nsamples,
                        batch_size=batch_size, length=length, temperature=temperature,
                        top_k=top_k, top_p=top_p, models_dir=models_dir, sentences=sentences)

    for raw, out in zip(sentences, gpt_text):
        print("gpt_text in\n", raw, "\ngpt_text out\n", out)
    '''

    # convert to pretrained to use transformers
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    model = TFGPT2LMHeadModel.from_pretrained('gpt2', dtype=tf.int32)
    print("loaded model", model)
    #add the special tokens into the vocabulary
    special_tokens = {'bos_token': '<bos>', 'cls_token': '<cls>', 'eos_token': '<eos>',
                      'pad_token': '<pad>'}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    #num_train_optimization_steps = len(self.train_dataset) * options['num_epochs'] // options['train_batchsize']

    #train(model, '../data/tldr_train80.jsonl', special_tokens)
    train(model, '../data/tldr-training-data.jsonl', special_tokens)
    
