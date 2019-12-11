
import sys
import time


import torch
import tensorflow as tf
import numpy as np
import pandas as pd

from transformers import *


def sample_sentence(model, tokenizer, paragraph, summary):

    #tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    #model = GPT2LMHeadModel.from_pretrained('gpt2')

    
    initialized = tokenizer.encode(summary)
    labels = tokenizer.encode(paragraph)
    
    paragraph_length = len(labels)
    vocab_size = len(tokenizer.decoder)
    
    context = torch.tensor([initialized])
    labels = torch.tensor([labels])

    past = None
    lm_loss = 0
    generated = []
    
    for i in range(paragraph_length):

        output, past = model(context, past=past)
        #print("output being argmaxed", output.shape, "\t", output[0,:])
        token = torch.argmax(output[0, :])
        if token.numpy() > vocab_size:
            # some words not in encoding dict?? replace with first
            # word of input
            token = context[0,0]
        #print("token taken from argmax", token, token.tolist())

        generated += [token.tolist()]
        context = token.unsqueeze(0)

    sequence = tokenizer.decode(generated)
    
    with torch.no_grad():
        lm_loss, output, past = model(torch.tensor(generated).clone().detach(), labels=labels)
        lm_loss = lm_loss.tolist()
    
    #print("input summary\n", summary, "\n")
    #print("input paragraph\n", paragraph, "\n")
    #print("generated paragraph\n", sequence)
    #print("lm loss", lm_loss)
    #print("loss", tf.reduce_mean(loss))

    return lm_loss, sequence



if __name__ == '__main__':
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    torch_model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = TFGPT2Model.from_pretrained('gpt2', dtype=tf.int32)
    #print("loaded model", model)
    #add the special tokens into the vocabulary
    special_tokens = {'bos_token': '<bos>', 'cls_token': '<cls>', 'eos_token': '<eos>',
                      'pad_token': '<pad>'}
    # tf transformers module not currently implemented resize_token_embeddings function :(
    #tokenizer.add_special_tokens(special_tokens)
    #model.resize_token_embeddings(len(tokenizer))

    file_path = '../data/tldr-training-data.jsonl'
    batch_size = 10
    reader = pd.read_json(file_path, precise_float=True, dtype=False, lines=True, chunksize=batch_size)
    
    for section in reader:
    
        train_words = section['normalizedBody'].tolist()
        test_words = section['summary'].tolist()

        first_paragraph = train_words[0]
        first_summary = test_words[0]
        
        sample_sentence(torch_model, tokenizer, first_paragraph, first_summary)
        break
