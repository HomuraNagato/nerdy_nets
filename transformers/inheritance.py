import os
import numpy as np
import tensorflow as tf
import numpy as np
import sys
import time

from preprocess import *
from transformer_model import Transformer_Seq2Seq
sys.path.append('../gpt-2/src/')
import model
import encoder
import sample

#tf.debugging.set_log_device_placement(True)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



@tf.function
def main():	

    start_time = time.time()

    hparams = model.default_hparams()
    enc = encoder.get_encoder('124M', '../gpt-2/models')

    
    '''
    #args.memory_saving_gradients = True memory saving?
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    '''
    sentences = ['a crying wish', 'When the world stopped turning']
    #context = tf.compat.v1.placeholder(tf.int32, [hparams['batch_size'], len(sentences[0])])
    #context = tf.compat.v1.placeholder(tf.int32, [hparams['batch_size'], None])
    #sentences = tf.convert_to_tensor(sentences)
    print("original sentences\n", sentences)
    sentences = [ enc.encode(s) for s in sentences ]
    print("encoded sentences\n", sentences)
    '''
    sentences = [ enc.decode(s) for s in sentences ]
    print("decoded sentences\n", sentences)
    '''
    start_token=enc.encoder['<|endoftext|>']

    first_sentence = np.array(sentences[0] * hparams['batch_size'])
    first_sentence = np.append(first_sentence, start_token)
    print("first_sentence", first_sentence)
    # build sentence tokens
    context = first_sentence.reshape(hparams['batch_size'], len(first_sentence))
    context = tf.convert_to_tensor(context, dtype=tf.int32)
    # initialize a model
    #results = model.model(hparams=hparams, X=context)
    #print("gpt-2 results", results)
    #logits = results['logits']

    # run sentence through model
    seed = 13
    nsamples, batch_size, length = 1, 1, hparams['n_ctx'] // 2
    temperature, top_k, top_p = 1, 0, 1
    tf.compat.v1.set_random_seed(seed)
    gpt_tokens = sample.sample_sequence(hparams=hparams, length=length, start_token=None,
                                        context=context,
                                    batch_size=batch_size, temperature=temperature,
                                    top_k=top_k, top_p=top_p)
    # gpt-2 logits {'present': <tf.Tensor: id=3351, shape=(128, 12, 2, 12, 3, 64), dtype=float32,
    # 'logits': <tf.Tensor: id=3390, shape=(128, 3, 50257)
    print("gpt_tokens\n", gpt_tokens)
    return gpt_tokens



if __name__ == '__main__':
    gpt_tokens = main()
    gpt_text = enc.decode(gpt_tokens)
    print("gpt generated text", gpt_text)
