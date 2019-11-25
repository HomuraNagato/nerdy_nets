# nerdy_nets

## Model

Draft 1 
Based heavily on Seq2Seq language translation assignment. A transformer with an encoder encodes the input
paragraph, then is run through a decoder. Loss is measured by cross-entropy loss comparing the probabilities
output from the decoder and the reference summary.

Expected Final Model
..

## Metadata

 - unfilted vocab size: 3053570 
 - filtered vocab size: 717515 (0.765 reduction)
 - longest paragraph:   1015
 - longest summary:     397
 - num examples:        3,084,500

## Preprocess

The data is gathered from https://tldr.webis.de. [about the data]

Data is first batched using pandas then preprocessed in the training module. Preprocess involves removing
punctuation, lowercasing, padding by window size (retain text from 0:window_size should it be longer), and
converted to an ID by a vocab lookup. The vocabulary is pre-computed and stored on disk to reduce overhead.
Vocab preprocessed as in training, retain only words that occur more than once.

## TODO

 - ROUGE test
 - query selector to reduce paragraph size
 - decoder_transformer shape error. Should be able to have different window sizes, but unable to currently.
   Work around is to have the same window sizes.
 - GPT-2 pre-trained model to initialize weights?
 - memory shortage! GCP tesla K80 has 11GB of memory. Vocab size of ~70,000 makes allocating memory for
   dense layer memory intensive. Work around, reduce batch size from 100 to 10.
    - Look into sparse memory computations to reduce computation
    - Look into reducing vocab size
    - Look into tensorflow paradigms that may allow reducing memory requirements

As noted above, our greatest challenge is memory. Our initial plan was to have a maximum window size for
paragraph and summary (1024 and 400 respectively) and as large a vocab as possible, but that's just not
feasible in any memory setting. 
 

## Draft 1

Ran model on GCP for 10000 examples, Per batch time average: . Total train time: .
perplexity during training: .