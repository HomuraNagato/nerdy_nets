import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_size, window_size, summary_window_size):

        ######vvv DO NOT CHANGE vvv##################
        super(Transformer_Seq2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.window_size = window_size
        self.summary_window_size = summary_window_size

        # 1) hyperparameters
        self.encoder_size = 15
                
        # Define batch size and optimizer/learning rate
        self.batch_size = 100
        self.embedding_size = 15

        # Define english and french embedding layers:
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.window_size)
        self.summary_embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.summary_window_size)

        # Create positional encoder layers
        self.positional_layer = transformer.Position_Encoding_Layer(self.window_size, self.embedding_size)
        self.summary_positional_layer = transformer.Position_Encoding_Layer(self.summary_window_size, self.embedding_size)

        # Define encoder and decoder layers:
        self.encoder_transformer = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=True)
        self.decoder_transformer = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed=True)

        # Define dense layer(s)
        self.dense = tf.keras.layers.Dense(self.vocab_size, activation='softmax')


    #@tf.function
    def call(self, encoder_input, summary_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
	
        # TODO:
        #1) Add the positional embeddings to french sentence embeddings
        #2) Pass the french sentence embeddings to the encoder
        #3) Add positional embeddings to the english sentence embeddings
        #4) Pass the english embeddings and output of your encoder, to the decoder
        #3) Apply dense layer(s) to the decoder out to generate probabilities

        '''
        french_embedding = self.french_embedding_layer(encoder_input)
        french_embedding = self.french_positional_layer(french_embedding)
        encoder_output = self.encoder_lstm(french_embedding)

        english_embedding = self.english_embedding_layer(decoder_input)
        english_embedding = self.english_positional_layer(english_embedding)
        decoder_output = self.decoder_lstm(english_embedding, encoder_output)

        dense = self.dense(decoder_output)
        return dense
        '''

        #print("transformer input", encoder_input.shape)
        embedding = self.embedding_layer(encoder_input)
        embedding = self.positional_layer(embedding)
        #print("embedding", embedding.shape)
        encoder_output = self.encoder_transformer(embedding)
        #print("encoder_output", encoder_output.shape)

        summary_embedding = self.summary_embedding_layer(summary_input)
        summary_embedding = self.summary_positional_layer(summary_embedding)
        #print("summary embedding", summary_embedding.shape)
        
        decoder_output = self.decoder_transformer(summary_embedding, encoder_output)
        #print("decoder_output", decoder_output.shape)

        dense = self.dense(decoder_output)
        #print("returning dense", dense.shape)
        return dense

    def accuracy_function(self, prbs, labels, mask):
        """
        DO NOT CHANGE

        Computes the batch accuracy

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: scalar tensor of accuracy of the batch between 0 and 1
        """

        #decoded_symbols = tf.argmax(input=prbs, axis=2)
        decoded_symbols = tf.argmax(input=prbs, axis=2)
        accuracy = tf.reduce_mean(tf.boolean_mask(tf.cast(tf.equal(decoded_symbols, labels), dtype=tf.float32),mask))
        return accuracy


    def loss_function(self, prbs, labels, mask):
        """
        Calculates the model cross-entropy loss after one forward pass

        :param prbs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
        :param labels:  integer tensor, word prediction labels [batch_size x window_size]
        :param mask:  tensor that acts as a padding mask [batch_size x window_size]
        :return: the loss of the model as a tensor
        """

        # Note: you can reuse this from rnn_model.

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, prbs, from_logits=False)
        loss = loss * mask
        loss = tf.reduce_mean(loss)
        return loss

    def produce_sentence(self, ori_paragraph, summary, prbs, reverse_vocab, sen_len):

        decoded_symbols = np.argmax(prbs, axis=1)
        decoded_sentence = [ reverse_vocab[x] for x in decoded_symbols ]
        decoded_sentence = " ".join(decoded_sentence)
        #ori_paragraph = " ".join([ reverse_vocab[x] for x in paragraph ])
        ori_summary = " ".join([ reverse_vocab[x] for x in summary ])
        print("original paragraph\n", ori_paragraph)
        print("summary sentence\n", ori_summary)
        print("decoded sentence\n", decoded_sentence)
