import numpy as np
import tensorflow as tf
import transformer_funcs as transformer

class Transformer_Seq2Seq(tf.keras.Model):
    def __init__(self, french_window_size, french_vocab_size, english_window_size, english_vocab_size):

        ######vvv DO NOT CHANGE vvv##################
        super(Transformer_Seq2Seq, self).__init__()

        self.french_vocab_size = french_vocab_size # The size of the french vocab
        self.english_vocab_size = english_vocab_size # The size of the english vocab

        self.french_window_size = french_window_size # The french window size
        self.english_window_size = english_window_size # The english window size
        ######^^^ DO NOT CHANGE ^^^##################


        # TODO:
        # 1) Define any hyperparameters
        self.encoder_size = 15
                
        # 2) Define embeddings, encoder, decoder, and feed forward layers
        
        # Define batch size and optimizer/learning rate
        self.batch_size = 100
        self.embedding_size = 15

        # Define english and french embedding layers:
        self.french_embedding_layer = tf.keras.layers.Embedding(self.french_vocab_size, self.embedding_size, input_length=self.french_window_size)
        self.english_embedding_layer = tf.keras.layers.Embedding(self.english_vocab_size, self.embedding_size, input_length=self.english_window_size)

        # Create positional encoder layers

        self.french_positional_layer = transformer.Position_Encoding_Layer(self.french_window_size, self.embedding_size)
        self.english_positional_layer = transformer.Position_Encoding_Layer(self.english_window_size, self.embedding_size)

        # Define encoder and decoder layers:
        self.encoder_lstm = transformer.Transformer_Block(self.embedding_size, is_decoder=False, multi_headed=True)
        self.decoder_lstm = transformer.Transformer_Block(self.embedding_size, is_decoder=True, multi_headed=True)

        # Define dense layer(s)
        self.dense = tf.keras.layers.Dense(self.french_vocab_size, activation='softmax')


    @tf.function
    def call(self, encoder_input, decoder_input):
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

        french_embedding = self.french_embedding_layer(encoder_input)
        french_embedding = self.french_positional_layer(french_embedding)
        encoder_output = self.encoder_lstm(french_embedding)

        english_embedding = self.english_embedding_layer(decoder_input)
        english_embedding = self.english_positional_layer(english_embedding)
        decoder_output = self.decoder_lstm(english_embedding, encoder_output)

        dense = self.dense(decoder_output)
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

