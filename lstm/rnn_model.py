import numpy as np
import tensorflow as tf

class RNN_Seq2Seq(tf.keras.Model):
	def __init__(self, vocab_size, paragraph_window_size, summary_window_size):

    ######vvv DO NOT CHANGE vvvv##############
		super(RNN_Seq2Seq, self).__init__()
		self.paragraph_window_size = paragraph_window_size 
		self.summary_window_size = summary_window_size 

		self.vocab_size = vocab_size 
		
		# Define batch size and optimizer/learning rate
		self.batch_size = 100
		self.embedding_size = 15
		self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
	
	
		self.paragraph_embedding = tf.Variable(tf.random.truncated_normal([ self.paragraph_window_size, self.embedding_size],stddev=0.01,dtype=tf.float32))
		self.summary_embedding = tf.Variable(tf.random.truncated_normal([ self.summary_window_size, self.embedding_size],stddev=0.01,dtype=tf.float32))
        
        # Define encoder and decoder layers:
		self.encoder = tf.keras.layers.LSTM(80)
		self.decoder = tf.keras.layers.LSTM(80 , return_sequences = True)
		self.dense_layer = tf.keras.layers.Dense(self.vocab_size, activation='softmax')
		
	@tf.function
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
	
		# TODO:
		#1) Pass your french sentence embeddings to your encoder 
		#2) Pass your english sentence embeddings, and final state of your encoder, to your decoder
		#3) Apply dense layer(s) to the decoder out to generate probabilities
		embedding_paragraph = tf.nn.embedding_lookup(self.paragraph_embedding, encoder_input)
		embedding_summary = tf.nn.embedding_lookup(self.summary_embedding, decoder_input)
		out = self.encoder(embedding_paragraph)
		out1= self.decoder(embedding_summary, out[0])
		dense_out = self.dense_layer(out1[0])
		return dense_out

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
		loss=tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels,prbs)*mask)
		return loss

