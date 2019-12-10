import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, LSTM, concatenate, Dense

class LSTM_Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_size, paragraph_window_size, summary_window_size):
        super(LSTM_Seq2Seq, self).__init__()
        self.paragraph_window_size = paragraph_window_size
        self.summary_window_size = summary_window_size
        self.vocab_size = vocab_size
        self.batch_size = 100
        self.embedding_size = 15
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)

        # self.inputs1 = Input(shape=(paragraph_window_size,))
        self.am1 = Embedding(vocab_size, 128)
        self.am2 = LSTM(128)

        # self.inputs2 = Input(shape=(summary_window_size,))
        self.sm1 = Embedding(vocab_size, 128)
        self.sm2 = LSTM(128)

        self.outputs = Dense(vocab_size, activation='softmax')
    
    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
        embedding_paragraph = self.am1(encoder_input)
        embedding_summary = self.sm1(decoder_input)
        out = self.am2(embedding_paragraph)
        out1= self.sm2(embedding_summary)
        out2 = concatenate([out, out1])
        dense_out = self.outputs(out2)
        return dense_out

    def accuracy_function(self, prbs, labels, mask):
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
        print(labels.shape)
        print(prbs.shape)
        loss=tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(labels,prbs)*mask)
        return loss
    def produce_sentence(self, ori_paragraph, summary, prbs, reverse_vocab, sen_len):
        decoded_symbols = np.argmax(prbs, axis=1)
        decoded_sentence = [ reverse_vocab[x] for x in decoded_symbols ]
        decoded_sentence = " ".join(decoded_sentence)
        ori_summary = " ".join([ reverse_vocab[x] for x in summary ])
        print("original paragraph\n", ori_paragraph)
        print("summary sentence\n", ori_summary)
        print("decoded sentence\n", decoded_sentence)
