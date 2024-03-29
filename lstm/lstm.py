import numpy as np
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, LSTM, concatenate, Dense
from attention import AttentionLayer

class LSTM_Seq2Seq(tf.keras.Model):
    def __init__(self, vocab_size, paragraph_window_size, summary_window_size):
        super(LSTM_Seq2Seq, self).__init__()
        self.paragraph_window_size = paragraph_window_size
        self.summary_window_size = summary_window_size
        self.vocab_size = vocab_size
        self.batch_size = 100
        self.embedding_size = 15
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.009)

        # self.paragraph_embedding = tf.Variable(tf.random.truncated_normal(shape=[self.paragraph_window_size,self.embedding_size],stddev=0.01,dtype=tf.float32))
        self.paragraph_embedding1 = Embedding(self.vocab_size, self.embedding_size, input_length = self.paragraph_window_size)
        self.summary_embedding1 = Embedding(self.vocab_size, self.embedding_size, input_length = self.summary_window_size)
        self.encoder = LSTM(80, activation ='relu', return_state = True, return_sequences = True)
        self.encoder1 = LSTM(80, activation ='relu', return_state = True, return_sequences = True)
        self.encoder2 = LSTM(80, activation ='relu', return_state = True, return_sequences = True)

        # self.inputs2 = Input(shape=(summary_window_size,))
        # self.summary_embedding = tf.Variable(tf.random.truncated_normal(shape=[self.summary_window_size,self.embedding_size],stddev=0.01,dtype=tf.float32))
        self.decoder = LSTM(80, activation ='relu', return_state = True, return_sequences = True)

        self.attn_layer = AttentionLayer(name='attention_layer')

        self.outputs = Dense(vocab_size, activation='softmax')
    
    @tf.function
    def call(self, encoder_input, decoder_input):
        """
        :param encoder_input: batched ids corresponding to french sentences
        :param decoder_input: batched ids corresponding to english sentences
        :return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
        """
        embedding_paragraph = self.paragraph_embedding1(encoder_input)
        embedding_summary = self.summary_embedding1(decoder_input)
        encoder_outputs, state_h, state_c = self.encoder(embedding_paragraph)
        encoder_outputs1, state_h1, state_c1 = self.encoder1(encoder_outputs)
        encoder_outputs2, state_h2, state_c2 = self.encoder2(encoder_outputs1)
        encoder_states = [state_h2, state_c2]
        decoder_out= self.decoder(embedding_summary, initial_state=encoder_states)
        attn_out, attn_states = self.attn_layer([encoder_outputs2, decoder_out[0]])
        decoder_concat_input = concatenate([decoder_out[0], attn_out], axis=-1)
        dense_out = self.outputs(decoder_concat_input)
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
        # print(labels.shape)
        # print(prbs.shape)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels,prbs, from_logits = False)
        # print(prbs[1])
        # print(loss)
        loss=tf.reduce_mean(loss*mask)
        # print(loss)
        # exit()
        # print(labels[1])
        return loss
    def produce_sentence(self, ori_paragraph, summary, prbs, reverse_vocab, sen_len):
        # print(prbs.shape)
        # exit()
        # decoded_symbols = np.random.randint(0, 717514, size=32)
        # decoded_symbols = np.argmax(prbs, axis=1)
        # arr.argsort()[-3:][::-1]
        indices =[]
        # print(prbs.shape)
        # exit()
        prbs = np.array(prbs)
        for prb in prbs:
            indices.append(prb.argsort()[-10:][::-1])
        decoded_symbols = [0]*32
        for i in range(len(indices)):
            decoded_symbols[i] = int(np.random.choice(indices[i],1))
        # print(decoded_symbols)
        # exit()
        decoded_sentence = [ reverse_vocab[x] for x in decoded_symbols ]
        decoded_sentence = " ".join(decoded_sentence)
        ori_summary = " ".join([ reverse_vocab[x] for x in summary ])
        print("original paragraph\n", ori_paragraph)
        print("summary sentence\n", ori_summary)
        print("decoded sentence\n", decoded_sentence)
