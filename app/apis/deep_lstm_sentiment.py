import numpy as np
import tensorflow as tf
from random import shuffle
import os
from datetime import datetime

class DeepLSTMSentiment():
	def __init__(self, sess, gpu_percent, n_hidden, max_step, num_layers):
		# config = tf.ConfigProto()
		# config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
		# self.sess = tf.Session(config = config)
		self.sess = sess
		self.n_hidden = n_hidden
		self.max_step = max_step
		self.num_layers = num_layers
		self.build_model()

	def build_model(self):
		with tf.variable_scope('deep_lstm'):
			self.init_placeholder()
			self.build_graph()
			self.loss_optimizer()
		self.sess.run(tf.global_variables_initializer())
		# summary_writer = tf.train.SummaryWriter('/home/tittit/python/web_mining2/logs', graph = tf.get_default_graph())

	def init_placeholder(self):
		### placeholders
		self.x = tf.placeholder(tf.float32, [None, self.max_step, 100])
		self.y = tf.placeholder(tf.float32, [None, 2])
		self.sequence_length = tf.placeholder(tf.int32, [None])
		self.lr = tf.placeholder(tf.float32, None)
		self.dropout = tf.placeholder(tf.float32, None)

	def build_graph(self):
		### softmax params
		self.w = tf.Variable(tf.truncated_normal([self.n_hidden, 2], dtype = tf.float32), name = 'w', dtype = tf.float32)
		self.b = tf.Variable(tf.truncated_normal([1, 2], dtype = tf.float32), name = 'b', dtype = tf.float32)
		### LSTM layer
		self.output = self.x
		self.lstm_cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_hidden), output_keep_prob = self.dropout) for i in range(self.num_layers)]
		for i in range(self.num_layers):
			self.output, _ = tf.nn.dynamic_rnn(cell = self.lstm_cells[i], inputs = self.output, scope = 'lstm_' + str(i), dtype = tf.float32)
		# self.lstm_cell = tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(num_units = self.n_hidden), output_keep_prob = self.dropout)
		# self.output, _ = tf.nn.dynamic_rnn(cell = self.lstm_cell, inputs = self.x, scope = 'lstm_layer', dtype = tf.float32)
		self.current_batch_size = tf.shape(self.output)[0]
		self.index = tf.range(0, self.current_batch_size)*self.max_step + (self.sequence_length - 1)
		self.output_last = tf.gather(tf.reshape(self.output, [-1, self.n_hidden]), self.index)
		self.pred = tf.nn.softmax(tf.matmul(self.output_last, self.w) + self.b)

	def loss_optimizer(self):
		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), axis = 1), axis = 0)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def shuffle(self, x, y, seqlen):
		z = list(zip(x, y, seqlen))
		shuffle(z)
		x, y, seqlen = zip(*z)
		return x, y, seqlen

	def evaluate(self, x_test, y_test, seqlen_test, dropout):
		pred = self.sess.run(self.pred, feed_dict = {self.x: x_test, self.sequence_length: seqlen_test, self.dropout: dropout})
		# return (np.sum(np.equal(np.argmax(pred, axis = 1), np.argmax(y_test, axis = 1)))/len(y_test))*100
		return np.sum(np.equal(np.argmax(pred, axis = 1), np.argmax(y_test, axis = 1)))

	def train_new_model(self, x_train, y_train, seqlen_train, x_valid, y_valid, seqlen_valid, lr, batch_size, num_epochs, save_path):
		accuracy = 0
		saver = tf.train.Saver(max_to_keep = 1000)
		for epoch in range(num_epochs):
			### shuffle all training data
			x, y, seqlen = self.shuffle(x_train, y_train, seqlen_train)
			sum_loss = 0
			start_time = datetime.now()
			while (len(x) != 0):
				### get training batch
				upperbound = min(batch_size, len(x))
				x_batch = x[:upperbound]
				y_batch = y[:upperbound]
				seqlen_batch = seqlen[:upperbound]
				x = x[upperbound:]
				y = y[upperbound:]
				seqlen = seqlen[upperbound:]
				_loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.x: x_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr})
				sum_loss += _loss
			### evaluate on valid set
			f1_score = self.evaluate(x_valid, y_valid, seqlen_valid)
			print('epoch: ', epoch, ' loss: ', sum_loss/(int((len(x_train)-1)/batch_size) + 1), ' f1_score: ', f1_score, ' time: ', str(datetime.now() - start_time))
			if (f1_score > accuracy):
				accuracy = f1_score
				saver.save(self.sess, save_path + '/model.ckpt')

	def train_new_partial_model(self, x_train, y_train, seqlen_train, lr, batch_size, dropout):
		x, y, seqlen = self.shuffle(x_train, y_train, seqlen_train)
		sum_loss = 0
		while (len(x) != 0):
			### get training batch
			upperbound = min(batch_size, len(x))
			x_batch = x[:upperbound]
			y_batch = y[:upperbound]
			seqlen_batch = seqlen[:upperbound]
			x = x[upperbound:]
			y = y[upperbound:]
			seqlen = seqlen[upperbound:]
			_loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.x: x_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr, self.dropout: dropout})
			sum_loss += _loss
		return sum_loss

	def load_trained_model(self, load_path):
		assert os.path.exists(load_path), load_path + ' is not exist!!!'
		saver = tf.train.Saver(self.sess, load_path)

	def save_model(self, save_path):
		saver = tf.train.Saver(max_to_keep = 1000)
		saver.save(self.sess, save_path)

	def load_model(self, load_path):
		saver = tf.train.Saver(max_to_keep = 1000)
		saver.restore(self.sess, load_path)

	def transform(self, text, word2vec, tokenizer, normalizer):
		### transform text to vector
		text = normalizer.transform(text)
		tokens = tokenizer.transform(text)
		senvec = []
		seqlen = 0
		for token in tokens:
			senvec.append(word2vec[token])
		senvec = senvec[:min(len(senvec), self.max_step)]
		seqlen = len(senvec)
		while (len(senvec) < self.max_step):
			senvec.append(np.zeros(100))
		### predict
		pred = self.sess.run(self.pred, feed_dict = {self.x: [senvec], self.sequence_length: [seqlen], self.dropout: 1.0})
		print(pred)
		res = {'positive': pred[0][0], 'negative': pred[0][1]}
		return res
