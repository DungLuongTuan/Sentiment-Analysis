import numpy as np
import tensorflow as tf
from random import shuffle
import os
from datetime import datetime
"""
	filter = [(filter_height, num_filter), (filter_height, num_filter), ...]
	hidden_units = [1024, 1024, 512, 512]
"""

class CNN_LSTM_Sentiment():
	def __init__(self, sess, max_step, inp_vec_length, num_class, filter_config, lstm_n_hidden, lstm_num_layers, cnn_embed_size, gpu_percent):
		self.max_step = max_step
		self.inp_vec_length = inp_vec_length
		self.num_class = num_class
		self.filter_config = filter_config
		self.lstm_n_hidden = lstm_n_hidden
		self.lstm_num_layers = lstm_num_layers
		self.cnn_embed_size = cnn_embed_size
		# config = tf.ConfigProto()
		# config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
		# self.sess = tf.Session(config = config)
		# self.sess = tf.InteractiveSession()
		self.sess = sess
		self.build_model()

	def build_model(self):
		with tf.variable_scope('cnn_lstm'):
			self.init_placeholder()
			self.build_graph()
		self.sess.run(tf.global_variables_initializer())

	def init_placeholder(self):
		self.x = tf.placeholder(tf.float32, [None, self.max_step, self.inp_vec_length])
		self.y = tf.placeholder(tf.float32, [None, self.num_class])
		self.lr = tf.placeholder(tf.float32, None)
		self.sequence_length = tf.placeholder(tf.int32, [None])
		self.batch_size = tf.shape(self.x)[0]

	def build_graph(self):
		### convolution parameters
		x_r = tf.reshape(self.x, [self.batch_size, self.max_step, self.inp_vec_length, 1])
		filters = []
		filter_heights = []
		output_dim_cnn = 0
		for config in self.filter_config:
			filters.append(tf.Variable(tf.truncated_normal(shape = [config[0], self.inp_vec_length, 1, config[1]]), name = 'filter_' + str(config[0])))
			filter_heights.append(config[0])
			output_dim_cnn += config[1]
		### build CNN layers
		conv_feature = tf.nn.conv2d(input = x_r, filter = filters[0], strides = [1, 1, 1, 1], padding = 'VALID')
		conv_feature_pool = tf.nn.max_pool(conv_feature, ksize = [1, self.max_step - filter_heights[0] + 1, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID')
		conv_feature_pool_r = tf.reshape(conv_feature_pool, [self.batch_size, -1])
		for index, _filter in enumerate(filters[1:]):
			conv = tf.nn.conv2d(input = x_r, filter = _filter, strides = [1, 1, 1, 1], padding = 'VALID')
			conv_pool = tf.nn.max_pool(conv, ksize = [1, self.max_step - filter_heights[index+1] + 1, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID')
			conv_pool_r = tf.reshape(conv_pool, [self.batch_size, -1])
			conv_feature_pool_r = tf.concat([conv_feature_pool_r, conv_pool_r], axis = -1)
		self.cnn_embed = conv_feature_pool_r
		with tf.name_scope('cnn_embed'):
			w = tf.Variable(tf.truncated_normal(shape = [output_dim_cnn, self.cnn_embed_size]), name = 'weight_input_layer')
			b = tf.Variable(tf.truncated_normal(shape = [1, self.cnn_embed_size]), name = 'bias_input_layer')
			self.cnn_embed = tf.tanh(tf.matmul(conv_feature_pool_r, w) + b)
		# build LSTM layers
		output = self.x
		lstm_cells = [tf.contrib.rnn.LSTMCell(num_units = self.lstm_n_hidden) for i in range(self.lstm_num_layers)]
		for i in range(self.lstm_num_layers):
			self.output, _ = tf.nn.dynamic_rnn(cell = lstm_cells[i], inputs = output, scope = 'lstm_' + str(i), dtype = tf.float32)
		index = tf.range(0, self.batch_size)*self.max_step + (self.sequence_length - 1)
		self.lstm_embed = tf.gather(tf.reshape(output, [-1, self.lstm_n_hidden]), index)
		# output layer
		with tf.name_scope('output_layer'):
			w = tf.Variable(tf.truncated_normal(shape = [self.lstm_n_hidden + self.cnn_embed_size, self.num_class]), name = 'weight_output_layer')
		#	w = tf.Variable(tf.truncated_normal(shape = [self.lstm_n_hidden + output_dim_cnn, self.num_class]), name = 'weight_output_layer')
			b = tf.Variable(tf.truncated_normal(shape = [1, self.num_class]), name = 'bias_output_layer')
			self.cnn_lstm_embed = tf.concat([self.cnn_embed, self.lstm_embed], axis = -1)
			self.pred = tf.nn.softmax(tf.matmul(self.cnn_lstm_embed, w) + b)
		# loss + optimazer
		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), axis = 1), axis = 0)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def shuffle(self, x, y, seqlen):
		z = list(zip(x, y, seqlen))
		shuffle(z)
		x, y, seqlen = zip(*z)
		return x, y, seqlen

	def evaluate(self, x_test, y_test, seqlen_test):
		pred = self.sess.run(self.pred, feed_dict = {self.x: x_test, self.sequence_length: seqlen_test})
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

	def train_new_partial_model(self, x_train, y_train, seqlen_train, lr, batch_size):
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
			_loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.x: x_batch, self.y: y_batch, self.sequence_length: seqlen_batch, self.lr: lr})
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
			senvec.append(np.zeros(self.inp_vec_length))
		### predict
		pred = self.sess.run(self.pred, feed_dict = {self.x: [senvec], self.sequence_length: [seqlen]})
		print(pred)
		res = {'positive': pred[0][0], 'negative': pred[0][1]}
		return res
