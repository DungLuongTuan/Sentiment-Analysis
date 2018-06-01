import numpy as np
import tensorflow as tf
from random import shuffle
import os
from datetime import datetime
"""
	filter = [(filter_height, num_filter), (filter_height, num_filter), ...]
	hidden_units = [1024, 1024, 512, 512]
"""

class CNNSentiment():
	def __init__(self, sess, max_step, inp_vec_length, num_class, filter_config, hidden_units, gpu_percent):
		self.max_step = max_step
		self.inp_vec_length = inp_vec_length
		self.num_class = num_class
		self.filter_config = filter_config
		self.hidden_units = hidden_units
		# config = tf.ConfigProto()
		# config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
		# self.sess = tf.Session(config = config)
		# self.sess = tf.InteractiveSession()
		self.sess = sess
		self.build_model()

	def build_model(self):
		self.init_placeholder()
		self.build_graph()
		self.sess.run(tf.global_variables_initializer())

	def init_placeholder(self):
		self.x = tf.placeholder(tf.float32, [None, self.max_step, self.inp_vec_length])
		self.y = tf.placeholder(tf.float32, [None, self.num_class])
		self.lr = tf.placeholder(tf.float32, None)
		self.batch_size = tf.shape(self.x)[0]

	def build_graph(self):
		### convolution parameters
		self.x_r = tf.reshape(self.x, [self.batch_size, self.max_step, self.inp_vec_length, 1])
		self.filters = []
		self.filter_heights = []
		self.output_dim_cnn = 0
		for config in self.filter_config:
			self.filters.append(tf.Variable(tf.truncated_normal(shape = [config[0], self.inp_vec_length, 1, config[1]]), name = 'filter_' + str(config[0])))
			self.filter_heights.append(config[0])
			self.output_dim_cnn += config[1]
		### build CNN layers
		conv_feature = tf.nn.conv2d(input = self.x_r, filter = self.filters[0], strides = [1, 1, 1, 1], padding = 'VALID')
		conv_feature_pool = tf.nn.max_pool(conv_feature, ksize = [1, self.max_step - self.filter_heights[0] + 1, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID')
		conv_feature_pool_r = tf.reshape(conv_feature_pool, [self.batch_size, -1])
		for index, _filter in enumerate(self.filters[1:]):
			conv = tf.nn.conv2d(input = self.x_r, filter = _filter, strides = [1, 1, 1, 1], padding = 'VALID')
			conv_pool = tf.nn.max_pool(conv, ksize = [1, self.max_step - self.filter_heights[index+1] + 1, 1, 1], strides = [1, 1, 1, 1], padding = 'VALID')
			conv_pool_r = tf.reshape(conv_pool, [self.batch_size, -1])
			conv_feature_pool_r = tf.concat([conv_feature_pool_r, conv_pool_r], axis = -1)
		### build DNN layers
		# input layer
		with tf.name_scope('input_layer'):
			w = tf.Variable(tf.truncated_normal(shape = [self.output_dim_cnn, self.hidden_units[0]]), name = 'weight_input_layer')
			b = tf.Variable(tf.truncated_normal(shape = [1, self.hidden_units[0]]), name = 'bias_input_layer')
			a = tf.tanh(tf.matmul(conv_feature_pool_r, w) + b)
		# hidden layers
		for index, num_hidden in enumerate(self.hidden_units[:-1]):
			with tf.name_scope('hidden_layer_' + str(index)):
				w = tf.Variable(tf.truncated_normal(shape = [num_hidden, self.hidden_units[index+1]]), name = 'weight_hidden_layer_' + str(index+1))
				b = tf.Variable(tf.truncated_normal(shape = [1, self.hidden_units[index+1]]), name = 'bias_hidden_layer_' + str(index + 1))
				a = tf.tanh(tf.matmul(a, w) + b)
		# output layer
		with tf.name_scope('output_layer'):
			w = tf.Variable(tf.truncated_normal(shape = [self.hidden_units[-1], self.num_class]), name = 'weight_output_layer')
			b = tf.Variable(tf.truncated_normal(shape = [1, self.num_class]), name = 'bias_output_layer')
			self.pred = tf.nn.softmax(tf.matmul(a, w) + b)
		# loss + optimazer
		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.pred), axis = 1), axis = 0)
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

	def shuffle(self, x, y):
		z = list(zip(x, y))
		shuffle(z)
		x, y = zip(*z)
		return x, y

	def evaluate(self, x_test, y_test):
		pred = self.sess.run(self.pred, feed_dict = {self.x: x_test})
		# return (np.sum(np.equal(np.argmax(pred, axis = 1), np.argmax(y_test, axis = 1)))/len(y_test))*100
		return np.sum(np.equal(np.argmax(pred, axis = 1), np.argmax(y_test, axis = 1)))

	def train_new_model(self, x_train, y_train, x_valid, y_valid, lr, batch_size, num_epochs, save_path):
		accuracy = 0
		saver = tf.train.Saver(max_to_keep = 1000)
		for epoch in range(num_epochs):
			### shuffle all training data
			x, y = self.shuffle(x_train, y_train)
			sum_loss = 0
			start_time = datetime.now()
			while (len(x) != 0):
				### get training batch
				upperbound = min(batch_size, len(x))
				x_batch = x[:upperbound]
				y_batch = y[:upperbound]
				x = x[upperbound:]
				y = y[upperbound:]
				_loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.x: x_batch, self.y: y_batch, self.lr: lr})
				sum_loss += _loss
			### evaluate on valid set
			f1_score = self.evaluate(x_valid, y_valid, seqlen_valid)
			print('epoch: ', epoch, ' loss: ', sum_loss/(int((len(x_train)-1)/batch_size) + 1), ' f1_score: ', f1_score, ' time: ', str(datetime.now() - start_time))
			if (f1_score > accuracy):
				accuracy = f1_score
				saver.save(self.sess, save_path + '/model.ckpt')

	def train_new_partial_model(self, x_train, y_train, lr, batch_size):
		x, y = self.shuffle(x_train, y_train)
		sum_loss = 0
		while (len(x) != 0):
			### get training batch
			upperbound = min(batch_size, len(x))
			x_batch = x[:upperbound]
			y_batch = y[:upperbound]
			x = x[upperbound:]
			y = y[upperbound:]
			_loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict = {self.x: x_batch, self.y: y_batch, self.lr: lr})
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
		for token in tokens:
			senvec.append(word2vec[token])
		senvec = senvec[:min(len(senvec), self.max_step)]
		while (len(senvec) < self.max_step):
			senvec.append(np.zeros(self.inp_vec_length))
		### predict
		pred = self.sess.run(self.pred, feed_dict = {self.x: [senvec]})
		print(pred)
		res = {'positive': pred[0][0], 'negative': pred[0][1]}
		return res