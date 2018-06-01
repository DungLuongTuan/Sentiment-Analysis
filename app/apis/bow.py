"""
	sentence to vector bag of word version
	-	tf_idf: use tf idf
	-	bag of word does not use the order of words in sentence as 1 feature => no need UNKNOWN????
"""
import numpy as np
import math

class BagOfWord(object):
	def __init__(self, dictionary_path, ngram = [1]):
		self.load_dictionary(dictionary_path)
		self.ngram = ngram

	def load_dictionary(self, path):
		f = open(path)
		self.dictionary = []
		self.df = []
		for row in f:
			splited_row = row[:-1].split('\t')
			self.dictionary.append(splited_row[0])
			self.df.append(float(splited_row[1]))
		f.close()

	def fit(self, *args, **kwargs):
		pass

	def transform(self, tokens, tf_idf = True):
		### caculate count vecterize
		res = np.zeros(len(self.dictionary))
		for n in self.ngram:
			for i in range(n, len(tokens) + 1):
				token = ' '.join(tokens[i-n:i])
				if (token in self.dictionary):
					res[self.dictionary.index(token)] += 1

		### if tf_idf = True, calculate tfidf
		if (tf_idf == True):
			max_res = max(res)
			if (max_res != 0):
				for i in range(len(res)):
					tf = res[i]/max_res
					idf = math.log10(1/self.df[i])
					res[i] = tf*idf
		return res

	def predict(self, *args, **kwargs):
		pass
