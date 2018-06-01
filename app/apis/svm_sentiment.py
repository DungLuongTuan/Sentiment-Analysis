from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np 
import pickle

class SVMSentiment():
	def __init__(self):
		pass

	def load_model(self, path):
		f = open(path, 'rb')
		self.clf = pickle.load(f)
		f.close()

	def transform(self, text, bow, tokenizer, normalizer):
		text = normalizer.transform(text)
		tokens = tokenizer.transform(text)
		senvec = bow.transform(tokens)
		probas = self.clf.predict_proba([senvec])[0]
		res = {'positive': probas[1], 'negative': probas[0]}
		return res



