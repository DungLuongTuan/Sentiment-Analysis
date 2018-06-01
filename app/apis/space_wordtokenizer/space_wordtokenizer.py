"""
	word segmentation by space and ... that's it

"""

class SpaceWordTokenizer(object):
	def __init__(self):
		pass

	def fit(self, *args, **kwargs):
		pass

	def transform(self, text):
		tokens = text.split(' ')
		return tokens

	def predict(self, *args, **kwargs):
		pass