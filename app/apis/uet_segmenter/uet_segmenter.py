"""
	link download UET segmenter https://github.com/phongnt570/UETsegmenter
"""

import jnius_config
jnius_config.set_classpath('.', '/home/tittit/java/test/test/uetsegmenter.jar')
from jnius import autoclass, cast

class UETSegmenter(object):
	def __init__(self):
		self.String = autoclass('java.lang.String')
		UETSegmenter = autoclass("vn.edu.vnu.uet.nlp.segmenter.UETSegmenter")
		model_path = cast('java.lang.String', self.String("/home/tittit/java/UETsegmenter/models"))
		self.segmenter = UETSegmenter(model_path)

	def fit(self):
		pass

	def transform(self, text):
		_text = self.String(text.encode('utf-8'))
		s = self.segmenter.segment(_text)
		print(s)
		return s.split(' ')