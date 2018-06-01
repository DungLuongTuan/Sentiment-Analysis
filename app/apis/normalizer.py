"""
	normalize sentence before feeding to word tokenizer
	-	remove numbers: delete all numbers and all words that have at least 1 character is number
	-	remove symbols: delete all symbols in sentences
	-	remove duplication: combine all duplicated characters into one (ngooooooon -> ngon)
	-	separate punctuation: separate punctuation from word (I like cinamon. -> I like cinamon .)
	-	replace all acronyms of 'không'
"""

class Normalizer(object):
	def __init__(self):
		self.vi_al = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','À', 'Á', 'Â', 'Ã', 'È', 'É', 'Ê', 'Ì', 'Í', 'Ð', 'Ò', 'Ó', 'Ô', 'Õ', '×', 'Ù', 'Ú', 'Ý', 'à', 'á', 'â', 'ã', 'ä', 'ç', 'è', 'é', 'ê', 'ì', 'í', 'î', 'ï', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ù', 'ú', 'û', 'ý', 'Ă', 'ă', 'ą', 'Đ', 'đ', 'ē', 'ė', 'ę', 'Ĩ', 'ĩ', 'į', 'Ũ', 'ũ', 'ū', 'Ơ', 'ơ', 'Ư', 'ư','Ạ', 'ạ', 'Ả', 'ả', 'Ấ', 'ấ', 'Ầ', 'ầ', 'Ẩ', 'ẩ', 'Ẫ', 'ẫ', 'Ậ', 'ậ', 'Ắ', 'ắ', 'Ằ', 'ằ', 'Ẳ', 'ẳ', 'Ẵ', 'ẵ', 'Ặ', 'ặ', 'Ẹ', 'ẹ', 'Ẻ', 'ẻ', 'Ẽ', 'ẽ', 'Ế', 'ế', 'Ề', 'ề', 'Ể', 'ể', 'Ễ', 'ễ', 'Ệ', 'ệ', 'Ỉ', 'ỉ', 'Ị', 'ị', 'Ọ', 'ọ', 'Ỏ', 'ỏ', 'Ố', 'ố', 'Ồ', 'ồ', 'Ổ', 'ổ', 'Ỗ', 'ỗ', 'Ộ', 'ộ', 'Ớ', 'ớ', 'Ờ', 'ờ', 'Ở', 'ở', 'Ỡ', 'ỡ', 'Ợ', 'ợ', 'Ụ', 'ụ', 'Ủ', 'ủ', 'Ứ', 'ứ', 'Ừ', 'ừ', 'Ử', 'ử', 'Ữ', 'ữ', 'Ự', 'ự', 'Ỳ', 'ỳ', 'ỵ', 'Ỷ', 'ỷ', 'Ỹ', 'ỹ']
		self.punctuation = ['.', ',', '!', '?']

	def fit(self, *args, **kwargs):
		pass

	def transform(self, text, remove_numbers = True, remove_symbols = True, remove_duplications = True, separate_punct = True):
		### replace all newline symbols = .
		text = text.replace('\n', '.')
		### remove duplication
		text_ = text[0]
		for c in text:
			if (c != text_[-1]):
				text_ += c
		text = text_
		### separate punctuation
		text_ = text[0]
		for i in range(1, len(text)):
			if (text[i] not in self.vi_al):
				if (text[i] in self.punctuation):
					if (text[i-1] != ' '):
						text_ += ' '
				else:
					if (text[i-1] in self.vi_al[1:]):
						text_ += ' '
			else:
				if (text[i] in self.vi_al[1:]) and (text[i-1] not in self.vi_al):
					text_ += ' '
			text_ += text[i]
		text = text_
		### remove symbols
		text_= ''
		for c in text:
			if c in (self.vi_al + self.punctuation):
				text_ += c
		text = text_
		### remove all number
		if (remove_numbers == True):
			splited_sen = text.split(' ')
			processed_sen = []
			for word in splited_sen:
				processed_sen.append(word)
				for c in word:
					if c.isdigit():
						processed_sen = processed_sen[:-1]
						break
			text = ' '.join(processed_sen)
		if (len(text) == 0):
			return text
		### replace all acconyms of 'không'
		while (' k ' in text):
			text = text.replace(' k ', ' không ')
		while (' K ' in text):
			text = text.replace(' K ', ' Không ')
		while (' ko ' in text):
			text = text.replace(' ko ', ' không ')
		while (' Ko ' in text):
			text = text.replace(' Ko ', ' Không ')
		while (' KO ' in text):
			text = text.replace(' KO ', ' Không ')
		### normalize space characters
		text = text.replace('\t', ' ')
		text = text.replace('\r', ' ')
		while (len(text) != 0) and (text[0] == ' '):
			text = text[1:]
		while (len(text) != 0) and (text[-1] == ' '):
			text = text[:-1]
		while ('  ' in text):
			text = text.replace('  ', ' ')
		### return
		return text.lower()


	def predict(self, *args, **kwargs):
		pass