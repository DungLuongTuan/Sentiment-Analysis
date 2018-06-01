from app import app
from flask import render_template, request
from .apis import *
import fasttext as ft
import tensorflow as tf

#=============================================       PROCESS      ========================================
### define all parameters
dictionary_path = 'app/data/dictionary/10000_words_dictionary'
word2vec_path = '/home/tittit/word_segmentation/vi.bin'
svm_path = 'app/models/bow_svm/model_17-05-2018.pkl'
deep_cnn_lstm_path = 'app/models/fix/model183.ckpt'
cnn_path = 'app/models/cnn_deep_dnn/model282.ckpt'
lstm_path = 'app/models/deep_lstm/space_tokenizer/model15.ckpt'
lstm_segmented_path = 'app/models/deep_lstm/uet_tokenizer/model13.ckpt'
ngrams = [1, 2]
writer = tf.summary.FileWriter("./summary")
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1

### load all models
print('load normalizer model...')
normalizer = Normalizer()

print('load tokenizer model...')
space_tokenizer = SpaceWordTokenizer()
uet_tokenizer = UETSegmenter()

print('load BagOfWord model...')
bow = BagOfWord(dictionary_path, ngrams)

print('load Word2Vec model...')
word2vec = ft.load_model(word2vec_path)

print('load BOW + SVM model...')
svm_sentiment = SVMSentiment()
svm_sentiment.load_model(svm_path)

print('load CNN + DNN model...')
graph1 = tf.Graph()
sess1 = tf.Session(graph = graph1, config = config)
with graph1.as_default():
	cnn_sentiment = CNNSentiment(sess = sess1, max_step = 500, inp_vec_length = 100, num_class = 2, filter_config = [(3, 150), (5, 150), (7, 150)], hidden_units = [100, 100], gpu_percent = 0.1)
	cnn_sentiment.load_model(cnn_path)

print('load LSTM model...')
graph2 = tf.Graph()
sess2 = tf.Session(graph = graph2, config = config)
with graph2.as_default():
	lstm_sentiment = DeepLSTMSentiment(sess = sess2, gpu_percent = 0.1, n_hidden = 100, max_step = 500, num_layers = 3)
	lstm_sentiment.load_model(lstm_path)

print('load CNN + LSTM model...')
graph3 = tf.Graph()
sess3 = tf.Session(graph = graph3, config = config)
with graph3.as_default():
	deep_cnn_lstm_sentiment = CNN_LSTM_Sentiment(sess = sess3, max_step = 500, inp_vec_length = 100, num_class = 2, filter_config = [(3, 150), (5, 150), (7, 150)], lstm_n_hidden = 100, lstm_num_layers = 3, cnn_embed_size = 450, gpu_percent = 0.1)
	deep_cnn_lstm_sentiment.load_model(deep_cnn_lstm_path)

print('load UETsegment + LSTM model...')
graph4 = tf.Graph()
sess4 = tf.Session(graph = graph4, config = config)
with graph4.as_default():
	lstm_segmented_sentiment = DeepLSTMSentiment(sess = sess4, gpu_percent = 0.1, n_hidden = 100, max_step = 500, num_layers = 3)
	lstm_segmented_sentiment.load_model(lstm_segmented_path)

writer.add_graph(cnn_sentiment.sess.graph)
writer.add_graph(lstm_sentiment.sess.graph)
writer.add_graph(deep_cnn_lstm_sentiment.sess.graph)
writer.add_graph(lstm_segmented_sentiment.sess.graph)

### define functions

def get_response(text):
	result = {}
	print(normalizer.transform(text))
	print('predicting using SVM model...')
	result['svm_result'] = svm_sentiment.transform(text, bow, space_tokenizer, normalizer)
	print('predicting using CNN + DNN model...')
	result['cnn_result'] = cnn_sentiment.transform(text, word2vec, space_tokenizer, normalizer)
	print('predicting using LSTM model...')
	result['lstm_result'] = lstm_sentiment.transform(text, word2vec, space_tokenizer, normalizer)
	print('predicting using CNN + LSTM model...')
	result['cnn_lstm_result'] = deep_cnn_lstm_sentiment.transform(text, word2vec, space_tokenizer, normalizer)
	print('predicting using UET segmenter + LSTM model...')
	result['lstm_segmented_result'] = lstm_segmented_sentiment.transform(text, word2vec, uet_tokenizer, normalizer)

	result = str(result).replace("'", '"')
	return result


#============================================= SERVER APPLICATION ========================================
@app.route('/')
def home():
	return render_template('main_page.html')

@app.route('/', methods = ['POST'])
def get_request():
	text = dict(request.form)['req'][0]
	res = get_response(text)
	return res
