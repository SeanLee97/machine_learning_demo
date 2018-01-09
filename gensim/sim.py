# !/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba, re, string, os, random
from gensim import corpora, models, similarities

CORPUS_DIR = './corpus'
RUNTIME_DIR = './runtime'

class Dataset(object):
	"""docstring for Dataset"""
	def __init__(self, dirname):
		super(Dataset, self).__init__()
		self.dirname = dirname
	def __iter__(self):
		for filename in os.listdir(self.dirname):
			if os.path.isfile(os.path.join(self.dirname, filename)):
				with open(os.path.join(self.dirname, filename), 'rb') as f:
					data = f.read()
					title = data[:data.decode().find('\r\n')]
					yield filename, title, data

class Corpus(object):
	"""docstring for Corpus"""
	def __init__(self, root_dir, dictionary):
		super(Corpus, self).__init__()
		self.root_dir = root_dir
		self.dictionary = dictionary
	def __iter__(self):
		for name, title, data in Dataset(self.root_dir):
			yield self.dictionary.doc2bow(jieba.cut(data, cut_all=False))  # doc 2 bow

def random_doc():
	name = random.choice(os.listdir(CORPUS_DIR))
	with open(CORPUS_DIR + '/' + name, 'rb') as f:
		data = f.read()
		return name, data
	

def main():
	texts = []
	if not os.path.exists(RUNTIME_DIR + '/dictionary.pt'):
		regex = re.compile("[^\u4e00-\u9f5aa-zA-Z0-9]")
		for name, title, data in Dataset(CORPUS_DIR):
			def etl(s):
				return regex.sub('', s)
			seg = filter(lambda x: len(x) > 0, map(etl, jieba.cut(data, cut_all=False)))
			texts.append(seg)
		# 词典 (id, word)
		dictionary = corpora.Dictionary(texts)
		# 去除低频词语(只出现一次)
		small_freq_ids = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq < 2 ]
		dictionary.filter_tokens(small_freq_ids)
		dictionary.compactify()
		dictionary.save(RUNTIME_DIR + '/dictionary.pt')
	else:
		dictionary = corpora.Dictionary.load(RUNTIME_DIR + '/dictionary.pt')

	# 得到了语料中每一篇文档对应的稀疏向量（这里是bow向量）；向量的每一个元素代表了一个word在这篇文档中出现的次数
	corpus = list(Corpus(CORPUS_DIR, dictionary)) # [(0, X), (1, XX), (2, XXX)] (词的ID, 词出现的次数)

	if not os.path.exists(RUNTIME_DIR + '/tfidf.pt'):
		# 对corpus中出现的每一个特征的IDF值的统计工作
		tfidf = models.TfidfModel(corpus)
		tfidf.save(RUNTIME_DIR + '/tfidf.pt')
	else:
		tfidf = models.TfidfModel.load(RUNTIME_DIR + '/tfidf.pt')

	# 得到整个corpus的TFIDF值
	corpus_tfidf = tfidf[corpus]
	'''
	# 计算指定词的tfidf值
	doc_bow = [(0, 1), (1, 1)]
	print tfidf[doc_bow] # [(0, 0.70710678), (1, 0.70710678)]
	'''
	# 通过TF-IDF获得LSI模型
	if not os.path.exists(RUNTIME_DIR + '/lsi.pt'):
		lsi_model = models.LsiModel(corpus_tfidf, id2word = dictionary, num_topics = 30)
		lsi_model.save(RUNTIME_DIR + '/lsi.pt')
	else:
		lsi_model = models.LsiModel.load(RUNTIME_DIR + '/lsi.pt')

	i = 0
	for t in lsi_model.print_topics(30):
		print('[topic#%s]: ' % i, t)
		i += 1

	# 
	index = similarities.MatrixSimilarity(lsi_model[corpus])

	name, doc = random_doc()
	vec_bow = dictionary.doc2bow(jieba.cut(doc, cut_all=False))
	# 
	vec_lsi = lsi_model[vec_bow]
	print(vec_lsi)
	sims = sorted(enumerate(index[vec_lsi]), key=lambda item: -item[1])
	print('top 10 similary notes:')
	print(sims[:10])

if __name__ == '__main__':
	main()
		
		