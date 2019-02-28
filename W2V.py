from gensim.models import word2vec
import jieba
import pandas as pd
import tensorflow as tf
import re
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
from langconv import *
#用于简繁字体转换（https://blog.csdn.net/tab_space/article/details/50823073）

import jieba.analyse





comments = pd.read_csv('DM/1.csv')  

comments.Score.replace([10.0, 20.0, 30.0, 40.0, 50.0], [1,2,3,4,5], inplace = True) 






voca = set()

def stopwordslist(filepath):  
    stopwords = [t2s(line.strip()) for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  



def add_voca(text):
	global voca, stopwords
	text = text.replace('。', '').replace('，', '').replace('－','').replace(' 】','').replace('【','')
	sentences.append(text)
	words = jieba.cut(text)
	col = []
	for word in words:
		if word not in stopwords:
			col.append(word)
			if word not in voca:
				voca.add(word)
	words_rows.append('/'.join(col))


def set_voca():
	global voca
	for idx, row in comments.iterrows():
		add_voca(t2s(row.CommentText))
	with open('w2v/seg.txt', 'w', encoding = 'utf-8') as f:
		for sen in sentences:
			words = jieba.cut(sen)
			for word in words:
				f.write(word + ' ')
			f.write('\n')



def t2s(text):

    text = Converter('zh-hans').convert(text)
    return text


stopwords = stopwordslist('w2v/stop_words.txt')
sentences = []
words_rows = []


set_voca()
#使用jieba.analyze分析

segs1 = []
segs2 = []
segs3 = []

#负面情感
index = comments.CommentText[comments.Score == 1.0].index
segs1.extend(pd.Series(words_rows).loc[index].tolist())    
index = comments.CommentText[comments.Score == 2.0].index
segs1.extend(pd.Series(words_rows).loc[index].tolist())  
seg1str = '/'.join(segs1)
keywords1 = jieba.analyse.extract_tags(seg1str, topK = 30, withWeight = True) 
keywords1_adj_v = jieba.analyse.extract_tags(seg1str, topK = 20, withWeight = True, allowPOS = ('adj', 'v')) 

#中立情感
index = comments.CommentText[comments.Score == 3.0].index
segs2.extend(pd.Series(words_rows).loc[index].tolist())
seg2str = '/'.join(segs2)

keywords2 = jieba.analyse.extract_tags(seg2str, topK = 30, withWeight = True) 
keywords2_adj_v = jieba.analyse.extract_tags(seg2str, topK = 20, withWeight = True, allowPOS = ('adj', 'v')) 


#正面情感
index = comments.CommentText[comments.Score == 4.0].index
segs3.extend(pd.Series(words_rows).loc[index].tolist())
index = comments.CommentText[comments.Score == 5.0].index
segs3.extend(pd.Series(words_rows).loc[index].tolist())
seg3str = '/'.join(segs3)
keywords3 = jieba.analyse.extract_tags(seg3str, topK = 30, withWeight = True) 
keywords3_adj_v = jieba.analyse.extract_tags(seg3str, topK = 20, withWeight = True, allowPOS = ('adj', 'v')) 
#使用Word2Vec分析
def train_w2v():
	sens = word2vec.Text8Corpus('w2v/seg.txt') 
	model = word2vec.Word2Vec(sens, size=25) 
	model.save("w2v/test.model")


train_w2v()

model = word2vec.Word2Vec.load("w2v/test.model")
print(model)

words = list(model.wv.vocab)
X = model[model.wv.vocab]

from sklearn.manifold import TSNE
tsne = TSNE(n_components = 2)
y = tsne.fit_transform(X)

from matplotlib import font_manager

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

c= np.random.rand(y.shape[0]) 
plt.scatter(y[:,0], y[:,1], c = c) 
words = list(model.wv.vocab) 
for i, word in enumerate(words): 
   plt.annotate(word, xy = (y[i,0], y[i,1])) 
 




	
	
