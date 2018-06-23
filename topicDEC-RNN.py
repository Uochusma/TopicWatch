from textSAE import DeepRecurrentAutoEncoder

import os
import pandas as pd
from IPython.display import display
import numpy as np
import sys
import MeCab
import time
starttime = time.ctime()
phase = 1
print("phase ",phase," start", "=" * 60, time.ctime())
df1 = pd.read_csv('~/notebooks/yoshino/Data/Discussion/TravelPlan-Facilitated-Eval-Full.csv', sep=',')
df2 = pd.read_csv('~/notebooks/yoshino/Data/Discussion/161228_ana.csv', sep=',')
argumentList = [df1,df2]
combed_df = pd.concat([df1, df2])
combed_df = combed_df.dropna(subset=['body'])
print('shape',combed_df.shape)
#display(df1.head(3))
print("phase ",phase," done", "=" * 60, time.ctime())
phase+=1
#=======================
print("phase ",phase," start", "=" * 60, time.ctime())
import re
# テキスト加工
def chgstr(string):
    r1 = re.compile('[a-z0-9]{24}')
    string = r1.sub('0' * 24, string)

    # !"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t
    r2 = re.compile(r'[!"#$%&()*+,-./:;<=>?@^_`{|}~]')
    string = r2.sub('。。', string)
    string = string.replace('→', 'b')

    string = string.replace('\n', '。。')
    string = string.replace('\t', '、、')

    r3 = re.compile(r'[【】「」“”—＝．～：；『』＋＜＝≠＞×■▲△○●↑←…]')
    string = r3.sub('。。', string)
    string = string.replace('①', '1')
    string = string.replace('②', '2')
    string = string.replace('③', '3')
    string = string.replace('④', '4')
    string = string.replace('⑤', '5')
    string = string.replace('⑥', '6')

    return string
# MeCabで分かち書き
def mecab_make(threadstring):
    mecab_list = []
    m = MeCab.Tagger("-Owakati")
    #print(threadstring)
    threadstring_ = chgstr(threadstring)
    mecab_string = m.parse(threadstring_)
    for t in m.parse(threadstring_).split():
        mecab_list.append(t)
    return mecab_list, mecab_string
#
vocabList = set()
textMat = []
def df2Matrix3D(aData):#発言数x最大発言長x語彙数
    wordNum = []
    for index, row in aData.iterrows():
        text = row['body']
        word_list,word_string = mecab_make(text)
        for word in word_list:
            vocabList.add(word)
        wordNum.append(len(word_list))
        textMat.append(word_list)
    return(max(wordNum))

def df2Matrix4D(aData):#議論数x最大発言数x最大発言長x語彙数
    wordNum = []
    textNum = []
    for df in aData: 
        mat = []
        for index, row in df.iterrows():
            text = row['body']
            word_list,word_string = mecab_make(text)
            mat.append(word_list)
            for word in word_list:
                vocabList.add(word)
            wordNum.append(len(word_list))
        textNum.append(len(df.index))
        textMat.append(mat)
    return(max(wordNum),max(textNum))
maxWordNum = df2Matrix3D(combed_df)
#maxWordNum,maxTextNum = df2Matrix4D()
vocabList = list(vocabList)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
from keras.utils.np_utils import to_categorical

t = Tokenizer()
trainlist = []
wordcnt = 0
def texts2train3D():
    global wordcnt
    t.fit_on_texts(vocabList)  # 全体の語彙を使用してTokenizerを実行
    for i, text in enumerate(textMat):
        traintts = t.texts_to_sequences(text)  # テキストの順番
        while(traintts.count([]) != 0):
                traintts.remove([])
        trainttm = t.texts_to_matrix(text)  # 表記
        trainttm2 = t.texts_to_matrix(text,mode='tfidf') # 表記
        print(trainttm2)
        maxlen = len(traintts)  # 議論スレッドの長さ（単語延べ数）
        wordcnt = len(trainttm[0])  # 語彙サイズ（単語の異なり数）
        #print(i, traintts, wordcnt)
        traintoc = to_categorical(traintts, wordcnt)  # Onehot配列作成
        trainnp = np.zeros((maxWordNum - maxlen, wordcnt), 'int8')  # 議論スレッドの長さに満たない分の配列を作成する
        trainlist.append(np.append(traintoc.astype(np.int8), trainnp, axis=0))  # Onehot配列と満たない分の配列を結合して訓練データに追加する

    return i + 1
def texts2train4D():
    global wordcnt
    t.fit_on_texts(vocabList)  # 全体の語彙を使用してTokenizerを実行
    for i, argument in enumerate(textMat):
        for text in argument:
            traintts = t.texts_to_sequences(text)  # テキストの順番
            #traintts.remove([])
            #traintts = [seq for seq in traintts1 if len(seq)>0]
            #print(traintts.count([]))
            while(traintts.count([]) != 0):
                traintts.remove([])
            trainttm = t.texts_to_matrix(text)  # 表記
            maxlen = len(traintts)  # 議論スレッドの長さ（単語延べ数）
            wordcnt = len(trainttm[0])  # 語彙サイズ（単語の異なり数）
            #print(i, traintts, wordcnt)
            traintoc = to_categorical(traintts, wordcnt)  # Onehot配列作成
            trainnp = np.zeros((maxWordNum - maxlen, wordcnt), 'int8')  # 議論スレッドの長さに満たない分の配列を作成する
            trainlist.append(np.append(traintoc.astype(np.int8), trainnp, axis=0))  # Onehot配列と満たない分の配列を結合して訓練データに追加する
        print("argument:", i, "done...")

    return i + 1

argumentCount = texts2train3D()
print("reshape開始")
ttc = np.reshape(trainlist, (argumentCount,maxWordNum,wordcnt))
#ttc = np.reshape(trainlist, (argucount, maxTextNum, maxWordNum,wordcnt))
print(np.shape(ttc))
print("reshape終了")
print("訓練データ件数", len(trainlist), "文章の長さ", len(trainlist[0]), "語彙数", len(trainlist[0][0]))
print("phase ",phase," done", "=" * 60, time.ctime())
phase+=1
#===================================================================================================
print("phase ",phase," start", "=" * 60, time.ctime())
import tensorflow as tf
from keras.backend import tensorflow_backend
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)
#
x_train1 = ttc[10:900]
x_test1 = ttc[0:10]
epoch = 20
#
print("phase ",phase," done", "=" * 60, time.ctime())
phase+=1
# step5
print("phase ",phase," start", "=" * 60, time.ctime())
drae = DeepRecurrentAutoEncoder(maxWordNum,wordcnt,100)
drae.compile()
drae.train(x_train=x_train1, x_test=x_test1, nb_epoch=epoch, batch_size=8)
print("phase ",phase," done", "=" * 60, time.ctime())
phase+=1
#==
print("phase ",phase," start", "=" * 60, time.ctime())
testNum = 20
stacked_test = drae.encoder.predict(x=ttc[900:900+testNum])
print(stacked_test)
print(np.shape(stacked_test))
np.save('test_stacked.npy', stacked_test)
from sklearn.cluster import KMeans
cls1 = KMeans(n_clusters=2)
fit = cls1.fit(featureS)
print(fit)
pred = cls1.fit_predict(featureS)
print(pred)
rint("phase ",phase," done", "=" * 60, time.ctime())
phase+=1
#===================================================================================================
print("phase ",phase," start", "=" * 60, time.ctime())
from xMeans import XMeans
featureS = stacked_test
x_means = XMeans(random_state = 1).fit(featureS)  
print(x_means.labels_)
print(x_means.cluster_centers_)
print(x_means.cluster_log_likelihoods_)
print(x_means.cluster_sizes_)
print("phase ",phase," done", "=" * 60, time.ctime())
phase+=1