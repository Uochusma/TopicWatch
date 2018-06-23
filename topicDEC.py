from textSAE import DeepAutoEncoder, \
    AutoEncoderStack01, AutoEncoderStack02, AutoEncoderStack03, \
    AutoEncoderStack04

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
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Reshape, Embedding, Flatten, Dropout, RepeatVector
from keras.layers import Conv1D,MaxPooling1D
from keras.layers import Concatenate
from keras.layers import LSTM, Bidirectional
from keras.optimizers import RMSprop,Adam
from keras.utils import multi_gpu_model
from matplotlib import pyplot as plt
#from keras.utils.vis_utils import plot_model
from keras import backend as K
dense1Unit = 500
dense2Unit = 500
dense3Unit = 2000
dense4Unit = 10
class DeepAutoEncoder(object):
    def __init__(self):
        input_text = Input(shape=(maxWordNum, wordcnt))#level0
        dense1 = Dense(dense1Unit,activation='relu')(input_text)#level1
        dense2 = Dense(dense2Unit,activation='relu')(dense1)#level2
        dense3 = Dense(dense3Unit,activation='relu')(dense2)#level3
        #
        dense4 = Dense(dense4Unit,activation='relu')(dense3)#level4
        encoded = dense4
        #
        undense4 = Dense(dense3Unit,activation='relu')(dense4)#level3
        undense3 = Dense(dense2Unit,activation='relu')(undense4)#level2
        undense2 = Dense(dense1Unit,activation='relu')(undense3)#level1
        decoded = Dense(wordcnt,activation='relu')(undense2)#level0
        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        self.autoencoder.summary()

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,epochs=nb_epoch,batch_size=batch_size,shuffle=shuffle,validation_data=(x_test, x_test))
        self.encoder.save('./save_data/full_model_encoder.h5')
        self.autoencoder.save('./save_data/full_model_autoencoder.h5')

    def load_weights(self, ae01, ae02, ae03, ae04):
        self.autoencoder.layers[1].set_weights(ae01.layers[1].get_weights())
        self.autoencoder.layers[2].set_weights(ae02.layers[1].get_weights())
        self.autoencoder.layers[3].set_weights(ae03.layers[1].get_weights())
        self.autoencoder.layers[4].set_weights(ae04.layers[1].get_weights())
        #
        self.autoencoder.layers[5].set_weights(ae04.layers[3].get_weights())
        self.autoencoder.layers[6].set_weights(ae03.layers[3].get_weights())
        self.autoencoder.layers[7].set_weights(ae02.layers[3].get_weights())
        self.autoencoder.layers[8].set_weights(ae01.layers[3].get_weights())

class AutoEncoderStack01(object):
    def __init__(self):
        input_text = Input(shape=(maxWordNum, wordcnt))#level0
        dense1 = Dense(dense1Unit,activation='relu')(input_text)#level1
        dense1 = Dropout(0.2)(dense1)
        encoded = dense1

        decoded = Dense(wordcnt, activation='relu')(dense1)#

        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.001, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        self.autoencoder.summary()

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             epochs=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/stack01_encoder.h5')
        self.autoencoder.save('./save_data/stack01_autoencoder.h5')
#
class AutoEncoderStack02(object):
    def __init__(self):
        input_text = Input(shape=(maxWordNum,dense1Unit))#level1
        dense2 = Dense(dense2Unit,activation='relu')(input_text)#level2
        dense2 = Dropout(0.2)(dense2)
        encoded = dense2

        decoded = Dense(dense1Unit, activation='relu')(dense2)#
        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.0005, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        self.autoencoder.summary()

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             epochs=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/stack02_encoder.h5')
        self.autoencoder.save('./save_data/stack02_autoencoder.h5')
#
class AutoEncoderStack03(object):
    def __init__(self):
        input_text =  Input(shape=(maxWordNum,dense2Unit))#level2
        dense3 = Dense(dense3Unit,activation='relu')(input_text)#level3
        dense3 = Dropout(0.2)(dense3)
        encoded = dense3

        decoded = Dense(dense2Unit, activation='relu')(dense3)#
        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.0005, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        self.autoencoder.summary()

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             epochs=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/stack03_encoder.h5')
        self.autoencoder.save('./save_data/stack03_autoencoder.h5')
#
class AutoEncoderStack04(object):
    def __init__(self):
        input_text =  Input(shape=(maxWordNum,dense3Unit))#level3
        dense4 = Dense(dense4Unit,activation='relu')(input_text)#level4
        dense4 = Dropout(0.2)(dense4)
        encoded = dense4

        decoded = Dense(dense3Unit, activation='relu')(dense4)#
        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.0005, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        self.autoencoder.summary()

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        self.autoencoder.fit(x_train, x_train,
                             epochs=nb_epoch,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             validation_data=(x_test, x_test))

        self.encoder.save('./save_data/stack04_encoder.h5')
        self.autoencoder.save('./save_data/stack04_autoencoder.h5')


x_train1 = ttc[10:900]
x_test1 = ttc[0:10]
epoch = 20
print("***** STEP 1 *****")
ae01 = AutoEncoderStack01()
ae01.compile()
ae01.train(x_train=x_train1, x_test=x_test1, nb_epoch=epoch, batch_size=128)

enc_train1 = ae01.encoder.predict(x=x_train1)
enc_test1 = ae01.encoder.predict(x=x_test1)

np.save('train_stack01.npy', enc_train1)
np.save('test_stack01.npy', enc_test1)

del enc_train1, enc_test1

# step2
print("***** STEP 2 *****")
ae02 = AutoEncoderStack02()
ae02.compile()
x_train2 = np.load('train_stack01.npy')
x_test2 = np.load('test_stack01.npy')
ae02.train(x_train=x_train2, x_test=x_test2, nb_epoch=epoch, batch_size=128)

enc_train2 = ae02.encoder.predict(x=x_train2)
enc_test2 = ae02.encoder.predict(x=x_test2)

np.save('train_stack02.npy', enc_train2)
np.save('test_stack02.npy', enc_test2)

del x_train2, x_test2, enc_train2, enc_test2

# step3
print("***** STEP 3 *****")
ae03 = AutoEncoderStack03()
ae03.compile()
x_train3 = np.load('train_stack02.npy')
x_test3 = np.load('test_stack02.npy')
ae03.train(x_train=x_train3, x_test=x_test3, nb_epoch=epoch, batch_size=128)

enc_train3 = ae03.encoder.predict(x=x_train3)
enc_test3 = ae03.encoder.predict(x=x_test3)

np.save('train_stack03.npy', enc_train3)
np.save('test_stack03.npy', enc_test3)

del x_train3, x_test3, enc_train3, enc_test3

# step4
print("***** STEP 4 *****")
ae04 = AutoEncoderStack04()
ae04.compile()
x_train4 = np.load('train_stack03.npy')
x_test4 = np.load('test_stack03.npy')
ae04.train(x_train=x_train4, x_test=x_test4, nb_epoch=epoch, batch_size=128)

enc_train4 = ae04.encoder.predict(x=x_train4)
enc_test4 = ae04.encoder.predict(x=x_test4)

np.save('train_stack04.npy', enc_train4)
np.save('test_stack04.npy', enc_test4)

del x_train4, x_test4, enc_train4, enc_test4

print("phase ",phase," done", "=" * 60, time.ctime())
phase+=1
# step5
print("phase ",phase," start", "=" * 60, time.ctime())
print("***** STEP 5 *****")
stacked_ae = DeepAutoEncoder()
stacked_ae.load_weights(ae01=ae01.autoencoder, ae02=ae02.autoencoder, ae03=ae03.autoencoder, ae04=ae04.autoencoder)
stacked_ae.compile()
stacked_ae.train(x_train=x_train1, x_test=x_test1, nb_epoch=epoch, batch_size=128)
print("phase ",phase," done", "=" * 60, time.ctime())
phase+=1
#==
print("phase ",phase," start", "=" * 60, time.ctime())
testNum = 20
stacked_test = stacked_ae.encoder.predict(x=ttc[900:900+testNum])
print(np.shape(stacked_test))
np.save('test_stacked.npy', stacked_test)
from xMeans import XMeans
featureS = np.reshape(stacked_test, (testNum,maxWordNum*10))
x_means = XMeans(random_state = 1).fit(featureS)  
print(x_means.labels_)
print(x_means.cluster_centers_)
print(x_means.cluster_log_likelihoods_)
print(x_means.cluster_sizes_)
print("phase ",phase," done", "=" * 60, time.ctime())
phase+=1