from textSAE import DeepAutoEncoder, \
    AutoEncoderStack01, AutoEncoderStack02, AutoEncoderStack03, \
    AutoEncoderStack04

import os
import pandas as pd
from IPython.display import display
import numpy as np
import sys
import MeCab

phase = 1
def main():
    print("phase ",phase," start", "=" * 60, time.ctime())
    df1 = pd.read_csv('~/notebooks/yoshino/Data/Discussion/TravelPlan-Facilitated-Eval-Full.csv', sep=',')
    df2 = pd.read_csv('~/notebooks/yoshino/Data/Discussion/TravelPlan-Facilitated-Eval-Full.csv', sep=',')
    df = pd.concat([df1, df2])
    #display(df1.head(3))
    print("phase ",phase," done", "=" * 60, time.ctime())
    phase+=1
    # step1
    print("phase ",phase," start", "=" * 60, time.ctime())
    # MeCabで分かち書き
    def mecab_make(threadstring):
        mecab_list = []
        m = MeCab.Tagger("-Owakati")
        threadstring_ = chgstr(threadstring)
        mecab_string = m.parse(threadstring_)
        for t in m.parse(threadstring_).split():
            mecab_list.append(t)
        return mecab_list, mecab_string

    def df2OneHot(aData)
         for index, row in aData.iterrows():
            text = row['body']

    print("phase ",phase," done", "=" * 60, time.ctime())
    phase+=1
    #========
    print("***** STEP 1 *****")
    ae01 = AutoEncoderStack01()
    ae01.compile()
    ae01.train(x_train=x_train1, x_test=x_test1, nb_epoch=100, batch_size=128)

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
    ae02.train(x_train=x_train2, x_test=x_test2, nb_epoch=100, batch_size=128)

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
    ae03.train(x_train=x_train3, x_test=x_test3, nb_epoch=100, batch_size=128)

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
    ae04.train(x_train=x_train4, x_test=x_test4, nb_epoch=100, batch_size=128)

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
    stacked_ae.train(x_train=x_train1, x_test=x_test1, nb_epoch=100, batch_size=128)


if __name__ == "__main__":
    main()