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
maxTextLen = 1000
wordcnt = 1000
dense1Unit = 500
dense2Unit = 500
dense3Unit = 2000
dense4Unit = 10
class DeepAutoEncoder(object):
    def __init__(self):
        input_text = Input(shape=(maxTextLen, wordcnt))#level0
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
        decoded = Dense(wordcn,activation='relu')(undense2)#level0
        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
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
        self.autoencoder.layers[10].set_weights(ae04.layers[4].get_weights())
        self.autoencoder.layers[12].set_weights(ae03.layers[4].get_weights())
        self.autoencoder.layers[14].set_weights(ae02.layers[4].get_weights())
        self.autoencoder.layers[16].set_weights(ae01.layers[4].get_weights())

class AutoEncoderStack01(object):
    def __init__(self):
        input_text = Input(shape=(maxTextLen, wordcnt))#level0
        dense1 = Dense(dense1Unit,activation='relu')(input_text)#level1
        dense1 = Dropout(0.2)(dense1)
        encoded = dense1

        decoded = Dense(wordcnt, activation='relu')(dense1)#

        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
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
        input_text = Input(shape=(maxTextLen,dense1Unit))#level1
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
        input_text =  Input(shape=(maxTextLen,dense2Unit))#level2
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
        input_text =  Input(shape=(maxTextLen,dense3Unit))#level3
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
#=============================================================================
class DeepRecurrentAutoEncoder(object):

    def __init__(self,maxTextLen,wordcnt,aFeature):
        hidden_dim = aFeature
        input_text = Input(shape=(maxTextLen, wordcnt))#level0
        #encode
        lstm1 = LSTM(hidden_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(input_text)
        encoded = lstm1
        #decode
        hidden = RepeatVector(maxTextLen)(encoded)
        decoded = LSTM(hidden_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden)
        decoded = Dense(hidden_dim, activation="relu")(decoded)
        decoded = Dense(wordcnt, activation="tanh")(decoded)
        self.encoder = Model(input=input_text,output=encoded)
        self.autoencoder = Model(input=input_text,output=decoded)
        return

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.0005, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        self.autoencoder.summary()

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        if(x_test is None):
            self.autoencoder.fit(x_train, x_train,epochs=nb_epoch,batch_size=batch_size,shuffle=shuffle,validation_split=0.1)
        else:
            self.autoencoder.fit(x_train, x_train,epochs=nb_epoch,batch_size=batch_size,shuffle=shuffle,validation_data=(x_test, x_test))
        self.encoder.save('./save_data/DRAE.h5')
        self.autoencoder.save('./save_data/DRAE.h5')
#=============================================================================
class DeepRecurrentConditionalAutoEncoder(object):

    def __init__(self):
        hidden_dim = 100
        input_text = Input(shape=(maxTextLen, wordcnt))#level0
        #encode
        lstm1 = LSTM(hidden_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=False)(input_text)
        encoded = lstm1
        #decode
        hidden = RepeatVector(maxTextLen)(encoded)
        reverse_input = Input(shape=(maxTextLen, wordcnt))
        hidden_revinput = Concatenate([hidden, reverse_input])
        decoded = LSTM(hidden_dim, activation="tanh", recurrent_activation="sigmoid", return_sequences=True)(hidden_revinput)
        decoded = Dense(hidden_dim, activation="relu")(decoded)
        decoded = Dense(wordcnt, activation="tanh")(decoded)
        self.encoder = Model(input=[input_text,reverse_input],output=encoded)
        self.autoencoder = Model(input=[input_text,reverse_input],output=decoded)
        return

    def compile(self, optimizer='adam', loss='mean_squared_error'):
        adam = Adam(lr=0.0005, decay=0.005)
        self.autoencoder.compile(optimizer=adam, loss=loss)
        self.autoencoder.summary()

    def train(self, x_train=None, x_test=None, nb_epoch=1, batch_size=128, shuffle=True):
        x_train_rev = x_train[:,::-1,:]
        if(x_test is None):
            self.autoencoder.fit([x_train,x_train_rev], x_train,epochs=nb_epoch,batch_size=batch_size,shuffle=shuffle,validation_split=0.1)
        else:
            X_test_rev = x_test[:,::-1,:]
            self.autoencoder.fit([x_train,x_train_rev], x_train,epochs=nb_epoch,batch_size=batch_size,shuffle=shuffle,validation_data=([x_test, X_test_rev], x_test))
        self.encoder.save('./save_data/DRAE.h5')
        self.autoencoder.save('./save_data/DRAE.h5')
#================================================================================
class DeepAutoEncoder(object):
    def __init__(self):
        input_text = Input(shape=(maxTextLen, wordcnt))#level0
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
        decoded = Dense(wordcn,activation='relu')(undense2)#level0
        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
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
        self.autoencoder.layers[10].set_weights(ae04.layers[4].get_weights())
        self.autoencoder.layers[12].set_weights(ae03.layers[4].get_weights())
        self.autoencoder.layers[14].set_weights(ae02.layers[4].get_weights())
        self.autoencoder.layers[16].set_weights(ae01.layers[4].get_weights())

class AutoEncoderStack01(object):
    def __init__(self):
        input_text = Input(shape=(maxTextLen, wordcnt))#level0
        dense1 = Dense(dense1Unit,activation='relu')(input_text)#level1
        dense1 = Dropout(0.2)(dense1)
        encoded = dense1

        decoded = Dense(wordcnt, activation='relu')(dense1)#

        self.encoder = Model(input=input_text, output=encoded)
        self.autoencoder = Model(input=input_text, output=decoded)

    def compile(self, optimizer='adam', loss='binary_crossentropy'):
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
        input_text = Input(shape=(maxTextLen,dense1Unit))#level1
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
        input_text =  Input(shape=(maxTextLen,dense2Unit))#level2
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
        input_text =  Input(shape=(maxTextLen,dense3Unit))#level3
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