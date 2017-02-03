

import numpy as np
import pandas as pd

 # import the necessary packages
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Flatten, Dropout, Dense, Permute, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop


#  parallel training models
class Strategy():
    def __init__(self, time_length, batch_size, learning_rate):
        self.time_length = time_length
        self.batch_size  = batch_size
        self.learning_rate = learning_rate

    def loss(self, truY, preY, batch_size, gamma=0.8):
        '''
        MSE-loss + gredient to have a favor 
        '''
        # 1D tensor for 15 dim as length
        gredient_matrix = []
        # 
        loss = tf.reduce_sum(tf.pow(truY - preY,2))
        return loss

    def cnn_bilstim(self, length, cols):
        model = Sequential()
        model.add(Convolution1D(64, 1, border_mode='same', input_shape=(length, cols)))
        model.add(BatchNormalization(mode=2))
        model.add(Convolution1D(64, 3, border_mode='same'))
        model.add(Convolution1D(64, 3, border_mode='same'))
        # model.add(Embedding(max_features, 128, input_length=maxlen))
        model.add(Convolution1D(192, 3, border_mode='same'))
        model.add(MaxPooling1D(pool_length = 2, stride = 2, border_mode='same'))
        model.add(Convolution1D(128, 1, border_mode='same'))
        model.add(Convolution1D(256, 3, border_mode='same'))
        model.add(Convolution1D(256, 3, border_mode='same'))
        model.add(MaxPooling1D(pool_length=2,stride = 2,border_mode='same'))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Bidirectional(LSTM(240)))
        model.add(Dropout(0.5))
        model.add(Dense(length * cols, activation='sigmoid'))
        model.summary()
        return model

    def gen_train_pair(self, address):
        '''
        Data Source Address
        '''
        length = self.time_length
        bs = self.batch_size
        # read table 
        df = pd.read_hdf(address)  

        # slice out col-0: date , col-11: no data only '-'
        df = df.ix[:,1:10].astype('float')
        df = np.array(df)    

        # numpy shuffle
        np.random.shuffle(df)    

        # sliding-slice (full-length = x+y)
        data = []
        for nife in range(len(df)-(2*length)):
            data.append(df[nife:(nife+(2*length))]) 
            # (batch, sequence, cols)
            if len(data)==bs:
                data = np.array(data)
                batch_x = data[: ,:length ,:]
                batch_y = data[: ,length: ,:]
                data = []
                yield batch_x, batch_y.reshape(self.batch_size ,self.time_length * 9)

    def build_model(self):
        n_hidden = 256
        n_timesteps = self.time_length
        nb_classes =  self.time_length * 9 

        model = Sequential()
        model.add(Convolution1D(32, 2, border_mode='same', input_shape=(n_timesteps, 9)))

        model.add(Convolution1D(32, 2, border_mode='same'))

        model.add(Activation('relu'))
        
        model.add(LSTM(n_hidden))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        
        rmsprop = RMSprop(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
        
        print model.summary()
        return model

    def build_model_2(self):
        n_hidden = 256
        n_timesteps = self.time_length
        nb_classes =  self.time_length * 9 

        model = Sequential()
        model.add(Convolution1D(32, 2, border_mode='same', input_shape=(n_timesteps, 9)))

        model.add(Convolution1D(32, 2, border_mode='same'))

        model.add(Activation('relu'))
        
        model.add(LSTM(n_hidden))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
        
        rmsprop = RMSprop(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
        
        print model.summary()
        return model



if __name__ == '__main__':
    C = Strategy(7,10,1e-5)
    model = C.build_model()
    for i in range(100):
        model.fit_generator(C.gen_train_pair('currency_data/currency.h5'),
                    samples_per_epoch=20, nb_epoch=3,verbose=2)
    for i in range(100):
        model.fit_generator(C.gen_train_pair('currency_data/currency.h5'),
                    samples_per_epoch=20, nb_epoch=3,verbose=2)
    for i in range(100):
        model.fit_generator(C.gen_train_pair('currency_data/currency.h5'),
                    samples_per_epoch=20, nb_epoch=3,verbose=2)
    rmsprop = RMSprop(lr=1e-6)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
    for i in range(100):
        model.fit_generator(C.gen_train_pair('currency_data/currency.h5'),
                    samples_per_epoch=20, nb_epoch=3,verbose=2)


"""
def scheduler(epoch):
    if epoch == 5:
        model.lr.set_value(.02)
    return model.lr.get_value()

change_lr = LearningRateScheduler(scheduler)

model.fit(x_embed, y, nb_epoch=1, batch_size = batch_size, show_accuracy=True,
       callbacks=[chage_lr])
"""