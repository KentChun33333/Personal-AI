import tensorflow as tf 

import keras
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, Bidirectional, Convolution1D, MaxPooling1D
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing import sequence
from keras.models import Sequential

#  parallel training models
# (batch, sequence, cols)
class CurrencyStrategy():
    def __init__(self):
        pass

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

    def yolo_small(self):
        S, B, C, W, H = self.S, self.B, self.C, self.W, self.H
        model = Sequential()

        model.add(Convolution2D(64, 7, 7, input_shape=(H,W,3), 
            border_mode='same' , subsample=(2,2)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same' , 
            strides=(2,2)))


        model.add(Convolution2D(192, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2) ))


        model.add(Convolution2D(128, 1, 1, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        
        model.add(Convolution2D(256, 3, 3, border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))   


        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(256, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))  

        model.add(Convolution2D(512, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(512, 1, 1,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))
        
        model.add(Convolution2D(1024, 3, 3, border_mode='same',
            subsample=(2,2)))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3,border_mode='same'))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Convolution2D(1024, 3, 3, border_mode='same'))
        model.add(BatchNormalization(mode=2))
        model.add(LeakyReLU(alpha=0.1))
        
        model.add(Flatten())
        model.add(Dense(512))
        model.add(BatchNormalization(mode=2))
        model.add(LeakyReLU(alpha=0.1))

        model.add(Dense(4096))
        model.add(BatchNormalization(mode=2))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.5))  

        model.add(Dense(S*S*(5*B+C), activation='linear'))

        model.summary()
        return model