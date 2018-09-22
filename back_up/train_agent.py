
import pandas as pd
import tensorflow as tf
import numpy as np
from strategy import Strategy

address = 'currency_data/currency.h5'

def gen_train_pair(address, length, batch_size):
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
        data.append(df[nife:(nife+(2*length))]) # (batch, sequence, cols)
        if len(data)==batch_size:
            data = np.array(data)
            batch_x = data[: ,:length ,:]
            batch_y = data[: ,length: ,:]
            data = []
            yield batch_x, batch_y



def get_estimator():
    pass


if __name__ == '__main__':
    address = 'currency_data/currency.h5'
    length = 15
    cols = 9
    batch_size = 26
    # try using different optimizers and different optimizer configs
    A = Strategy(length ,batch_size, 0.001  )
    model = A.cnn_bilstim(length, cols)
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

    print('Train...')
    model.compile(loss='categorical_crossentropy', optimizer='adadelta')

    model.fit_generator(gen_train_pair(address, length, batch_size),
    samples_per_epoch = 60000, nb_epoch = 2, verbose=2,
    show_accuracy=True, callbacks=[],
    validation_data=None, class_weight=None, nb_worker=1)


