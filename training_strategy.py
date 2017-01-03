
import pandas as pd 
import tensorflow as tf 
import numpy as np


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


address = 'currency_data/currency.h5'














