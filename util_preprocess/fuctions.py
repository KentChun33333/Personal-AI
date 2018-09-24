import pandas as pd 
import numpy as  np 


from sklearn.model_selection import KFold, StratifiedKFold



def read_data(data_dir='../PySCT_Demo/data/'):
    cols = ['trading date', 'traded sum stockshard' , 'traded sum price' , 
            'open price', 'highest proce', 'lowest price' , 
            'close price', 'price changed', 'trading numbers' ]

    df_list=[]
    for stock_id in os.listdir(data_dir):
        try:
            df = pd.read_csv(data_dir + stock_id, header=None, encoding = "ISO-8859-1" )
            df.columns = cols
            df['stock_id'] = stock_id.split('.')[0]
        except:
            print(stock_id)
        
    df_list.append(df)
    return pd.concat(df_list)


def up_within_Ndays(y, ndays=5):
    '''
    input with day-feature-sequence
    '''
    res = np.zeros(len(y))
    for i in range( len(y)):
        gap = min(ndays, i)+1
        res[i] = max(y[i: i+gap])
    return res

def kflod_trainer(X, y, model, nsplit=3):
    fold = StratifiedKFold(nsplit)
    for train_ind, valid_ind in fold.split(X, y):
        trX, trY = X[train_ind], y[train_ind]
        vaX, vaY = X[valid_ind], y[valid_ind]
        model.fit(trX, trY)
        pre_y = model.predict(vaX)
        loss = classification_report(vaY, pre_y, )
        print('----------------------------------------------------------------')
        print(loss)

def trend(x):
    '''
    show the delta-summary for an order sequence
    '''
    res = [0]
    for i in range(1, len(x)):
        if x[i] > x[i-1]  : res.append(res[i-1] + 1 )
        elif x[i] < x[i-1]:res.append(res[i-1] -1 )
        else              :res.append(res[i-1])
    return res 

def trend_2(x):
    '''
    simple up and down ==> this version favor to zero
    '''
    res = [0]
    for i in range(1, len(x)):
        if x[i] > x[i-1]  : res.append( 1 )
        elif x[i] < x[i-1]: res.append(0 )
        else              : res.append(0)
    return res 

def previous_Ndays_price(x, N):
    res = np.zeros(len(x))
    res[N:] = x[:-N]
    return res

def previous_Ndays_price_delta(x, N):
    res = np.zeros(len(x))
    res[N:] = x[:-N]
    return res - np.array(x)



def previous_Ndays_price_delta_percent(x, N):
    epison = 1e-7
    res = np.zeros(len(x))
    res[N:] = x[:-N]
    res -= np.array(x)
    res /= np.array(x)+np.array([epison]*len(x))
    return res














    