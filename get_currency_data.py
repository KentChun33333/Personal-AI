import requests
import re
import pandas as pd
from datetime import datetime

# -----------------------
# external data source 
# if there is anything changed, just check variables/ops in function
# hard-code due to we are not able to control the external factors

# -----------------------
# Global_Variables
last_date_file = 'currency_data/last_date.txt'
date_index = u'\u65e5\u671f'
saveH5_add = 'currency_data/currency.h5'
COL = [u'\u65e5\u671f', u'\u7f8e\u5143\uff0f\u65b0\u53f0\u5e63', u'\u4eba\u6c11\u5e63\uff0f\u65b0\u53f0\u5e63', u'\u6b50\u5143\uff0f\u7f8e\u5143', u'\u7f8e\u5143\uff0f\u65e5\u5e63', u'\u82f1\u938a\uff0f\u7f8e\u5143', u'\u6fb3\u5e63\uff0f\u7f8e\u5143', u'\u7f8e\u5143\uff0f\u6e2f\u5e63', u'\u7f8e\u5143\uff0f\u4eba\u6c11\u5e63', u'\u7f8e\u5143\uff0f\u5357\u975e\u5e63', u'\u7d10\u5e63\uff0f\u7f8e\u5143']


def rebase_from_csv(address, saveH5_add):
    '''Download the CSV and rebase the h5'''
    df = pd.read_csv(address)
    df.columns = COL
    df[date_index]= pd.to_datetime(df[date_index], format= '%Y/%m/%d')
    df = df.convert_objects(convert_numeric=True)
    df.to_hdf(saveH5_add, 'df', mode='w')



def get_currency_table():
    # address
    pageHTML = requests.get('http://www.taifex.com.tw/chinese/3/3_5.asp')
    # get html-table obj (string)
    pageHTML = pageHTML.content.decode('utf-8').split('TBODY')[1]
    # split table, get a list-obj
    pageHTML = pageHTML.split('<tr>')
    # remove <tag> in all items in list 
    pageHTML = map(lambda x : re.sub('<.*?>', '', x), pageHTML)

    # -----------------------
    col = pageHTML.pop(0) # colume , due to different struce 
    col = col.split('\n')      # remove <> and get list-obj
    col = col[2:13]            # slice, due to some null obj in list
    col = map(lambda x:x.strip(), col) 
    validate_col(col)
    
    # -----------------------
    df = pd.DataFrame(columns=col) # creat df-obj

    for i in range(len(pageHTML)):
        row = pageHTML[i]
        row = row.split('\n')
        row = row[1:len(col)+1]
        row = map(lambda x: x.strip(), row)
        df.loc[i] = row
    # 2016/11/7
    df[date_index]= pd.to_datetime(df[date_index], format= '%Y/%m/%d')
    return df

def validate_col(col):
    '''validate the col name before map external-name to internal-id'''
    assert col==COL

def get_last_date(path):
    with open(path,'r') as f:
        str_time = f.read()
    return datetime.strptime( str_time, '%Y-%m-%d %H:%M:%S')


def filt_df(df, last_date):
    df = df[ df[date_index] > last_date ]
    return df

def save_pair(df, path):
    # Append; an existing file is opened for reading and writing, 
    # and if the file does not exist it is created.
    df.to_hdf(saveH5_add, 'df', mode='a')
    # ,format='t'
    ref = str(list(df[date_index])[-1])
    with open(path,'w') as f:
        f.write(ref)
    print ('[*] Updated the Currency Data')

def updata_pair(df, path):
    '''Append, to_hd5 have some bugs'''
    # dtype error would stop the append
    df = df.convert_objects(convert_numeric=True)
    with pd.HDFStore(saveH5_add) as store:
        store.append('df',df)
    #store.close()

    ref = str(list(df[date_index])[-1])
    with open(path,'w') as f:
        f.write(ref)
    print ('[*] Updated the Currency Data')

if __name__=='__main__':
    df = get_currency_table()
    last_date = get_last_date(last_date_file)
    df = filt_df(df, last_date)
    # update new data
    if len(df)>0:
        updata_pair(df, last_date_file)


# isolated external data source and internal data source 
# if map fail => external data source change => trigger modification alert 
# 


# we actually make no impact to the market 
# so it is not nessary to use reinforcement learning, 
# but we could still use the reinforcement-framework as an high-level summary






