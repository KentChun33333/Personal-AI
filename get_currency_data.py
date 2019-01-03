import requests
import re
import pandas as pd
import datetime 

# Script
# Global Variable 
last_date_file = 'currency_data/last_date.txt'
date_index = u'\u65e5\u671f'
saveH5_add = 'currency_data/currency.h5'
COL = [u'\u65e5\u671f', u'\u7f8e\u5143\uff0f\u65b0\u53f0\u5e63', u'\u4eba\u6c11\u5e63\uff0f\u65b0\u53f0\u5e63', u'\u6b50\u5143\uff0f\u7f8e\u5143', u'\u7f8e\u5143\uff0f\u65e5\u5e63', u'\u82f1\u938a\uff0f\u7f8e\u5143', u'\u6fb3\u5e63\uff0f\u7f8e\u5143', u'\u7f8e\u5143\uff0f\u6e2f\u5e63', u'\u7f8e\u5143\uff0f\u4eba\u6c11\u5e63', u'\u7f8e\u5143\uff0f\u5357\u975e\u5e63', u'\u7d10\u5e63\uff0f\u7f8e\u5143']
web_url = 'http://www.taifex.com.tw/cht/3/dailyFXRate'



def rebase_from_csv(address, saveH5_add):
    '''Download the CSV and rebase the h5'''
    df = pd.read_csv(address)
    df.columns = COL
    df[date_index]= pd.to_datetime(df[date_index], format= '%Y/%m/%d')
    df = df.convert_objects(convert_numeric=True)
    df.to_hdf(saveH5_add, 'df', mode='w')


def get_currency_table(backtrack_days=180, end_date=None):
    '''
    Args: 
       backtrack_days : how many days you want to craw, default 180
       end_date       : default today 
    '''
    if backtrack_days > 365:
        raise Exception('this api is limit for 1 year, ' 
                     'please specify end_date ex 2019/01/23, '
                     'default is today')
    if end_date==None:
        end_date = datetime.datetime.now().date()
    start_date = end_date - datetime.timedelta(days=backtrack_days)
    
    data={
    'queryStartDate': str(start_date).replace("-","/"), 
    'queryEndDate':  str(end_date).replace("-", "/"),
    }

    # address
    pageHTML = requests.post(web_url,  data=data)
    # get html-table obj (string)
    pageHTML = pageHTML.content.decode('utf-8').split('TBODY')[1]
    # split table, get a list-obj
    pageHTML = pageHTML.split('<tr>')
    # remove <tag> in all items in list 
    pageHTML = list(map(lambda x : re.sub('<.*?>', '', x), pageHTML))

    # -----------------------
    col = pageHTML.pop(0) # colume , due to different struce 
    col = col.split('\n')      # remove <> and get list-obj
    col = col[2:13]            # slice, due to some null obj in list
    col = list(map(lambda x:x.strip(), col) )
    validate_col(col)
    
    # -----------------------
    df = pd.DataFrame(columns=col) # creat df-obj

    for i in range(len(pageHTML)):
        row = pageHTML[i]
        row = row.split('\n')
        row = row[1:len(col)+1]
        row = list(map(lambda x: x.strip(), row))
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
    return datetime.datetime.strptime( str_time, '%Y-%m-%d %H:%M:%S')


def filt_df(df, last_date):
    df = df[ df[date_index] > last_date ]
    return df

def save_pair(df, path):
    # Append; an existing file is opened for reading and writing, 
    # and if the file does not exist it is created.
    df.to_hdf(saveH5_add, 'df', format='table', append=True)
    # ,format='t'
    ref = str(list(df[date_index])[-1])
    with open(path,'w') as f:
        f.write(ref)
    print ('[*] Updated the Currency Data')


if __name__=='__main__':

    df = get_currency_table()
    last_date = get_last_date(last_date_file)
    print('->', last_date)
    df = filt_df(df, last_date)
    # update new data
    if len(df)>0:
        save_pair(df, last_date_file)







